"""The gated RMS norm of linear-attention layers, as SGLang-TorchTPU's Qwen3.5
and Qwen3.8 gated DeltaNet compute their output (PyTorch's pure RMSNormGated:
FP32 math on BF16 operands, silu as z / (1 + exp(-z))), runs as one
ttnn.rms_norm_gated and matches the FP32 math within a BF16 rounding; in the
decode layer's form it reads the FP32 recurrence output and z in place from
the qkvz projection, and makes the [tokens, heads * 128] rows out_proj
reads."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

EPS = 1e-6


def gated_norm(x, z, w):
    xf = x.astype(jnp.float32)
    var = jnp.mean(jax.lax.pow(xf, 2.0), axis=-1, keepdims=True)
    out = xf * jax.lax.rsqrt(var + EPS) * w.astype(jnp.float32)
    zf = z.astype(jnp.float32)
    return (out * (zf / (1 + jnp.exp(-zf)))).astype(x.dtype)


def reference(x, z, w):
    xf, zf, wf = (np.asarray(a, np.float64) for a in (x, z, w))
    rstd = 1 / np.sqrt(np.mean(xf * xf, axis=-1, keepdims=True) + EPS)
    return xf * rstd * wf * (zf / (1 + np.exp(-zf)))


@pytest.mark.parametrize("trace", [False, True])
@pytest.mark.parametrize(
    "shape",
    [
        # Qwen3.8-27B's 12 value heads per chip at TP4: one and four tokens.
        (12, 128),
        (48, 128),
        (1, 12, 128),
        # Narrower heads, rows across tile rows.
        (70, 64),
        # A speculative verify of four requests' 4-token windows, and a
        # 256-token prefill chunk.
        (16 * 12, 128),
        (256 * 12, 128),
    ],
)
def test_rms_norm_gated(trace, shape, tmp_path):
    run = jax.jit(
        gated_norm,
        compiler_options={"optimization_level": "O0", "enable_trace": str(trace).lower(), "export_path": str(tmp_path)},
    )
    device = jax.devices("tt")[0]
    rng = np.random.default_rng(len(shape) * 1000 + shape[-2])
    x = jnp.asarray(rng.normal(0, 1, shape) * np.exp2(rng.integers(-4, 4, shape[:-1] + (1,))), jnp.bfloat16)
    z = jnp.asarray(rng.normal(0, 2, shape), jnp.bfloat16)
    w = jnp.asarray(rng.normal(1, 0.2, shape[-1]), jnp.bfloat16)
    got = np.asarray(run(*(jax.device_put(a, device) for a in (x, z, w))), np.float64)
    want = reference(x, z, w)
    np.testing.assert_allclose(got, want, rtol=2**-7, atol=1e-6)
    irs = [path.read_text() for path in (tmp_path / "irs").glob("ttnn_*.mlir")]
    assert irs and any("ttnn.rms_norm_gated" in ir for ir in irs), "the gated norm was not fused"
    assert not any('"ttnn.rms_norm"' in ir for ir in irs), "an RMS norm ran as its own program"


HEADS, ALL_HEADS, DIM = 12, 32, 128


def gdn_output(core, qkvz, w):
    """As Qwen3.8's gated DeltaNet layer: core [T, 12, 128] (BF16 here: the
    RMS norm fusion narrows an FP32 x only where it can retype its producer,
    as it does the recurrence's), z heads 20 .. 31 of the [T, 32 * 128] qkvz
    projection."""
    tokens = core.shape[0]
    x = core.reshape(-1, DIM)
    z = qkvz.reshape(tokens, ALL_HEADS, DIM)[:, ALL_HEADS - HEADS :].reshape(-1, DIM)
    return gated_norm(x, z, w).reshape(tokens, HEADS * DIM)


@pytest.mark.parametrize("tokens", [1, 4])
def test_rms_norm_gated_decode_layer(tokens, tmp_path):
    run = jax.jit(
        gdn_output, compiler_options={"optimization_level": "O0", "export_path": str(tmp_path)}
    )
    device = jax.devices("tt")[0]
    rng = np.random.default_rng(tokens)
    core = jnp.asarray(rng.normal(0, 1, (tokens, HEADS, DIM)), jnp.bfloat16)
    qkvz = jnp.asarray(rng.normal(0, 2, (tokens, ALL_HEADS * DIM)), jnp.bfloat16)
    w = jnp.asarray(rng.normal(1, 0.2, DIM), jnp.bfloat16)
    got = np.asarray(run(*(jax.device_put(a, device) for a in (core, qkvz, w))), np.float64)
    z = np.asarray(qkvz, np.float64).reshape(tokens, ALL_HEADS, DIM)[:, ALL_HEADS - HEADS :]
    want = reference(core, z, w).reshape(tokens, HEADS * DIM)
    np.testing.assert_allclose(got, want, rtol=2**-7, atol=1e-6)
    (ir,) = [path.read_text() for path in (tmp_path / "irs").glob("ttnn_[0-9]*.mlir")]
    assert ir.count('"ttnn.rms_norm_gated"') == 1, "the gated norm was not fused"
    assert '"ttnn.slice_static"' not in ir, "z was sliced instead of read in place"


def test_scalar_multiply_compiles():
    # The fusion matches multiplies; a rank-0 one has no last dimension.
    run = jax.jit(lambda s: jnp.ones(()) * s)
    device = jax.devices("tt")[0]
    assert float(run(jax.device_put(np.float32(2), device))) == 2
