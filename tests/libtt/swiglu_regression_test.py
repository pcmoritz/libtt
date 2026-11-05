"""Check fused SwiGLU decode projections across widths and reduction sizes."""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize(
    "rows,inner_size,width",
    [
        # One K tile, an odd K tile count, a prime output tile count, and a
        # partial core row.
        (1, 32, 6144),
        (1, 96, 6144),
        (1, 64, 3104),
        (1, 512, 256),
        # Several and all 32 rows of the activation tile.
        (2, 512, 12800),
        (32, 512, 6144),
        # Qwen3-8B TP1, Qwen3-32B TP4/TP2 and Qwen3-14B TP1 per-chip shapes;
        # 14B needs more output pairs per core than fit in DST at once.
        (1, 4096, 12288),
        (1, 5120, 6400),
        (1, 5120, 12800),
        (1, 5120, 17408),
        # Reductions far beyond typical hidden sizes.
        (1, 12800, 6144),
        (1, 16384, 3072),
    ],
)
def test_swiglu_projection_width(rows, inner_size, width, tmp_path):
    def swiglu(x, weights):
        weights = weights[None]
        weights = jax.ffi.ffi_call(
            "tt.weight_dtype_override",
            jax.ShapeDtypeStruct(weights.shape, weights.dtype),
            vmap_method="sequential",
        )(weights, **{"ttcore.weight_dtype": "bfp_bf8"})[0]
        up, gate = jnp.split(x @ weights, 2, axis=-1)
        return up * jax.nn.silu(gate)

    run = jax.jit(
        swiglu,
        compiler_options={
            "optimization_level": "O1",
            "enable_trace": "true",
            "export_path": str(tmp_path),
        },
    )
    device = jax.devices("tt")[0]
    rng = np.random.default_rng(6)
    for _ in range(3):
        # Exactly representable BFP8 weights keep this focused on the fusion.
        x = (rng.integers(-4, 5, (rows, inner_size)) / 16).astype(jnp.bfloat16)
        weights = (rng.integers(-4, 5, (inner_size, 2 * width)) / 16).astype(jnp.bfloat16)
        actual = run(jax.device_put(x, device), jax.device_put(weights, device))
        up, gate = np.split(x.astype(np.float32) @ weights.astype(np.float32), 2, axis=-1)
        expected = up * gate / (1 + np.exp(-gate))
        np.testing.assert_allclose(
            np.asarray(actual).astype(np.float32), expected, atol=0.01, rtol=0.04
        )

    # A numerical pass alone could also come from the unfused path.
    assert any(
        "ttnn.fused_swiglu" in path.read_text()
        for path in (tmp_path / "irs").glob("ttnn*.mlir")
    ), "projection did not select fused SwiGLU"


def test_swiglu_halves_of_partial_tiles_stay_unfused(tmp_path):
    """Halves that are not whole tiles cannot be split by the kernel: the
    projection must keep the separate matmul, split, silu and multiply. The
    weight is BFP8 like a fused one, and the total width of 100 pads to 128,
    which a check on the padded width would accept."""

    def swiglu(x, weights):
        weights = weights[None]
        weights = jax.ffi.ffi_call(
            "tt.weight_dtype_override",
            jax.ShapeDtypeStruct(weights.shape, weights.dtype),
            vmap_method="sequential",
        )(weights, **{"ttcore.weight_dtype": "bfp_bf8"})[0]
        up, gate = jnp.split(x @ weights, 2, axis=-1)
        return up * jax.nn.silu(gate)

    run = jax.jit(swiglu, compiler_options={"optimization_level": "O1", "export_path": str(tmp_path)})
    device = jax.devices("tt")[0]
    rng = np.random.default_rng(8)
    x = (rng.integers(-4, 5, (1, 512)) / 16).astype(jnp.bfloat16)
    weights = (rng.integers(-4, 5, (512, 100)) / 16).astype(jnp.bfloat16)
    actual = run(jax.device_put(x, device), jax.device_put(weights, device))
    up, gate = np.split(x.astype(np.float32) @ weights.astype(np.float32), 2, axis=-1)
    expected = up * gate / (1 + np.exp(-gate))
    np.testing.assert_allclose(np.asarray(actual).astype(np.float32), expected, atol=0.01, rtol=0.04)
    irs = list((tmp_path / "irs").glob("ttnn*.mlir"))
    assert irs, "no IR was exported"
    assert not any("ttnn.fused_swiglu" in path.read_text() for path in irs), "halves of partial tiles were fused"


def _silu_and_mul(y):
    """PyTorch's SiluAndMul as SGLang-TorchTPU traces it: the silu of the gate
    half in FP32, x / (1 + exp(-x)), cast back to BF16, times the up half."""
    gate, up = jnp.split(y, 2, axis=-1)
    gate = gate.astype(jnp.float32)
    return (gate / (1 + jnp.exp(-gate))).astype(jnp.bfloat16) * up


def _assert_exact_swiglu(tmp_path, op):
    irs = [path.read_text() for path in (tmp_path / "irs").glob("ttnn_[0-9]*.mlir")]
    assert any(re.search(rf'"ttnn\.{op}".*ttnn\.exact_swiglu', ir) for ir in irs), f"no exact SwiGLU {op}"
    assert not any('"ttnn.silu"' in ir for ir in irs), "silu was left unfused"


@pytest.mark.parametrize(
    "rows,inner_size,width,transposed",
    [
        # Qwen3-8B's gate and up projection on four chips and one, PyTorch's
        # [2N, K] weight, one and four tokens; a [K, 2N] weight.
        (1, 4096, 3072, True),
        (4, 4096, 3072, True),
        (1, 4096, 12288, True),
        (2, 1024, 2048, False),
    ],
)
def test_torch_silu_and_mul_projection(rows, inner_size, width, transposed, tmp_path):
    """PyTorch's merged gate and up projection and SiluAndMul (SGLang-TorchTPU's
    decode MLP) run as one exact SwiGLU decode matmul and equal the separate
    ops (the projection and SiluAndMul as two programs) bit for bit."""
    options = {"optimization_level": "O0"}
    project = (lambda x, w: x @ w.T) if transposed else (lambda x, w: x @ w)
    device = jax.devices("tt")[0]
    rng = np.random.default_rng(rows + width)
    x = jax.device_put(jnp.asarray(rng.normal(0, 1, (rows, inner_size)), jnp.bfloat16), device)
    shape = (2 * width, inner_size) if transposed else (inner_size, 2 * width)
    w = jax.device_put(jnp.asarray(rng.normal(0, 0.05, shape), jnp.bfloat16), device)
    fused = jax.jit(
        lambda x, w: _silu_and_mul(project(x, w)), compiler_options={**options, "export_path": str(tmp_path)}
    )
    want = jax.jit(_silu_and_mul, compiler_options=options)(jax.jit(project, compiler_options=options)(x, w))
    np.testing.assert_array_equal(np.asarray(fused(x, w), np.float32), np.asarray(want, np.float32))
    _assert_exact_swiglu(tmp_path, "matmul")


@pytest.mark.parametrize("rows,width", [(1, 3072), (4, 3072), (1, 12288)])
@pytest.mark.parametrize("scale_shape", ["column", "scalar"])
def test_torch_fp8_silu_and_mul_projection(rows, width, scale_shape, tmp_path):
    """The same after SGLang-TorchTPU's FP8 linear (libtt's fp8_linear_small_m:
    the official checkpoints' [2N, 1] weight scales, or online quantization's
    one): one exact SwiGLU scaled matmul, bit for bit."""
    from jax_plugins.libtt import sglang_tpu_kernels

    options = {"optimization_level": "O0"}
    device = jax.devices("tt")[0]
    rng = np.random.default_rng(rows + width)
    x = jax.device_put(jnp.asarray(rng.normal(0, 1, (rows, 4096)), jnp.bfloat16), device)
    w = jax.device_put(jnp.asarray(rng.normal(0, 30, (4096, 2 * width)), jnp.float8_e4m3fn), device)
    scale = rng.uniform(1e-4, 1e-3, (2 * width, 1) if scale_shape == "column" else (1,))
    s = jax.device_put(jnp.asarray(scale, jnp.float32), device)
    linear = sglang_tpu_kernels._fp8_linear_small_m
    fused = jax.jit(
        lambda x, w, s: _silu_and_mul(linear(x, w, s)), compiler_options={**options, "export_path": str(tmp_path)}
    )
    want = jax.jit(_silu_and_mul, compiler_options=options)(jax.jit(linear, compiler_options=options)(x, w, s))
    np.testing.assert_array_equal(np.asarray(fused(x, w, s), np.float32), np.asarray(want, np.float32))
    _assert_exact_swiglu(tmp_path, "scaled_matmul")
