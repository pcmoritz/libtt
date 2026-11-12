"""tt.quantize_rows: per-row dynamic float8 quantization in one TTNN program.

Per row of x ([M, K], M <= 32): scale = max(amax(|x|) / fp8_max, min_scale)
and q = float8(x / scale). The results must equal those of the jnp ops that
SGLang-TorchTPU's FP8 linears ran before (abs, max, divide, maximum, typecasts)
bit for bit, and the IEEE float32 math up to ttnn.divide's last bit. The
patched fp8_linear_small_m must equal its jnp form.
"""

import re

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from jax_plugins.libtt import sglang_tpu_kernels

MIN_SCALE = 1e-12


def quantize_rows(x, dtype):
    return jax.ffi.ffi_call(
        "tt.quantize_rows",
        (
            jax.ShapeDtypeStruct(x.shape, dtype),
            jax.ShapeDtypeStruct((x.shape[0], 1), jnp.float32),
        ),
    )(x, min_scale=np.float32(MIN_SCALE), e5m2=np.bool_(dtype == jnp.float8_e5m2))


def quantize_ops(x, dtype):
    fp8_max = float(jnp.finfo(dtype).max)
    scale = jnp.maximum(
        jnp.max(jnp.abs(x), axis=-1, keepdims=True).astype(jnp.float32) / fp8_max, MIN_SCALE
    )
    return (x.astype(jnp.float32) / scale).astype(dtype), scale


def activations(rng, rows, cols):
    x = rng.normal(0, 1, (rows, cols)) * np.exp2(np.arange(rows) % 9 - 4)[:, None]
    outliers = rng.random((rows, cols)) < 0.002
    return np.where(outliers, x * 40, x).astype(np.float32)


def codes(a):
    return np.asarray(a).view(np.uint8)


def assert_near_ieee(x, q, scale, dtype):
    """q is the float8 of x / scale or of a float32 neighbor of that quotient."""
    fp8_max = float(ml_dtypes.finfo(dtype).max)
    want_scale = np.maximum(np.abs(x).max(axis=-1, keepdims=True) / np.float32(fp8_max), np.float32(MIN_SCALE))
    np.testing.assert_array_equal(np.asarray(scale), want_scale.astype(np.float32))
    quotient = (x / want_scale).astype(np.float32)
    got = codes(q)
    ok = got == codes(quotient.astype(dtype))
    for direction in (-np.inf, np.inf):
        ok |= got == codes(np.nextafter(quotient, np.float32(direction)).astype(dtype))
    assert ok.all(), f"{(~ok).sum()} codes off: {np.argwhere(~ok)[:5]}"


@pytest.mark.parametrize("trace", [False, True])
@pytest.mark.parametrize("opt", ["O0", "O1"])
@pytest.mark.parametrize(
    "rows,cols,in_dtype,dtype",
    [
        # SGLang decode: one and four tokens through the TP4 and TP1 linears.
        (1, 4096, jnp.bfloat16, jnp.float8_e4m3fn),
        (4, 1024, jnp.bfloat16, jnp.float8_e4m3fn),
        (4, 3072, jnp.bfloat16, jnp.float8_e4m3fn),
        (1, 12288, jnp.bfloat16, jnp.float8_e4m3fn),
        # Every row stride, partial dense tiles, FP32 input, e5m2.
        (7, 96, jnp.bfloat16, jnp.float8_e4m3fn),
        (16, 4096, jnp.bfloat16, jnp.float8_e4m3fn),
        (29, 2048, jnp.float32, jnp.float8_e4m3fn),
        (3, 512, jnp.bfloat16, jnp.float8_e5m2),
    ],
)
def test_quantize_rows(trace, opt, rows, cols, in_dtype, dtype):
    device = jax.devices("tt")[0]
    options = {"optimization_level": opt, "enable_trace": str(trace).lower()}
    fused = jax.jit(lambda x: quantize_rows(x, dtype), compiler_options=options)
    ops = jax.jit(lambda x: quantize_ops(x, dtype), compiler_options=options)
    rng = np.random.default_rng(rows * 131 + cols)
    for _ in range(2):
        x = jnp.asarray(activations(rng, rows, cols), in_dtype)
        q, scale = fused(jax.device_put(x, device))
        q_ops, scale_ops = ops(jax.device_put(x, device))
        assert q.dtype == dtype and q.shape == (rows, cols) and scale.shape == (rows, 1)
        np.testing.assert_array_equal(np.asarray(scale), np.asarray(scale_ops))
        np.testing.assert_array_equal(codes(q), codes(q_ops))
        assert_near_ieee(np.asarray(x, np.float32), q, scale, dtype)


@pytest.mark.parametrize("scale_shape", ["vector", "column"])
@pytest.mark.parametrize("tokens,k,n", [(1, 4096, 6144), (4, 1024, 4096)])
def test_fp8_linear_small_m(tmp_path, tokens, k, n, scale_shape):
    """The patched SGLang kernel against its math as jnp ops, with the
    dequantization and the narrowing to BF16 in the matmul's epilogue. A
    column of weight scales ([n, 1], as the official FP8 checkpoints store
    them) goes into the epilogue without the reshape to a row."""
    device = jax.devices("tt")[0]
    rng = np.random.default_rng(k + n)
    x = jnp.asarray(activations(rng, tokens, k), jnp.bfloat16)
    weight = jnp.asarray(rng.normal(0, 30, (k, n)), jnp.float8_e4m3fn)
    weight_scale = jnp.asarray(rng.uniform(1e-4, 1e-3, n), jnp.float32)
    if scale_shape == "column":
        weight_scale = weight_scale.reshape(n, 1)

    def reference(x, weight, weight_scale):
        x_q, x_scale = quantize_ops(x, weight.dtype)
        acc = jax.lax.dot_general(x_q, weight, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32)
        return (acc * x_scale * weight_scale.reshape(1, n)).astype(x.dtype)

    options = {"optimization_level": "O0"}
    args = [jax.device_put(a, device) for a in (x, weight, weight_scale)]
    kernel = jax.jit(sglang_tpu_kernels._fp8_linear_small_m, compiler_options={**options, "export_path": str(tmp_path)})
    got = kernel(*args)
    want = jax.jit(reference, compiler_options=options)(*args)
    np.testing.assert_array_equal(np.asarray(got, np.float32), np.asarray(want, np.float32))
    (ir,) = [path.read_text() for path in (tmp_path / "irs").glob("ttnn_[0-9]*.mlir")]
    assert re.search(r'"ttnn\.scaled_matmul".*-> tensor<\d+x\d+xbf16', ir), "no BF16 ttnn.scaled_matmul"
    assert '"ttnn.multiply"' not in ir, "a scale multiply was left unfused"
    assert not re.search(rf'"ttnn\.reshape".*tensor<{n}x1xf32', ir), "the weight scales were reshaped"


@pytest.mark.parametrize("trace", [False, True])
@pytest.mark.parametrize(
    "rows,width,dtype",
    [
        # Qwen3.8-27B's hidden size and its MLP down projection's input on
        # four chips; every row stride; e5m2.
        (1, 5120, jnp.float8_e4m3fn),
        (4, 4352, jnp.float8_e4m3fn),
        (13, 2048, jnp.float8_e4m3fn),
        (32, 1024, jnp.float8_e4m3fn),
        (1, 5120, jnp.float8_e5m2),
    ],
)
def test_quantize_rows_of_a_decode_norm(trace, rows, width, dtype, tmp_path):
    """The quantization of a decode norm's output runs in the norm's program
    (rms_norm_residual), which still returns the norm, and equals the
    separate programs bit for bit."""
    eps = 1e-6

    def norm(x, r, w):
        h = x + r
        hf = h.astype(jnp.float32)
        var = jnp.mean(hf * hf, axis=-1, keepdims=True)
        return (hf * (jax.lax.rsqrt(var + eps) * w.reshape(1, -1))).astype(jnp.bfloat16), h

    def layer(x, r, w):
        n, h = norm(x, r, w)
        return quantize_rows(n, dtype), n, h

    options = {"optimization_level": "O1", "enable_trace": str(trace).lower()}
    fused = jax.jit(layer, compiler_options={**options, "export_path": str(tmp_path)})
    separate_norm = jax.jit(norm, compiler_options=options)
    separate_quantize = jax.jit(lambda n: quantize_rows(n, dtype), compiler_options=options)
    device = jax.devices("tt")[0]
    rng = np.random.default_rng(rows * 7 + width)
    for _ in range(2):
        x, r = (jnp.asarray(activations(rng, rows, width), jnp.bfloat16) for _ in range(2))
        w = jnp.asarray(rng.uniform(0.5, 1.5, width), jnp.float32)
        args = [jax.device_put(a, device) for a in (x, r, w)]
        (q, scale), n, h = fused(*args)
        want_n, want_h = separate_norm(*args)
        want_q, want_scale = separate_quantize(want_n)
        np.testing.assert_array_equal(np.asarray(n, np.float32), np.asarray(want_n, np.float32))
        np.testing.assert_array_equal(np.asarray(h, np.float32), np.asarray(want_h, np.float32))
        np.testing.assert_array_equal(np.asarray(scale), np.asarray(want_scale))
        np.testing.assert_array_equal(codes(q), codes(want_q))
    irs = [path.read_text() for path in (tmp_path / "irs").glob("ttnn_*.mlir")]
    assert irs and any("ttnn.rms_norm_residual" in ir for ir in irs), "the add and the norm were not fused"
    assert not any("ttnn.quantize_rows" in ir for ir in irs), "the quantization ran as its own program"
