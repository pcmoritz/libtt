"""An FP32 RMS norm of a BF16 tensor cast to FP32 normalizes the BF16 tensor
directly (FP32 accumulation unpacks BF16 exactly), producing FP32 without the
typecast program and with the same result."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def compile_and_run(fn, tmp_path, opt_level, *args):
    run = jax.jit(fn, compiler_options={"optimization_level": opt_level, "export_path": str(tmp_path)})
    device = jax.devices("tt")[0]
    out = run(*(jax.device_put(a, device) for a in args))
    (ir,) = [path.read_text() for path in (tmp_path / "irs").glob("ttnn_[0-9]*.mlir")]
    return np.asarray(out), ir


def grouped_norm(x32, w):
    # The hyper-connection norm as sglang-jax writes it: the flat [T, 4 * 2560]
    # streams normalized per stream, scaled per element.
    grouped = x32.reshape(1, 4, 2560)
    variance = jnp.mean(jnp.square(grouped), axis=-1, keepdims=True)
    normed = (grouped * jax.lax.rsqrt(variance + 1e-6)).reshape(x32.shape)
    return normed * (1.0 + w)


@pytest.mark.parametrize("opt_level", ["O0", "O1"])
def test_norm_of_upcast_matches_fp32_input(tmp_path, opt_level):
    rng = np.random.default_rng(0)
    x = (rng.normal(size=(1, 10240)) * 3).astype(jnp.bfloat16)
    w = (0.1 * rng.normal(size=(10240,))).astype(np.float32)
    folded, ir = compile_and_run(lambda x, w: grouped_norm(x.astype(jnp.float32), w), tmp_path / "bf16", opt_level, x, w)
    assert "ttnn.rms_norm" in ir
    assert "ttnn.typecast" not in ir
    reference, _ = compile_and_run(grouped_norm, tmp_path / "f32", opt_level, x.astype(np.float32), w)
    np.testing.assert_array_equal(folded, reference)
