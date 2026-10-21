"""Activation fusion must preserve small, nonzero clamp bounds."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("level", ["O0", "O1"])
@pytest.mark.parametrize("bound", [-1e-7, -1e-12, 1e-12, 1e-7, 0.0])
def test_maximum_bound(level, bound, tmp_path):
    bound = np.float32(bound)
    x = np.array([-1, -1e-7, 0, 1e-7, 1], np.float32)
    run = jax.jit(lambda x: jnp.maximum(x, bound),
                  compiler_options={"optimization_level": level, "export_path": str(tmp_path)})
    got = run(jax.device_put(x, jax.devices("tt")[0]))
    np.testing.assert_array_equal(np.asarray(got), np.maximum(x, bound))
    if bound == 0:
        ir = "\n".join(p.read_text() for p in (tmp_path / "irs").glob("ttnn_[0-9]*.mlir"))
        assert "ttnn.relu" in ir or "op_type = relu>" in ir


@pytest.mark.parametrize("level", ["O0", "O1"])
def test_relu6_nearby_bound(level):
    bound = np.nextafter(np.float32(6), np.float32(7))
    x = np.array([-1, 0, 6, 7], np.float32)
    run = jax.jit(lambda x: jnp.minimum(jnp.maximum(x, 0), bound),
                  compiler_options={"optimization_level": level})
    got = run(jax.device_put(x, jax.devices("tt")[0]))
    np.testing.assert_array_equal(np.asarray(got), np.clip(x, 0, bound))


@pytest.mark.parametrize("level", ["O0", "O1"])
@pytest.mark.parametrize("dtype", [np.float32, jnp.bfloat16])
def test_bce_backward_boundaries(level, dtype):
    x = np.array([0, 1, 0, 1], dtype=dtype)
    def backward(x):
        value = x.astype(jnp.float32)
        target = jnp.array([1, 0, 0, 1], jnp.float32)
        denominator = jnp.maximum((1 - value) * value, np.float32(1e-12))
        return ((value - target) / denominator).astype(x.dtype)
    run = jax.jit(backward, compiler_options={"optimization_level": level})
    got = run(jax.device_put(x, jax.devices("tt")[0]))
    expected = np.array([-1e12, 1e12, 0, 0], dtype=dtype).astype(np.float32)
    np.testing.assert_allclose(np.asarray(got, np.float32), expected, rtol=4e-5, atol=0)
