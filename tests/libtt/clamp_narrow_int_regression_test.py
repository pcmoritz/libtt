"""ttnn clamp supports only int32 and uint32 integers; narrower ones are
clamped as int32."""

import jax
import numpy as np
import pytest


def device_put(x):
    return jax.device_put(x, jax.devices("tt")[0])


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_clamp_narrow_unsigned(dtype):
    x = np.array([0, 1, 7, 200, 255], dtype)
    got = jax.jit(lambda x: jax.lax.clamp(dtype(2), x, dtype(100)))(device_put(x))
    np.testing.assert_array_equal(np.asarray(got), np.clip(x, 2, 100))


def test_vmap_dynamic_update_slice_uint8_indices():
    # vmap lowers this to a scatter whose uint8 start indices JAX clamps.
    operand = np.arange(6, dtype=np.float32).reshape(2, 3)
    update = np.full((2, 1), -1, np.float32)
    indices = np.array([[1], [200]], np.uint8)
    run = jax.jit(jax.vmap(jax.lax.dynamic_update_slice))
    got = run(device_put(operand), device_put(update), device_put(indices))
    expected = operand.copy()
    expected[0, 1] = -1
    expected[1, 2] = -1  # out-of-range starts clamp to the last valid one
    np.testing.assert_array_equal(np.asarray(got), expected)
