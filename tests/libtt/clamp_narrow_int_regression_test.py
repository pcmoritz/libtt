"""ttnn clamp supports only int32 and uint32 integers; narrower ones are
clamped as int32."""

import jax
import numpy as np
import pytest

OPTIMIZATION_LEVELS = pytest.mark.parametrize("level", ["O0", "O1"])


def device_put(x):
    return jax.device_put(x, jax.devices("tt")[0])


@OPTIMIZATION_LEVELS
@pytest.mark.parametrize(
    "dtype,values,low,high",
    [
        (np.uint8, [0, 1, 7, 200, 255], 2, 100),
        (np.uint16, [0, 255, 256, 32767, 32768, 60001, 65535], 2, 60000),
    ],
)
def test_clamp_narrow_unsigned(level, dtype, values, low, high):
    x = np.array(values, dtype)
    run = jax.jit(
        lambda x: jax.lax.clamp(dtype(low), x, dtype(high)),
        compiler_options={"optimization_level": level},
    )
    got = np.asarray(run(device_put(x)))
    assert got.dtype == x.dtype
    np.testing.assert_array_equal(got, np.clip(x, low, high))


# O1 fails separately: the optimizer rejects the scatter's uint8 ttnn.arange.
@pytest.mark.parametrize("level", ["O0"])
def test_vmap_dynamic_update_slice_uint8_indices(level):
    # vmap lowers this to a scatter whose uint8 start indices JAX clamps.
    operand = np.arange(6, dtype=np.float32).reshape(2, 3)
    update = np.full((2, 1), -1, np.float32)
    indices = np.array([[1], [200]], np.uint8)
    run = jax.jit(
        jax.vmap(jax.lax.dynamic_update_slice),
        compiler_options={"optimization_level": level},
    )
    got = run(device_put(operand), device_put(update), device_put(indices))
    expected = operand.copy()
    expected[0, 1] = -1
    expected[1, 2] = -1  # out-of-range starts clamp to the last valid one
    np.testing.assert_array_equal(np.asarray(got), expected)
