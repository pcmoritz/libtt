"""Integer sums must stay exact beyond float32 precision."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("axis", [0, 1, None])
def test_int32_sum_precision_and_overflow(axis):
    values = np.array(
        [[2**24, 1], [-2**24, -1], [2**31 - 1, 1], [-2**31, -1]],
        dtype=np.int32,
    )
    device_values = jax.device_put(values, jax.devices("tt")[0])
    actual = jax.jit(lambda x: jnp.sum(x, axis=axis))(device_values)
    expected = np.sum(values, axis=axis, dtype=np.int32)
    np.testing.assert_array_equal(np.asarray(actual), expected)
