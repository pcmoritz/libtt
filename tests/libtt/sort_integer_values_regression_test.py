"""Integer tensors sorted along with a key must keep their exact values."""

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("dtype", [jnp.int16, jnp.uint16, jnp.int32, jnp.uint32])
def test_sort_carries_integer_values_exactly(dtype):
    rng = np.random.default_rng(0)
    key = rng.standard_normal((3, 64)).astype(np.float32)
    info = np.iinfo(dtype)
    values = rng.integers(info.min, info.max, size=key.shape, dtype=dtype)

    sorted_key, sorted_values = jax.jit(
        lambda k, v: lax.sort((k, v), dimension=1, num_keys=1, is_stable=True)
    )(key, values)

    order = np.argsort(key, axis=1, kind="stable")
    np.testing.assert_array_equal(sorted_key, np.take_along_axis(key, order, 1))
    np.testing.assert_array_equal(sorted_values, np.take_along_axis(values, order, 1))
