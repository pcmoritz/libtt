"""Integer tensors sorted along with a key must keep their exact values."""

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("dtype", [jnp.int16, jnp.uint16, jnp.int32, jnp.uint32])
@pytest.mark.parametrize(
    "shape,dimension",
    [
        ((3, 64), 1),
        # Indices of the lower faces and of more than one tile row.
        ((33, 128), 1),
        # Above rank 4, ttnn.gather restores a non-last dimension incorrectly.
        ((2, 3, 2, 32, 32), 0),
        ((2, 3, 2, 32, 32), 2),
    ],
)
def test_sort_carries_integer_values_exactly(dtype, shape, dimension):
    rng = np.random.default_rng(0)
    key = rng.standard_normal(shape).astype(np.float32)
    info = np.iinfo(dtype)
    values = rng.integers(info.min, info.max, size=key.shape, dtype=dtype)

    sorted_key, sorted_values = jax.jit(
        lambda k, v: lax.sort((k, v), dimension=dimension, num_keys=1, is_stable=True)
    )(key, values)

    order = np.argsort(key, axis=dimension, kind="stable")
    np.testing.assert_array_equal(sorted_key, np.take_along_axis(key, order, dimension))
    np.testing.assert_array_equal(
        sorted_values, np.take_along_axis(values, order, dimension)
    )
