"""32-bit integer keys must sort exactly, including values float32 cannot hold."""

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import pytest


def _keys(dtype, shape):
    rng = np.random.default_rng(0)
    info = np.iinfo(dtype)
    keys = rng.integers(info.min, info.max, size=shape, dtype=dtype, endpoint=True)
    # Ties, and neighbours that differ only in the lowest bit.
    keys[..., : shape[-1] // 4] = keys[..., shape[-1] // 4 : shape[-1] // 2]
    keys[..., -2] = keys[..., -1] ^ 1
    return keys


@pytest.mark.parametrize("dtype", [jnp.int32, jnp.uint32])
@pytest.mark.parametrize(
    "shape,dimension",
    [
        ((3, 200), 1),
        # Above rank 4, ttnn.gather restores a non-last dimension incorrectly.
        ((2, 3, 2, 32, 32), 0),
        ((2, 3, 2, 32, 32), 2),
    ],
)
def test_sort_int32_keys_with_values(dtype, shape, dimension):
    keys = np.moveaxis(_keys(dtype, np.moveaxis(np.empty(shape), dimension, -1).shape), -1, dimension)
    payload = np.arange(keys.size, dtype=np.float32).reshape(keys.shape)
    sorted_keys, sorted_payload = jax.jit(
        lambda k, p: lax.sort((k, p), dimension=dimension, num_keys=1, is_stable=True)
    )(keys, payload)
    order = np.argsort(keys, axis=dimension, kind="stable")
    np.testing.assert_array_equal(sorted_keys, np.take_along_axis(keys, order, dimension))
    np.testing.assert_array_equal(
        sorted_payload, np.take_along_axis(payload, order, dimension)
    )


@pytest.mark.parametrize("dtype", [jnp.int32, jnp.uint32])
@pytest.mark.parametrize("descending", [False, True])
def test_sort_and_argsort_int32_keys(dtype, descending):
    keys = _keys(dtype, (2, 300))
    values = jax.jit(lambda k: jnp.sort(k, axis=-1, descending=descending))(keys)
    indices = jax.jit(
        lambda k: jnp.argsort(k, axis=-1, stable=True, descending=descending)
    )(keys)
    expected = np.sort(keys, axis=-1)
    if descending:
        expected = expected[..., ::-1]
        # A stable descending argsort keeps equal keys in their original order.
        expected_indices = np.argsort(-keys.astype(np.int64), axis=-1, kind="stable")
    else:
        expected_indices = np.argsort(keys, axis=-1, kind="stable")
    np.testing.assert_array_equal(values, expected)
    np.testing.assert_array_equal(indices, expected_indices)
