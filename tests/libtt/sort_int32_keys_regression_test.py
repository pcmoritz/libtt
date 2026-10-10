"""Integer keys must sort exactly: 32-bit ones, including values float32 cannot
hold, and signed 8- and 16-bit ones, which are int32 tensors on device."""

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


@pytest.mark.parametrize("dtype", [jnp.int32, jnp.uint32, jnp.int8, jnp.int16])
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


@pytest.mark.parametrize("dtype", [jnp.int32, jnp.uint32, jnp.int8, jnp.int16])
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


_BOUNDARY_KEYS = {
    jnp.int32: [-(2**31), -65537, -65536, -65535, -1, 0, 1, 65535, 65536, 65537, 2**31 - 1],
    jnp.uint32: [0, 1, 65535, 65536, 65537, 2**31 - 1, 2**31, 2**32 - 65536, 2**32 - 1],
    # Narrow keys sort as float32: their range extremes and neighbours.
    jnp.int8: [-128, -127, -2, -1, 0, 1, 126, 127],
    jnp.int16: [-32768, -32767, -257, -256, -1, 0, 1, 255, 256, 32766, 32767],
}


def _boundary_keys(dtype, shape, dimension):
    """Boundary keys (see _BOUNDARY_KEYS), each value repeated along `dimension`."""
    rng = np.random.default_rng(1)
    pool = np.array(_BOUNDARY_KEYS[dtype], dtype=np.int64)
    keys = rng.choice(pool, size=shape)
    # Make sure every boundary value occurs at least twice in each slice.
    reps = np.resize(np.repeat(pool, 2), shape[dimension])
    keys[(slice(None),) * dimension + (slice(0, reps.size),)] = np.expand_dims(
        reps, tuple(i for i in range(len(shape)) if i != dimension)
    )
    keys = np.take_along_axis(keys, rng.permuted(np.indices(shape)[dimension], axis=dimension), dimension)
    return keys.astype(dtype)


@pytest.mark.parametrize("dtype", [jnp.int32, jnp.uint32, jnp.int8, jnp.int16])
@pytest.mark.parametrize("shape,dimension", [((300, 3), 0), ((2, 3, 40, 2, 2), 2)])
def test_sort_int32_boundary_keys_with_integer_values(dtype, shape, dimension):
    keys = _boundary_keys(dtype, shape, dimension)
    payload = np.arange(keys.size, dtype=np.int32).reshape(shape) * 4099 - 2**30
    sorted_keys, sorted_payload = jax.jit(
        lambda k, p: lax.sort((k, p), dimension=dimension, num_keys=1, is_stable=True)
    )(keys, payload)
    order = np.argsort(keys, axis=dimension, kind="stable")
    np.testing.assert_array_equal(sorted_keys, np.take_along_axis(keys, order, dimension))
    np.testing.assert_array_equal(
        sorted_payload, np.take_along_axis(payload, order, dimension)
    )


@pytest.mark.parametrize("dtype", [jnp.int32, jnp.uint32, jnp.int8, jnp.int16])
@pytest.mark.parametrize("descending", [False, True])
def test_argsort_int32_boundary_keys(dtype, descending):
    keys = _boundary_keys(dtype, (300, 3), 0)
    indices = jax.jit(
        lambda k: jnp.argsort(k, axis=0, stable=True, descending=descending)
    )(keys)
    signed = keys.astype(np.int64)
    expected = np.argsort(-signed if descending else signed, axis=0, kind="stable")
    np.testing.assert_array_equal(indices, expected)
