"""Max pooling with torch_tpu's lowest-finite init failed to legalize, and ttnn
pooling rounds float32, float16 and wide integers through bfloat16. tt-mlir
takes the maximum of window slices in the input type instead, and applies the
body of a variadic window (max pooling with indices) to the slices (torch_tpu's
max_pool1d/2d/3d)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

CASES = [
    # shape, window, strides, padding, dilation
    ((2, 3, 16), (1, 1, 3), (1, 1, 2), ((0, 0), (0, 0), (1, 1)), (1, 1, 1)),
    ((2, 3, 9, 8), (1, 1, 3, 2), (1, 1, 2, 2), ((0, 0), (0, 0), (1, 1), (0, 1)), (1, 1, 1, 1)),
    ((1, 2, 7, 9, 8), (1, 1, 3, 2, 3), (1, 1, 2, 1, 2), ((0, 0),) * 5, (1, 1, 1, 2, 1)),
    # ttnn pads rank-5 tensors as rank 4, which dropped the pad value.
    ((1, 2, 4, 5, 6), (1, 1, 3, 3, 3), (1, 1, 2, 2, 2), ((0, 0), (0, 0)) + ((1, 1),) * 3, (1,) * 5),
]
DTYPES = [np.float32, np.float16, jnp.bfloat16, np.int32, np.int16, np.uint8]


def lowest(dtype):
    return jnp.finfo(dtype).min if jnp.issubdtype(dtype, jnp.floating) else jnp.iinfo(dtype).min


def sample(shape, dtype):
    rng = np.random.default_rng(0)
    if jnp.issubdtype(dtype, jnp.floating):
        return (rng.standard_normal(shape) * 1000).astype(dtype)
    info = np.iinfo(dtype)
    return rng.integers(info.min, info.max, shape, dtype=dtype, endpoint=True)


@pytest.mark.parametrize("init", ["lowest", "-inf"])
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: np.dtype(d).name)
@pytest.mark.parametrize("case", CASES, ids=lambda case: "x".join(map(str, case[0])))
def test_max_pool(case, dtype, init):
    if init == "-inf" and not jnp.issubdtype(dtype, jnp.floating):
        pytest.skip("integers have no -inf")
    shape, window, strides, padding, dilation = case
    x = sample(shape, dtype)
    start = np.array(lowest(dtype) if init == "lowest" else -np.inf, dtype)

    def pool(x):
        return lax.reduce_window(x, start, lax.max, window, strides, padding, window_dilation=dilation)

    got = jax.jit(pool)(jax.device_put(x, jax.devices("tt")[0]))
    expected = jax.jit(pool, backend="cpu")(x)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))


def max_with_index(a, b):
    # torch_tpu's max_pool2d_with_indices body: the larger value, the smaller
    # index between equal values, and NaN wins.
    (a_value, a_index), (b_value, b_index) = a, b
    a_wins = (a_value > b_value) | ((a_value == b_value) & (a_index < b_index))
    a_wins |= jnp.isnan(a_value) & (~jnp.isnan(b_value) | (a_index < b_index))
    return jnp.maximum(a_value, b_value), jnp.where(a_wins, a_index, b_index)


@pytest.mark.parametrize("dtype", [np.float32, np.float16, jnp.bfloat16])
@pytest.mark.parametrize("case", CASES, ids=lambda case: "x".join(map(str, case[0])))
def test_max_pool_with_indices(case, dtype):
    shape, window, strides, padding, dilation = case
    x = sample(shape, dtype)
    x.flat[::7] = x.flat[3]  # ties
    indices = np.arange(x.size, dtype=np.int32).reshape(shape)
    starts = (np.array(lowest(dtype), dtype), np.array(np.iinfo(np.int32).max, np.int32))

    def pool(x, indices):
        return lax.reduce_window(
            (x, indices), starts, max_with_index, window, strides, padding,
            window_dilation=dilation,
        )

    device = jax.devices("tt")[0]
    got = jax.jit(pool)(jax.device_put(x, device), jax.device_put(indices, device))
    expected = jax.jit(pool, backend="cpu")(x, indices)
    for g, e in zip(got, expected):
        np.testing.assert_array_equal(np.asarray(g), np.asarray(e))


def test_max_pool_with_indices_starts_from_init():
    # A window that starts from 0.5 keeps 0.5 and the init index where all its
    # values are smaller.
    shape, window, strides, padding, dilation = CASES[1]
    x = sample(shape, np.float32) / 1000
    indices = np.arange(x.size, dtype=np.int32).reshape(shape)
    starts = (np.array(0.5, np.float32), np.array(-1, np.int32))

    def pool(x, indices):
        return lax.reduce_window(
            (x, indices), starts, max_with_index, window, strides, padding,
            window_dilation=dilation,
        )

    device = jax.devices("tt")[0]
    got = jax.jit(pool)(jax.device_put(x, device), jax.device_put(indices, device))
    expected = jax.jit(pool, backend="cpu")(x, indices)
    for g, e in zip(got, expected):
        np.testing.assert_array_equal(np.asarray(g), np.asarray(e))


@pytest.mark.parametrize("dtype", [np.float32, jnp.bfloat16])
def test_reduce_max_with_index(dtype):
    # torch_tpu's adaptive_max_pool2d reduces gathered windows this way.
    x = sample((2, 3, 4, 5), dtype)
    x.flat[::5] = x.flat[2]  # ties
    indices = np.arange(x.size, dtype=np.int32).reshape(x.shape)
    starts = (np.array(lowest(dtype), dtype), np.array(np.iinfo(np.int32).max, np.int32))

    def reduce(x, indices):
        return lax.reduce((x, indices), starts, max_with_index, (3,))

    device = jax.devices("tt")[0]
    got = jax.jit(reduce)(jax.device_put(x, device), jax.device_put(indices, device))
    expected = jax.jit(reduce, backend="cpu")(x, indices)
    for g, e in zip(got, expected):
        np.testing.assert_array_equal(np.asarray(g), np.asarray(e))
