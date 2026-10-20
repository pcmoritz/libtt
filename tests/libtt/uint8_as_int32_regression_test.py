"""tt-mlir computes uint8 argmax and tensor clamp as int32: ttnn argmax and the
tensor clamp don't take uint8. uint8 transposes (at O1 row-major, e.g. before
an argmax over axis 0) and pads run natively; see
uint8_permute_regression_test.py and uint8_pad_regression_test.py. torch_tpu's
argmax, argmin, max/min with dim, clamp, hardtanh and padded uint8 pooling take
these paths."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

OPTIMIZATION_LEVELS = pytest.mark.parametrize("level", ["O0", "O1"])

X = np.array([[4, 5, 6, 4, 7], [9, 9, 5, 6, 0], [1, 200, 3, 255, 2]], np.uint8)


def run(f, level, *args):
    device = jax.devices("tt")[0]
    run = jax.jit(f, compiler_options={"optimization_level": level})
    return np.asarray(run(*(jax.device_put(a, device) for a in args)))


@OPTIMIZATION_LEVELS
@pytest.mark.parametrize("axis", [None, 0, 1])
def test_argmax_argmin_uint8(level, axis):
    got = run(lambda x: jnp.argmax(x, axis=axis), level, X)
    np.testing.assert_array_equal(got, np.argmax(X, axis=axis))
    got = run(lambda x: jnp.argmin(x, axis=axis), level, X)
    np.testing.assert_array_equal(got, np.argmin(X, axis=axis))


@OPTIMIZATION_LEVELS
@pytest.mark.parametrize("shape", [(3, 5), (64, 128), (2, 7, 300)])
def test_transpose_uint8(level, shape):
    x = np.random.default_rng(0).integers(0, 256, shape, dtype=np.uint8)
    got = run(lambda x: jnp.swapaxes(x, -1, -2), level, x)
    np.testing.assert_array_equal(got, np.swapaxes(x, -1, -2))


@OPTIMIZATION_LEVELS
def test_clamp_tensor_bounds_uint8(level):
    low = np.full_like(X, 2)
    low[0] = 5
    high = np.full_like(X, 250)
    high[1] = 6
    got = run(lambda low, x, high: jax.lax.clamp(low, x, high), level, low, X, high)
    assert got.dtype == np.uint8
    np.testing.assert_array_equal(got, np.minimum(np.maximum(X, low), high))


@OPTIMIZATION_LEVELS
@pytest.mark.parametrize(
    "padding", [((1, 0), (2, 3)), ((0, 2), (0, 1))], ids=["front", "back_only"]
)
def test_pad_uint8(level, padding):
    got = run(lambda x: jnp.pad(x, padding, constant_values=np.uint8(7)), level, X)
    assert got.dtype == np.uint8
    np.testing.assert_array_equal(got, np.pad(X, padding, constant_values=7))
