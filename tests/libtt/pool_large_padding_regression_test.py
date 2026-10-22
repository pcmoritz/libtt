"""Pooling padding beyond half the window, as torch_tpu writes ceil_mode
pooling, is applied by ttnn pooling to its input. Max pooling padded it with
zeros, and select_and_scatter (max pooling gradients) scattered indices into
the padded input to the unpadded tensor. Max pooling now pads with -inf and
select_and_scatter pads explicitly."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

CASES = [
    # shape, window, strides, padding
    ((1, 2, 8), (1, 1, 3), (1, 1, 2), ((0, 0), (0, 0), (1, 2))),
    ((1, 2, 8), (1, 1, 3), (1, 1, 2), ((0, 0), (0, 0), (0, 2))),
    ((1, 2, 5, 6), (1, 1, 3, 3), (1, 1, 2, 2), ((0, 0), (0, 0), (1, 2), (2, 1))),
]
IDS = ["1d-pad1-2", "1d-pad0-2", "2d"]


def max_pool(x, window, strides, padding):
    return lax.reduce_window(x, -jnp.inf, lax.max, window, strides, padding)


def compare(f, x):
    got = jax.jit(f)(jax.device_put(x, jax.devices("tt")[0]))
    expected = jax.jit(f, backend="cpu")(x)
    np.testing.assert_array_equal(np.asarray(got, np.float32), np.asarray(expected, np.float32))


@pytest.mark.parametrize("dtype", [np.float32, jnp.bfloat16])
@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_max_pool_large_padding(case, dtype):
    shape, window, strides, padding = case
    # Distinct, exactly representable values isolate padding from rounding.
    # All negative, so that zero padding would be the maximum.
    x = (-1 - np.random.default_rng(0).permutation(np.prod(shape))).reshape(shape).astype(dtype)
    compare(lambda x: max_pool(x, window, strides, padding), x)


@pytest.mark.parametrize("dtype", [np.float32, jnp.bfloat16])
@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_max_pool_gradient_large_padding(case, dtype):
    shape, window, strides, padding = case
    x = np.random.default_rng(0).permutation(np.prod(shape)).reshape(shape).astype(dtype)
    compare(jax.grad(lambda x: max_pool(x, window, strides, padding).sum()), x)
