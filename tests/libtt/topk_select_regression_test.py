"""lax.top_k on TTNN's selection path: up to two tile rows and wide rows.

The data-movement cores scan each row once, keep the best k with the lower
index first among equal values, and order floats like lax.top_k on CPU (IEEE
total order: NaN above inf, -0.0 below +0.0). Wide rows split into column
segments that the tile row's first core merges.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax


def reference(x, k):
    # lax.top_k on CPU is the contract, including NaN and signed zeros.
    with jax.default_device(jax.devices("cpu")[0]):
        values, indices = lax.top_k(jnp.asarray(x), k)
    return np.asarray(values), np.asarray(indices)


def run(x, k):
    return jax.jit(lambda a: lax.top_k(a, k))(jax.device_put(x, jax.devices("tt")[0]))


def check(x, k):
    values, indices = run(x, k)
    want_values, want_indices = reference(x, k)
    assert values.shape == indices.shape == x.shape[:-1] + (k,)
    np.testing.assert_array_equal(np.asarray(indices), want_indices)
    # Compare bits: NaN payloads and signed zeros included.
    got = np.asarray(values).astype(np.float32).view(np.uint32)
    np.testing.assert_array_equal(got, want_values.astype(np.float32).view(np.uint32))


@pytest.mark.parametrize("dtype", [np.float32, "bfloat16"])
@pytest.mark.parametrize(
    "shape,k",
    [
        # Routers: one token and a few, 32 to 512 experts.
        ((1, 32), 4),
        ((1, 60), 4),
        ((1, 128), 8),
        ((1, 128), 1),
        ((3, 128), 5),
        ((32, 128), 8),
        ((33, 256), 64),
        ((64, 512), 10),
        # Widths off the tile grid and rows past one tile.
        ((1, 1000), 33),
        ((40, 2048), 16),
        # Wide rows split into segments: logits over a vocabulary.
        ((1, 32768), 20),
        ((1, 151936), 20),
        ((8, 50000), 5),
        ((3, 4097), 64),
        # Several [H, W] slabs.
        ((2, 3, 4096), 7),
    ],
)
def test_top_k_selection(dtype, shape, k):
    rng = np.random.default_rng(sum(shape) + k)
    # Few distinct values give long runs of ties.
    x = rng.integers(-50, 50, shape).astype(np.float32).astype(dtype)
    check(x, k)


@pytest.mark.parametrize("dtype", [np.float32, "bfloat16"])
@pytest.mark.parametrize("width", [128, 5000])
def test_top_k_selection_orders_nan_inf_and_signed_zeros(dtype, width):
    rng = np.random.default_rng(width)
    x = rng.standard_normal((2, width)).astype(np.float32)
    specials = [np.nan, -np.nan, np.inf, -np.inf, 0.0, -0.0, 0.0, -0.0, np.nan]
    for row in range(2):
        cols = rng.choice(width, len(specials), replace=False)
        x[row, cols] = specials
    check(x.astype(dtype), 12)


@pytest.mark.parametrize("dtype", [np.float32, "bfloat16"])
def test_top_k_selection_all_equal(dtype):
    """Every value tied: the first k indices, in order."""
    check(np.full((2, 3000), 1.5, np.float32).astype(dtype), 9)
