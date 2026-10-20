"""ttnn pooling runs in bfloat16, so tt-mlir adds the window slices of float32
and float16 sum pooling (and average pooling, which divides the sum) in their
own type (torch_tpu's avg_pool1d/2d/3d). A pad folds into a pooling op only if
it pads with the reduction's init value."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

CASES = [
    # shape, window, strides, padding, dilation
    ((2, 3, 16), (1, 1, 3), (1, 1, 2), ((0, 0), (0, 0), (1, 1)), (1, 1, 1)),
    ((2, 3, 8, 8), (1, 1, 2, 2), (1, 1, 2, 2), ((0, 0),) * 4, (1, 1, 1, 1)),
    ((1, 2, 9, 9), (1, 1, 3, 3), (1, 1, 1, 1), ((0, 0), (0, 0), (1, 1), (1, 2)), (1, 1, 1, 1)),
    ((3, 20), (1, 2), (1, 1), ((0, 0), (0, 1)), (1, 2)),
    ((1, 2, 4, 6, 6), (1, 1, 2, 2, 2), (1, 1, 2, 2, 2), ((0, 0),) * 5, (1, 1, 1, 1, 1)),
    # A 6x3x4 window (72 elements) sums 13 slices, one dimension at a time.
    ((1, 2, 7, 9, 8), (1, 1, 6, 3, 4), (1, 1, 2, 3, 2), ((0, 0), (0, 0), (3, 3), (1, 1), (0, 0)), (1,) * 5),
]
# (rtol, atol); bfloat16 pooling is off by about 1e-2.
TOLERANCES = {np.float32: (1e-5, 1e-5), np.float16: (4e-3, 1e-2)}


def reference(x, window, strides, padding, dilation):
    x = np.pad(x.astype(np.float64), padding)
    extent = [(w - 1) * d + 1 for w, d in zip(window, dilation)]
    shape = [(n - e) // s + 1 for n, e, s in zip(x.shape, extent, strides)]
    out = np.zeros(shape)
    for offset in np.ndindex(*window):
        out += x[
            tuple(
                slice(o * d, o * d + (n - 1) * s + 1, s)
                for o, d, n, s in zip(offset, dilation, shape, strides)
            )
        ]
    return out


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize("case", CASES, ids=lambda case: "x".join(map(str, case[0])))
def test_sum_and_average_pool(dtype, case):
    shape, window, strides, padding, dilation = case
    x = np.random.default_rng(0).standard_normal(shape).astype(dtype)

    def pools(x):
        total = lax.reduce_window(
            x, np.array(0, dtype), lax.add, window, strides, padding,
            window_dilation=dilation,
        )
        return total, total / np.array(np.prod(window), dtype)

    total, average = jax.jit(pools)(jax.device_put(x, jax.devices("tt")[0]))
    expected = reference(x, window, strides, padding, dilation)
    rtol, atol = TOLERANCES[dtype]
    np.testing.assert_allclose(np.asarray(total, np.float64), expected, rtol=rtol, atol=atol)
    np.testing.assert_allclose(
        np.asarray(average, np.float64), expected / np.prod(window), rtol=rtol, atol=atol
    )


@pytest.mark.parametrize("dtype", [jnp.bfloat16, np.float32])
def test_window_counts_of_padded_ones(dtype):
    # torch_tpu's avg_pool with ceil_mode counts its divisor this way; folding
    # the ones padding into the pool as zeros gave 2 instead of 3 at the edges.
    def counts(x):
        ones = jnp.pad(jnp.ones_like(x), ((0, 0), (0, 0), (1, 1)), constant_values=dtype(1))
        return lax.reduce_window(
            ones, dtype(0), lax.add, (1, 1, 3), (1, 1, 2), ((0, 0), (0, 0), (0, 1))
        )

    x = np.zeros((2, 3, 16), dtype)
    got = np.asarray(jax.jit(counts)(jax.device_put(x, jax.devices("tt")[0])), np.float32)
    expected = np.full((2, 3, 9), 3, np.float32)
    expected[..., -1] = 2
    # bfloat16 still takes ttnn's pooling, which rounds 3 to 2.984375.
    np.testing.assert_allclose(got, expected, atol=0.05 if dtype == jnp.bfloat16 else 0)
