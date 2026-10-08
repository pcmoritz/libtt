"""Reductions ignore the tile padding of their input.

TTNN fills a tiled input's padding with the reduction's identity before
reducing (fill_pad), using mask tiles it builds on the device. The input here
is y + 3 computed on the device, so its padding holds 3 rather than zeros and a
wrong mask changes the result: y is -2 or -1, so every valid value is 1 or 2
and padding left in a max wins, and padding left in a sum adds 3 per element.
The values and their sums are small integers, exact in BF16 too, so most
results must match exactly. The shapes cover right-edge padding, bottom-edge
padding and both, over each reduced axis, with reduced dimensions on either
side of a face boundary (15, 16, 17).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

SHIFT = 3.0


@pytest.mark.parametrize("dtype", [np.float32, "bfloat16"])
@pytest.mark.parametrize(
    "shape,op,axis",
    [
        ((1, 8), "sum", -1),  # a router's top-8 weights, renormalized
        ((1, 8), "max", -1),
        ((32, 8), "sum", -1),
        ((3, 31), "sum", -1),
        ((5, 40), "max", -1),
        ((4, 15), "sum", -1),
        ((4, 16), "max", -1),
        ((4, 17), "sum", -1),
        ((4, 17), "max", -1),
        ((15, 32), "sum", 0),
        ((15, 32), "max", 0),
        ((16, 32), "sum", 0),
        ((16, 32), "max", 0),
        ((17, 32), "sum", 0),
        ((17, 32), "max", 0),
        ((33, 32), "sum", 0),
        ((33, 32), "max", 0),
        ((70, 17), "sum", 0),
        ((33, 100), "sum", None),
        ((2, 3, 47), "sum", -1),
        ((2, 49, 32), "max", 1),
    ],
)
def test_reduction_ignores_padding(dtype, shape, op, axis):
    rng = np.random.default_rng(sum(shape))
    y = rng.integers(-2, 0, shape).astype(np.float32).astype(dtype)
    run = jax.jit(lambda a: getattr(jnp, op)(a + jnp.asarray(SHIFT, a.dtype), axis=axis))
    got = np.asarray(run(jax.device_put(y, jax.devices("tt")[0]))).astype(np.float64)
    want = getattr(np, op)(np.asarray(y).astype(np.float64) + SHIFT, axis=axis)
    if dtype == "bfloat16" and np.abs(want).max() > 256:
        # Beyond 256, BF16 no longer holds every integer: the sum rounds.
        np.testing.assert_allclose(got, want, rtol=2**-7)
    else:
        np.testing.assert_array_equal(got, want)
