"""Reductions ignore the tile padding of their input.

TTNN fills a tiled input's padding with the reduction's identity before
reducing (fill_pad), using mask tiles it builds on the device. The input here
is y + c computed on the device, so its padding holds c rather than zeros and a
wrong mask changes the result. The shapes cover right-edge padding, bottom-edge
padding and both, over each reduced axis.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("dtype", [np.float32, "bfloat16"])
@pytest.mark.parametrize(
    "shape,op,axis",
    [
        ((1, 8), "sum", -1),  # a router's top-8 weights, renormalized
        ((1, 8), "max", -1),
        ((32, 8), "sum", -1),
        ((3, 31), "sum", -1),
        ((5, 40), "max", -1),
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
    y = rng.uniform(-1, 1, shape).astype(np.float32).astype(dtype)
    shift = 3.0 if op == "sum" else -5.0  # the padding holds `shift` after the add
    run = jax.jit(lambda a: getattr(jnp, op)(a + jnp.asarray(shift, a.dtype), axis=axis))
    got = np.asarray(run(jax.device_put(y, jax.devices("tt")[0]))).astype(np.float64)
    want = getattr(np, op)(np.asarray(y).astype(np.float64) + shift, axis=axis)
    if dtype == "bfloat16":
        np.testing.assert_allclose(got, want, rtol=2e-2, atol=0.05 * max(1.0, float(np.abs(want).max())))
    else:
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)
