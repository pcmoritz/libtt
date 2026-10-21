"""Complex contractions reuse the real kernels and existing real/imag packing."""

import jax
import numpy as np
import pytest


@pytest.mark.parametrize("level", ["O0", "O1"])
@pytest.mark.parametrize("precision", ["default", "highest"])
@pytest.mark.parametrize("op", ["dot", "batched_dot", "conv1d", "conv2d"])
def test_complex_contraction(op, precision, level):
    rng = np.random.default_rng(7)
    if op == "dot":
        shapes = ((13,), (13,))
        run = lambda x, w: jax.lax.dot_general(
            x, w, (((0,), (0,)), ((), ())), precision=precision
        )
    elif op == "batched_dot":
        shapes = ((2, 3, 4), (2, 4, 5))
        run = lambda x, w: jax.lax.dot_general(
            x, w, (((2,), (1,)), ((0,), (0,))), precision=precision
        )
    else:
        shapes = ((1, 9, 2), (3, 2, 3)) if op == "conv1d" else (
            (1, 5, 7, 2), (2, 3, 2, 3)
        )
        dims = ("NWC", "WIO", "NWC") if op == "conv1d" else (
            "NHWC", "HWIO", "NHWC"
        )
        run = lambda x, w: jax.lax.conv_general_dilated(
            x, w, (1,) * (len(shapes[0]) - 2), "SAME",
            dimension_numbers=dims, precision=precision,
        )
    args = [
        (rng.normal(size=s) + 1j * rng.normal(size=s)).astype(np.complex64)
        for s in shapes
    ]
    expected = jax.jit(run)(*[jax.device_put(x, jax.devices("cpu")[0]) for x in args])
    actual = jax.jit(run, compiler_options={"optimization_level": level})(
        *[jax.device_put(x, jax.devices("tt")[0]) for x in args]
    )
    assert actual.dtype == np.complex64
    tolerance = 1e-5 if precision == "highest" else 5e-3
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
