"""FP32 convolutions at Precision.HIGHEST run as im2col and an FP32 matmul on
the vector unit, with FP32 products and sums (the TTNN convolution reads FP32
inputs as TF32, ~1e-3 relative error). jax.scipy.linalg.toeplitz, hankel and
convolution_matrix use such convolutions."""

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np
import pytest
from jax import lax


def on(device, f, *args):
    return np.asarray(jax.jit(f)(*(jax.device_put(a, jax.devices(device)[0]) for a in args)))


def assert_exact(f, *args):
    np.testing.assert_allclose(on("tt", f, *args), on("cpu", f, *args), rtol=2e-6, atol=1e-6)


def values(shape, seed=0):
    return np.random.default_rng(seed).normal(size=shape).astype(np.float32)


def test_toeplitz_and_hankel():
    c, r = values((7,)), values((9,), seed=1)
    assert_exact(jsl.toeplitz, c, r)
    assert_exact(jsl.hankel, c, r)


@pytest.mark.parametrize("mode", ["full", "same", "valid"])
def test_convolution_matrix(mode):
    assert_exact(lambda a: jsl.convolution_matrix(a, 6, mode), values((4,)))


# Padding, strides, kernel dilation and a non-default layout.
def test_conv_general_dilated_highest():
    x, w = values((2, 3, 9, 8)), values((4, 3, 3, 2), seed=1)

    def conv(a, b):
        return lax.conv_general_dilated(
            a, b, window_strides=(2, 1), padding=[(1, 2), (0, 1)], rhs_dilation=(1, 2),
            dimension_numbers=("NCHW", "OIHW", "NHWC"), precision=lax.Precision.HIGHEST)

    assert_exact(conv, x, w)
