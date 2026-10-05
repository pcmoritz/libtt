"""Single-channel convolutions with a dimension of 1.

jnp.convolve and jnp.correlate are 1D convolutions with one input and one
output channel. TTNN took any convolution with groups equal to its channel
counts for a depthwise one, so these ran on the 1D depthwise path, which
returned garbage for a single channel. Results must match NumPy.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax


def assert_close(got, want):
    got = np.asarray(got).astype(np.float64)
    scale = np.max(np.abs(want)) + 1
    # The convolution multiplies in BF16, so errors scale with the operands.
    np.testing.assert_allclose(got, want, atol=0.03 * scale, rtol=0.03)


@pytest.mark.parametrize("dtype", [np.float32, jnp.bfloat16])
@pytest.mark.parametrize("mode", ["full", "same", "valid"])
@pytest.mark.parametrize("fn", ["convolve", "correlate"])
@pytest.mark.parametrize("n,k", [(7, 3), (64, 5), (1000, 7)])
def test_numpy_convolutions(fn, mode, dtype, n, k):
    rng = np.random.default_rng(n + k)
    x = rng.standard_normal(n).astype(dtype)
    y = rng.standard_normal(k).astype(dtype)
    device = jax.devices("tt")[0]
    got = getattr(jnp, fn)(jax.device_put(x, device), jax.device_put(y, device), mode=mode)
    want = getattr(np, fn)(x.astype(np.float64), y.astype(np.float64), mode=mode)
    assert_close(got, want)


@pytest.mark.parametrize(
    "shape,kernel",
    [
        ((1, 1, 64), (1, 1, 5)),  # 1D, one channel.
        ((1, 1, 64, 1), (1, 1, 5, 1)),  # 2D with a width of 1.
        ((1, 1, 1, 64), (1, 1, 1, 5)),  # 2D with a height of 1.
        ((1, 4, 64), (4, 1, 5)),  # A real 1D depthwise convolution still uses its path.
    ],
)
def test_single_channel_convolutions(shape, kernel):
    rng = np.random.default_rng(len(shape))
    x = rng.standard_normal(shape).astype(np.float32)
    w = rng.standard_normal(kernel).astype(np.float32)
    spatial = len(shape) - 2
    groups = shape[1] if kernel[1] == 1 and shape[1] > 1 else 1
    conv = lambda a, b: lax.conv_general_dilated(a, b, (1,) * spatial, "VALID", feature_group_count=groups)
    got = jax.jit(conv)(*(jax.device_put(a, jax.devices("tt")[0]) for a in (x, w)))
    want = np.asarray(jax.jit(conv, device=jax.devices("cpu")[0])(x, w), np.float64)
    assert_close(got, want)
