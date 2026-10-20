"""An average pool written as a window sum divided by a scalar, as flax's
avg_pool does, crashed the compiler: the average-pool fusion read the float
divisor as an integer ("ElementsAttr does not provide iteration facilities for
type `APInt`")."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

CASES = [
    # dtype, NHWC shape, window, strides, padding, divisor
    (jnp.bfloat16, (1, 16, 16, 32), (1, 3, 3, 1), (1, 2, 2, 1), ((0, 0), (1, 1), (1, 1), (0, 0)), 9),
    (jnp.bfloat16, (2, 8, 8, 32), (1, 2, 2, 1), (1, 2, 2, 1), ((0, 0),) * 4, 2),
    # Wider than the float32 window slices take: ttnn pooling.
    (np.float32, (1, 4, 96, 32), (1, 1, 80, 1), (1, 1, 1, 1), ((0, 0),) * 4, 80),
]


@pytest.mark.parametrize("dtype,shape,window,strides,padding,divisor", CASES)
def test_avg_pool_scalar_divisor(dtype, shape, window, strides, padding, divisor):
    x = np.random.default_rng(0).standard_normal(shape).astype(dtype)

    def avg_pool(x):
        total = lax.reduce_window(x, jnp.zeros((), x.dtype), lax.add, window, strides, padding)
        return total / divisor

    got = jax.jit(avg_pool)(jax.device_put(x, jax.devices("tt")[0]))
    expected = jax.jit(avg_pool, backend="cpu")(x)
    np.testing.assert_allclose(np.asarray(got, np.float32), np.asarray(expected, np.float32), rtol=2e-2, atol=2e-2)
