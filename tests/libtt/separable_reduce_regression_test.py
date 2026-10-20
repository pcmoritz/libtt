"""Variadic reduces whose results reduce independently, side by side.

PyTorch frontends reduce a tensor's minimum and maximum in one reduce over
(x, x), whose body keeps the minimum of one pair and the maximum of the other,
starting from the highest and lowest finite values.
"""

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

INFO = np.finfo(np.float32)


def _aminmax(x):
    def body(acc, element):
        return jnp.minimum(acc[0], element[0]), jnp.maximum(acc[1], element[1])

    init = (jnp.array(INFO.max, x.dtype), jnp.array(INFO.min, x.dtype))
    return lax.reduce((x, x), init, body, (1,))


def test_min_and_max_together():
    x = np.random.default_rng(0).standard_normal((4, 6)).astype(np.float32)
    # The inits take part: a row of +inf has the minimum finfo.max.
    x[1] = np.inf
    x[2] = -np.inf
    low, high = jax.jit(_aminmax)(x)
    np.testing.assert_array_equal(low, np.minimum(x.min(axis=1), INFO.max))
    np.testing.assert_array_equal(high, np.maximum(x.max(axis=1), INFO.min))
