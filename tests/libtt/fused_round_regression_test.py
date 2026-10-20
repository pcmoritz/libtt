"""Rounding fused into binary ops.

The TTNN fusing pass folds a round into a neighboring binary op as an
activation, and ttnn's ROUND activation takes the number of decimals.
"""

import jax
import jax.numpy as jnp
import numpy as np


def test_round_fused_into_binary_ops():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((32, 32)) * 10).astype(np.float32)
    y = rng.standard_normal((32, 32)).astype(np.float32)
    rounded_input, rounded_output = jax.jit(
        lambda x, y: (jnp.round(x) * y, jnp.round(x * y))
    )(x, y)
    np.testing.assert_allclose(rounded_input, np.round(x) * y, rtol=1e-2)
    np.testing.assert_allclose(rounded_output, np.round(x * y), atol=1.0)
