"""Rounding fused into binary ops.

The TTNN fusing pass folds a round into a neighboring binary op as an
activation, and ttnn's ROUND activation takes the number of decimals. The
inputs, multipliers and products are exact in bfloat16, so the results are
exact, with ties rounding to even.
"""

import jax
import jax.numpy as jnp
import numpy as np

# Halfway cases of both signs and values next to them.
VALUES = np.array(
    [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, -2.25, -0.75, 0.25, 1.75, 2.0, -3.0, 0.0, 4.5],
    np.float32,
)
# Rows of multipliers; times 3 and -1, the halfway cases stay halfway.
MULTIPLIERS = np.array([3.0, -1.0, 0.5, 2.0], np.float32)


def test_round_fused_into_binary_ops():
    x = np.tile(VALUES, (32, 2))
    y = np.repeat(np.tile(MULTIPLIERS, 8)[:, None], 32, axis=1)
    rounded_input, rounded_output = jax.jit(
        lambda x, y: (jnp.round(x) * y, jnp.round(x * y))
    )(x, y)
    np.testing.assert_array_equal(rounded_input, np.round(x) * y)
    np.testing.assert_array_equal(rounded_output, np.round(x * y))
