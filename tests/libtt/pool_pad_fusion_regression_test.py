"""Pooling must preserve explicit padding that differs from its init value.

torch_tpu counts average-pool divisors by padding ones with ones. Folding
that pad into the sum reduction silently replaces the padded ones with zeros.
"""

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("level", ["O0", "O1"])
@pytest.mark.parametrize("dtype", [np.float32, jnp.bfloat16])
@pytest.mark.parametrize("pad_value", [0, 1])
def test_pool_preserves_padding(level, dtype, pad_value):
    x = np.ones((2, 3, 16), dtype)

    def counts(x):
        padded = jnp.pad(x, ((0, 0), (0, 0), (1, 1)), constant_values=pad_value)
        return lax.reduce_window(padded, dtype(0), lax.add, (1, 1, 3), (1, 1, 2),
                                 ((0, 0), (0, 0), (0, 1)))

    got = jax.jit(counts, compiler_options={"optimization_level": level})(
        jax.device_put(x, jax.devices("tt")[0]))
    expected = jax.jit(counts)(jax.device_put(x, jax.devices("cpu")[0]))
    # Native pooling computes in bfloat16, independently of the padding fix.
    np.testing.assert_allclose(np.asarray(got, np.float32), np.asarray(expected, np.float32),
                               rtol=0, atol=0.05)
