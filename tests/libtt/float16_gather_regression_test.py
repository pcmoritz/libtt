"""Gathers lower to ttnn.embedding, which takes bfloat16 or float32 tables, or
to ttnn.gather, which returns zeros for float16. Float16 data is gathered as
float32 in both: as bfloat16 tables lost three mantissa bits, so flip, take and
friends rounded float16 values, and take_along_axis returned zeros.

References come from NumPy on the host (jnp would also run on the device).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

INDICES = np.array([2, 0, 1, 2])

# Each op as f(numpy_module, x, order), so NumPy and jax.numpy run the same code.
OPS = {
    "flip": lambda m, x, order: m.flip(x, 0),
    "take": lambda m, x, order: m.take(x, INDICES, axis=0),
    "take_along_axis": lambda m, x, order: m.take_along_axis(x, order, axis=0),
    "rot90": lambda m, x, order: m.rot90(x.reshape(x.shape[0], -1)),
}


@pytest.mark.parametrize("op", sorted(OPS))
@pytest.mark.parametrize("shape", [(3,), (5, 7)], ids=["1d", "2d"])
def test_float16_gather_is_exact(op, shape):
    x = np.random.default_rng(0).normal(size=shape).astype(np.float16)
    order = np.argsort(-x.astype(np.float32), axis=0)
    expected = OPS[op](np, x, order)
    got = jax.jit(lambda v: OPS[op](jnp, v, jnp.asarray(order)))(jax.device_put(x, jax.devices("tt")[0]))
    assert got.dtype == np.float16
    np.testing.assert_array_equal(np.asarray(got), expected)
