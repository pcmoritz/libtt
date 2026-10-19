"""Boolean constants survive VHLO serialization.

Frontends pack i1 constants into bits (a splat is one 0xFF byte), which is how
this MLIR stores them too, so the reader must not unpack them.
"""

import jax
import jax.numpy as jnp
import numpy as np

MASK = np.array([True, False, True, False, False, True, True, False, True])


def test_bool_constants():
    out = jax.jit(
        lambda x: (
            jnp.where(jnp.asarray(MASK), x, 0.0),
            jnp.where(jnp.ones((2, 2), bool), 1.0, 0.0),
            jnp.where(jnp.asarray(MASK[:3]), 1.0, 0.0),
        )
    )(jnp.arange(1.0, 10.0))
    np.testing.assert_array_equal(out[0], np.where(MASK, np.arange(1.0, 10.0), 0.0))
    np.testing.assert_array_equal(out[1], np.ones((2, 2)))
    np.testing.assert_array_equal(out[2], MASK[:3].astype(np.float32))
