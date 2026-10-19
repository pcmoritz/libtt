"""A while loop that writes rows of its own state keeps the state's layout.

The rows ops read their table row-major. Forcing a loop body's argument
row-major, while the body still returned the state tiled, handed rows_scatter a
tiled table from the second iteration on.
"""

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np


def test_loop_scatters_rows_into_its_state():
    def body(i, table):
        update = jnp.full((1, 1), i, table.dtype) * 2.0
        return lax.dynamic_update_slice(table, update, (0, i))

    out = jax.jit(lambda table: lax.fori_loop(0, 5, body, table))(
        jnp.zeros((1, 5), jnp.float32)
    )
    np.testing.assert_array_equal(out, [[0.0, 2.0, 4.0, 6.0, 8.0]])
