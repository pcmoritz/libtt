"""A row-major input is tilized before the first op on it. Wide inputs with
fewer tile rows than cores are tilized by width slices of their tile rows, one
per core, read at an offset into each row; more tile rows split by row. Every
element has its own value here, so a slice read from or written to the wrong
place changes the result. 8 and 40 rows pad their last tile row; 64 and 4096
rows fill theirs, the latter taking whole tile rows per core.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("rows,cols", [(8, 2048), (40, 2048), (64, 2048), (64, 4096), (4096, 2048)])
@pytest.mark.parametrize("dtype", [jnp.int32, jnp.float32, jnp.bfloat16])
def test_tilize_wide_rows(rows, cols, dtype):
    index = np.arange(rows)[:, None] * cols + np.arange(cols)
    # Distinct values for int32 and float32; bf16 holds integers only up to
    # 256, so its values mix the row and column into that range.
    x = (index if dtype != jnp.bfloat16 else (np.arange(rows)[:, None] * 31 + np.arange(cols)) % 251 - 125).astype(
        dtype
    )
    actual = jax.jit(lambda x: x * 2)(jax.device_put(x, jax.devices("tt")[0]))
    np.testing.assert_array_equal(np.asarray(actual), x * 2)
