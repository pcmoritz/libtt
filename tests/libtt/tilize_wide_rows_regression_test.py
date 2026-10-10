"""A row-major input is tilized before the first op on it. Inputs with fewer
tile rows than cores are tilized by width slices of their tile rows, one per
core, read at an offset into each row; more tile rows split by row. Every
element has its own value here, so a slice read from or written to the wrong
place changes the result.

- 8 and 40 rows pad their last tile row.
- 64 rows take slices: 2048 and 4096 columns in power-of-two counts, 2080 (65
  tiles) in 13 slices of 5 tiles, 2304 (72 tiles) in slices of 4. 2144 columns
  (67 tiles, prime) do not split and take whole tile rows on 2 cores.
- 3552 rows (111 tile rows) take whole tile rows, 2 per core, with 1 on the
  last core on a 110-core grid; 4096 rows divide evenly.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize(
    "rows,cols",
    [(8, 2048), (40, 2048), (64, 2048), (64, 4096), (64, 2080), (64, 2304), (64, 2144), (3552, 2048), (4096, 2048)],
)
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
