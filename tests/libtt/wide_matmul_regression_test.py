"""Matmuls with outputs many times wider than tall.

ttnn runs them on a 1D program config: each core computes all rows of a share
of the columns, in blocks that divide its share when they don't fit L1, and
reads its inputs once per block row and column. A share with few divisors (71
tiles of Qwen3.5's 248,320-column logits over 110 cores) left 4 x 71 blocks that
read the weights 16 times; the share now rounds up to a multiple of 4 when that
reads less, on fewer cores, the last of which computes fewer columns.
"""

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

ROWS, DEPTH = 2048, 256


@pytest.mark.parametrize("transpose", [False, True], ids=["w", "w_transposed"])
@pytest.mark.parametrize("columns", [248320, 151936])
def test_wide_matmul(columns, transpose):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((ROWS, DEPTH)).astype(ml_dtypes.bfloat16)
    w = (rng.standard_normal((columns, DEPTH) if transpose else (DEPTH, columns)) * 0.1).astype(ml_dtypes.bfloat16)
    out = jax.jit(lambda x, w: jnp.dot(x, w.T if transpose else w))(x, w)
    want = x.astype(np.float64) @ (w.T if transpose else w).astype(np.float64)
    error = np.max(np.abs(np.asarray(out).astype(np.float64) - want)) / np.max(np.abs(want))
    assert out.shape == (ROWS, columns) and error < 1e-2
