"""lax.top_k keeps the lowest index first among equal values.

TTNN's top-k sort is unstable unless asked otherwise; libtt's runtime asks for
the stable bitonic network, whose descending sort orders ties by index. Rows
with many ties catch both an unstable sort and a tie order that follows the
sort direction instead.
"""

import jax
import numpy as np
import pytest
from jax import lax


def reference(x, k):
    indices = np.argsort(-x.astype(np.float64), axis=-1, kind="stable")[..., :k]
    return np.take_along_axis(x, indices, axis=-1), indices


@pytest.mark.parametrize("dtype", [np.float32, "bfloat16"])
# Narrow rows take the single-core kernel, wide ones the multi-core kernels.
@pytest.mark.parametrize("shape,k", [((4, 64), 8), ((3, 256), 32), ((2, 4096), 64), ((2, 8192), 32)])
def test_top_k_ties_keep_lowest_index(dtype, shape, k):
    rng = np.random.default_rng(shape[-1] + k)
    # Few distinct values, so most rows have long runs of ties.
    x = rng.integers(-4, 5, shape).astype(np.float32).astype(dtype)
    values, indices = jax.jit(lambda a: lax.top_k(a, k))(jax.device_put(x, jax.devices("tt")[0]))
    want_values, want_indices = reference(np.asarray(x), k)
    np.testing.assert_array_equal(np.asarray(values).astype(np.float32), want_values.astype(np.float32))
    np.testing.assert_array_equal(np.asarray(indices), want_indices)
