"""lax.top_k keeps the lowest index first among equal values.

TTNN's top-k sort is unstable unless asked otherwise; libtt's runtime asks for
the stable bitonic network, whose descending sort orders ties by index. Rows
with many ties catch both an unstable sort and a tie order that follows the
sort direction instead. Up to two tile rows from 128 wide, such as a
mixture-of-experts router's top-8 of 128 or 256 experts, take the multi-core
kernel, which keeps ties in index order by sorting fused value and index keys
(BF16) or with the comparator-stable network (FP32).
"""

import jax
import numpy as np
import pytest
from jax import lax


def reference(x, k):
    indices = np.argsort(-x.astype(np.float64), axis=-1, kind="stable")[..., :k]
    return np.take_along_axis(x, indices, axis=-1), indices


@pytest.mark.parametrize("dtype", [np.float32, "bfloat16"])
# Rows of 2 to 256 tiles, including router widths and widths upstream sends to the multi-core kernels.
@pytest.mark.parametrize(
    "shape,k",
    [
        ((4, 64), 8),
        # Router widths take the multi-core kernel for up to two tile rows. k = 64 at widths 128 and
        # 256 gives each core exactly k elements, and two tile rows of 128 merge two cores per row;
        # 33 rows with k = 64 take both paths at once.
        ((1, 128), 8),
        ((1, 128), 16),
        ((32, 128), 8),
        ((33, 128), 8),
        ((1, 128), 64),
        ((33, 128), 64),
        ((1, 256), 8),
        ((1, 256), 32),
        ((1, 256), 64),
        ((33, 256), 64),
        ((3, 256), 32),
        ((2, 4096), 64),
        ((2, 8192), 32),
    ],
)
def test_top_k_ties_keep_lowest_index(dtype, shape, k):
    rng = np.random.default_rng(shape[-1] + k)
    # Few distinct values, so most rows have long runs of ties.
    x = rng.integers(-4, 5, shape).astype(np.float32).astype(dtype)
    values, indices = jax.jit(lambda a: lax.top_k(a, k))(jax.device_put(x, jax.devices("tt")[0]))
    want_values, want_indices = reference(np.asarray(x), k)
    np.testing.assert_array_equal(np.asarray(values).astype(np.float32), want_values.astype(np.float32))
    np.testing.assert_array_equal(np.asarray(indices), want_indices)

