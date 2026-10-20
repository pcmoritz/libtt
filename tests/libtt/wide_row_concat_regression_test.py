"""Concatenations along a last dimension that tiles don't divide run row-major,
and TT-NN stages each output row whole in L1. torch.distributed's object
collectives concatenate megabytes of pickled bytes in one row; such rows are
concatenated as rows of g elements, g dividing every input's last dimension.

- Four 1,135,488-byte pickles (a TorchTitan checkpoint plan) concatenate as
  rows of 47,312 bytes.
- Rows of 50,000 and 30,000 float32 values concatenate as rows of 10,000.
- Widths without a common divisor keep whole rows, which still fit L1.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize(
    "widths,dtype",
    [
        ([1_135_488] * 4, np.uint8),
        ([50_000, 30_000], np.float32),
        ([40_001, 17_003], np.int32),
    ],
)
def test_wide_row_concat(widths, dtype):
    rng = np.random.default_rng(0)
    if dtype == np.uint8:
        xs = [rng.integers(0, 256, (w,), dtype=np.uint8) for w in widths]
    else:
        xs = [(rng.integers(-1000, 1000, (2, w))).astype(dtype) for w in widths]
    device = jax.devices("tt")[0]
    out = jax.jit(lambda *xs: jnp.concatenate(xs, axis=-1))(*(jax.device_put(x, device) for x in xs))
    np.testing.assert_array_equal(np.asarray(out), np.concatenate(xs, axis=-1))
