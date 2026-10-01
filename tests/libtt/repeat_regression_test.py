"""Repeating row-major tensors whose rows are wider than L1."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize(
    "width,dtype",
    [
        # A vocabulary-wide index row, like a sampler's token ids: ~1 MB rows.
        (248320, np.int32),
        # Rows that end in a partial chunk, including an unaligned length.
        (100003, np.int32),
        (131075, jnp.bfloat16),
        (8, np.int32),
    ],
)
def test_repeat_wide_rows(width, dtype):
    # Broadcasting one row to three lowers to a row-major repeat, which used to
    # hold whole rows in L1 and fail once a row outgrew it.
    row = (np.arange(width) % 1000).astype(dtype).reshape(1, width)
    out = jax.jit(lambda x: jnp.broadcast_to(x, (3, width)))(
        jax.device_put(row, jax.devices("tt")[0])
    )
    np.testing.assert_array_equal(np.asarray(out), np.broadcast_to(row, (3, width)))
