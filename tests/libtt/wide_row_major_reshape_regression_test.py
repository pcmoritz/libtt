"""A row-major reshape that changes rows stages whole source and destination
rows in L1 unless both are 16-byte aligned. torch.distributed's object
collectives flatten rows of pickled bytes that are neither aligned nor small
enough (gather_object writes four 227,093-byte pickles into the rows of a
buffer and flattens it), so such reshapes stream 16 KiB chunks of the
destination rows instead.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize(
    "width,dtype", [(227_093, np.uint8), (227_093, np.int32), (1_000_003, np.int32)]
)
def test_flatten_written_rows(width, dtype):
    rng = np.random.default_rng(0)
    pieces = rng.integers(0, 120, (4, width)).astype(dtype)

    def gather(pieces):
        buffer = jnp.full((4, width), 7, dtype)
        for i in range(4):
            buffer = buffer.at[i].set(pieces[i])
        return buffer.reshape(-1)

    got = jax.jit(gather)(jax.device_put(pieces, jax.devices("tt")[0]))
    np.testing.assert_array_equal(np.asarray(got), pieces.reshape(-1))


@pytest.mark.parametrize("dtype", [np.uint8, np.int32])
def test_split_vector_into_rows(dtype):
    # scatter_object_list's reverse: rows of a flat buffer, picked by a
    # dynamic index.
    width = 227_093
    flat = (np.arange(4 * width) % 113).astype(dtype)

    def split(flat, i):
        return flat.reshape(4, width)[i]

    device = jax.devices("tt")[0]
    for i in (0, 3):
        got = jax.jit(split)(jax.device_put(flat, device), jax.device_put(np.int32(i), device))
        np.testing.assert_array_equal(np.asarray(got), flat.reshape(4, width)[i])
