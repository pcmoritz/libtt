"""Slices of megabyte vectors, as torch.distributed's object collectives take
them out of gathered or reduce-scattered pickles:

- TT-NN slices a tiled vector as one row of tiles ([1, N]) when the slice
  starts on a tile, instead of untilizing it into a single row-major page
  that has to fit in L1.
- Other starts slice row-major, in chunks of the row, also when the start
  isn't aligned to the buffer (a 194,799-element piece starts 60 bytes into
  an aligned block).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize(
    "size,begin,end",
    [
        (4 * 227_104, 227_104, 227_104 + 227_093),  # an unaligned piece of a gathered vector
        (4 * 227_104, 0, 227_093),
        (4 * 227_104, 3 * 227_104, 4 * 227_104),
        (4_096, 1_000, 3_000),  # an unaligned start keeps the row-major path
    ],
)
def test_slice_tiled_vector(size, begin, end):
    x = (np.arange(size) % 1009).astype(np.int32)
    got = jax.jit(lambda x: (x + 1)[begin:end])(jax.device_put(x, jax.devices("tt")[0]))
    np.testing.assert_array_equal(np.asarray(got), (x + 1)[begin:end])


@pytest.mark.parametrize("shape,begin,end", [((779_196,), 194_799, 389_598), ((2, 779_196), 194_799, 389_598)])
def test_slice_misaligned_wide_rows(shape, begin, end):
    x = (np.arange(np.prod(shape)) % 1009).astype(np.int32).reshape(shape)
    got = jax.jit(lambda x: (x + 1)[..., begin:end])(jax.device_put(x, jax.devices("tt")[0]))
    np.testing.assert_array_equal(np.asarray(got), (x + 1)[..., begin:end])
