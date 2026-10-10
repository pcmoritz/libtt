"""Float16 values must survive a tilize that also changes the dtype.

A Float16 concatenation whose last dimension is not a multiple of 32 runs on
row-major Float32 and tilizes the result back to Float16. Tilize converted the
dtype in the packer, which turned infinities into NaN and flushed subnormals
to zero, so jnp.concatenate([[-inf], x]) started with NaN. Tilize now keeps
the input dtype and converts with the exact SFPU typecast.
"""

import jax
import jax.numpy as jnp
import numpy as np

# Every Float16 bit pattern, as a 256 x 256 array.
ALL_FLOAT16 = np.arange(1 << 16, dtype=np.uint16).view(np.float16).reshape(256, 256)

SPECIAL = np.float16([-np.inf, np.inf, np.nan, -0.0, 6e-8, -3.05e-5, 65504.0, 1.0])


def assert_same_float16(actual, expected):
    """Bit-for-bit equality, with every NaN counted as equal to every other."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.dtype == expected.dtype == np.float16
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    numbers = ~np.isnan(expected)
    np.testing.assert_array_equal(actual[numbers].view(np.uint16), expected[numbers].view(np.uint16))


def test_every_float16_through_unaligned_concat():
    tail = np.resize(SPECIAL, (256, 3))
    device = jax.devices("tt")[0]
    run = jax.jit(lambda x, y: jnp.concatenate([x, y], axis=1))
    actual = run(jax.device_put(ALL_FLOAT16, device), jax.device_put(tail, device))
    assert_same_float16(actual, np.concatenate([ALL_FLOAT16, tail], axis=1))


def test_rank1_concat():
    device = jax.devices("tt")[0]
    run = jax.jit(lambda x, y: jnp.concatenate([x, y]))
    actual = run(jax.device_put(SPECIAL, device), jax.device_put(SPECIAL[1:], device))
    assert_same_float16(actual, np.concatenate([SPECIAL, SPECIAL[1:]]))

