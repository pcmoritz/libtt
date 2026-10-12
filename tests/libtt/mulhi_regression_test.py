"""lax.mulhi on uint32 runs as a Blackhole kernel that gives the high half of
the 64-bit product (it used to be 0: the decomposition widens to 64 bits, which
are narrowed back to 32). Philox random bits depend on it.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax._src.random.philox2x32 import philox2x32_p


def run(f, *args):
    return np.asarray(jax.jit(f)(*(jax.device_put(a, jax.devices("tt")[0]) for a in args)))


# Values around the 23-bit split the kernel uses, and the extremes.
EDGES = np.array([0, 1, 0x7FFFFF, 0x800000, 0xFFFFFF, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF], np.uint32)


def test_mulhi():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2**32, size=(4, 64), dtype=np.uint64).astype(np.uint32)
    b = rng.integers(0, 2**32, size=(4, 64), dtype=np.uint64).astype(np.uint32)
    a[0] = np.repeat(EDGES, 8)
    b[0] = np.tile(EDGES, 8)
    expected = ((a.astype(np.uint64) * b.astype(np.uint64)) >> np.uint64(32)).astype(np.uint32)
    np.testing.assert_array_equal(run(lax.mulhi, a, b), expected)


# Known-answer vectors from JAX's random_impl_test.
@pytest.mark.parametrize(
    "key, counter, expected",
    [
        (0x00000000, (0x00000000, 0x00000000), (0xFF1DAE59, 0x6CD10DF2)),
        (0xFFFFFFFF, (0xFFFFFFFF, 0xFFFFFFFF), (0x2C3F628B, 0xAB4FD7AD)),
        (0x13198A2E, (0x243F6A88, 0x85A308D3), (0xDD7CE038, 0xF62A4C12)),
    ],
)
def test_philox2x32_known_answers(key, counter, expected):
    got = run(lambda k, c0, c1: jnp.stack(philox2x32_p.bind(k, c0, c1)), np.uint32(key), *map(np.uint32, counter))
    np.testing.assert_array_equal(got, np.array(expected, np.uint32))
