"""Float16 data movement must be bit-exact. Gathers lower to ttnn.embedding,
which took bfloat16 tables, losing three mantissa bits, or to ttnn.gather, which
zeroed tiles with padding; the Float16 transpose kernels flushed subnormals and
-0 to +0, as did where. Embedding, gather, permute, transpose and where now move
Float16 as raw UInt16 bits.

Results must be bit-identical, including signed zeros, subnormals and
infinities. References come from NumPy on the host (jnp would also run on the
device).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Values float16 holds but bfloat16 does not, or that conversions tend to lose.
SPECIAL = np.array(
    [
        0.0,
        -0.0,
        np.inf,
        -np.inf,
        np.nan,
        65504.0,  # largest finite float16
        -65504.0,
        1.0009765625,  # 1 + 2**-10: one float16 ulp above 1, not a bfloat16
        2.0**-14,  # smallest normal
        2.0**-24,  # smallest subnormal
        -(2.0**-24),
        3 * 2.0**-24,
    ],
    np.float16,
)


def values(shape, seed=0):
    """Normal values with the special values scattered over every tile."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=shape).astype(np.float16)
    flat = x.reshape(-1)
    positions = rng.choice(flat.size, size=min(flat.size, 8 * SPECIAL.size), replace=False)
    flat[positions] = np.resize(SPECIAL, positions.size)
    return x


def assert_same_bits(actual, expected):
    """Bitwise, including NaN payloads, for ops that only move values."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.dtype == expected.dtype == np.float16
    np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))


# Tolerates NaN payload changes, for paths that may convert values.
def assert_same_float16(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.dtype == expected.dtype == np.float16
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    numbers = ~np.isnan(expected)
    np.testing.assert_array_equal(actual[numbers].view(np.uint16), expected[numbers].view(np.uint16))


# Each op as f(numpy_module, x, order), so NumPy and jax.numpy run the same code.
OPS = {
    "flip": lambda m, x, order: m.flip(x, 0),
    "take": lambda m, x, order: m.take(x, np.array([2, 0, 1, 2]), axis=0),
    "take_along_axis": lambda m, x, order: m.take_along_axis(x, order, axis=0),
    "take_along_last_axis": lambda m, x, order: m.take_along_axis(x, order, axis=-1),
    "rot90": lambda m, x, order: m.rot90(x.reshape(x.shape[0], -1)),
}


# One tile, and several with partial tiles in both dimensions.
@pytest.mark.parametrize("op", sorted(OPS))
@pytest.mark.parametrize("shape", [(3,), (5, 7), (33, 65)], ids=["1d", "2d", "multi_tile"])
def test_float16_data_movement_is_exact(op, shape):
    x = values(shape)
    # A random permutation along the gathered axis.
    axis = -1 if op == "take_along_last_axis" else 0
    order = np.argsort(np.random.default_rng(1).random(shape), axis=axis)
    expected = OPS[op](np, x, order)
    got = jax.jit(lambda v: OPS[op](jnp, v, jnp.asarray(order)))(jax.device_put(x, jax.devices("tt")[0]))
    assert_same_float16(got, expected)


# Every float16 bit pattern through a transpose, with full tiles and with partial tile rows wide
# enough for the block tilize factory.
@pytest.mark.parametrize("shape", [(256, 256), (16, 4096)], ids=["tiles", "partial_tile_rows"])
def test_float16_transpose_all_bit_patterns(shape):
    x = np.arange(1 << 16, dtype=np.uint16).view(np.float16).reshape(shape)
    got = jax.jit(lambda v: v.T)(jax.device_put(x, jax.devices("tt")[0]))
    assert_same_bits(got, x.T)


# Higher-rank permutations run on the HC and CN transpose factories, padding partial tiles.
@pytest.mark.parametrize("perm", [(0, 2, 1, 3), (1, 0, 2, 3), (3, 1, 0, 2)], ids=["hc", "cn", "mixed"])
def test_float16_permute_is_exact(perm):
    x = values((2, 3, 33, 65))
    got = jax.jit(lambda v: jnp.transpose(v, perm))(jax.device_put(x, jax.devices("tt")[0]))
    assert_same_bits(got, np.transpose(x, perm))


# where selects raw bits too. The predicate and both payloads come in at runtime. The first elements
# select every special value from the true branch, the next ones every special value from the false
# branch; the rest use a random predicate.
@pytest.mark.parametrize("shape", [(30,), (33, 65)], ids=["one_tile", "multi_tile"])
def test_float16_where_is_exact(shape):
    n = SPECIAL.size
    a, b = values(shape, seed=1), values(shape, seed=2)
    p = np.random.default_rng(3).random(shape) < 0.5
    flat_a, flat_b, flat_p = a.reshape(-1), b.reshape(-1), p.reshape(-1)
    flat_p[:n], flat_a[:n] = True, SPECIAL
    flat_p[n : 2 * n], flat_b[n : 2 * n] = False, SPECIAL
    dev = jax.devices("tt")[0]
    got = jax.jit(lambda p, a, b: jnp.where(p, a, b))(*(jax.device_put(v, dev) for v in (p, a, b)))
    assert_same_bits(got, np.where(p, a, b))


# A predicate computed by an earlier program reaches where as a device tensor; where keeps it
# bfloat16 instead of converting it to float16.
def test_float16_where_device_predicate():
    dev = jax.devices("tt")[0]
    c = np.array([0.0, -1.0, 0.5, 0.0, 2.0], np.float32)
    a = values((5,))
    p = jax.jit(lambda c: c != 0)(jax.device_put(c, dev))
    got = jax.jit(lambda p, a, s: jnp.where(p, a, s))(p, jax.device_put(a, dev), jax.device_put(np.float16(0), dev))
    assert_same_bits(got, np.where(c != 0, a, np.float16(0)))
