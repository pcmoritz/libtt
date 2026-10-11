"""uint32 max and min reductions run on ttnn's exact integer reduce, with an
unsigned compare-and-swap (uint32 min used to go through a float negate and
came back off by 0x80000000; max was rounded through float32). Values cover
the full range, including both sides of the top bit.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

BOUNDARIES = {
    np.uint32: [0, 1, 0x7FFFFFFF, 0x80000000, 0x80000001, 0xFFFFFFFE, 0xFFFFFFFF],
    np.int32: [-(2**31), -(2**31) + 1, -1, 0, 1, 2**31 - 2, 2**31 - 1],
}


def values(dtype, shape, seed=0):
    info = np.iinfo(dtype)
    x = np.random.default_rng(seed).integers(info.min, info.max, size=shape, dtype=np.int64, endpoint=True)
    flat = x.reshape(-1)
    flat[: len(BOUNDARIES[dtype])] = BOUNDARIES[dtype]
    return x.astype(dtype)


def run(f, x):
    return np.asarray(jax.jit(f)(jax.device_put(x, jax.devices("tt")[0])))


# NumPy references: jnp would run the reference on the TT device too.
OPS = [(jnp.max, np.max), (jnp.min, np.min)]


@pytest.mark.parametrize("jax_op, numpy_op", OPS, ids=["max", "min"])
@pytest.mark.parametrize("dtype", [np.uint32, np.int32], ids=["u32", "i32"])
# Several tiles with partial ones, so tile folding and padding identities matter.
@pytest.mark.parametrize("axis", [0, 1, 2, (1, 2), None], ids=["d0", "d1", "d2", "d12", "all"])
def test_integer_max_min(jax_op, numpy_op, dtype, axis):
    x = values(dtype, (3, 40, 70))
    np.testing.assert_array_equal(run(lambda v: jax_op(v, axis=axis), x), numpy_op(x, axis=axis))


# A single extreme value in a later tile decides each result.
@pytest.mark.parametrize("jax_op, numpy_op", OPS, ids=["max", "min"])
@pytest.mark.parametrize("axis", [1, 2], ids=["d1", "d2"])
def test_uint32_decisive_value(jax_op, numpy_op, axis):
    x = np.full((2, 40, 70), 0x80000000 if numpy_op is np.max else 0x7FFFFFFF, np.uint32)
    x[1, 39, 69] = 0xFFFFFFFF if numpy_op is np.max else 0
    np.testing.assert_array_equal(run(lambda v: jax_op(v, axis=axis), x), numpy_op(x, axis=axis))


# One pair per compare-and-swap case: both top bits clear, different, both set.
PAIRS = [(0, 1), (0x7FFFFFFF, 0x80000000), (0x80000000, 0x80000001)]


@pytest.mark.parametrize("jax_op, numpy_op", OPS, ids=["max", "min"])
@pytest.mark.parametrize("pair", PAIRS, ids=["clear", "mixed", "set"])
@pytest.mark.parametrize("reverse", [False, True], ids=["ab", "ba"])
@pytest.mark.parametrize("axis", [0, 1], ids=["H", "W"])
def test_uint32_compare_pairs(jax_op, numpy_op, pair, reverse, axis):
    x = np.array([pair[::-1] if reverse else pair], np.uint32)
    if axis == 0:
        x = x.T
    np.testing.assert_array_equal(run(lambda v: jax_op(v, axis=axis), x), numpy_op(x, axis=axis))


# A full reduction of a single tile, which would otherwise take the single-core
# scalar path; values on both sides of the top bit.
@pytest.mark.parametrize("jax_op, numpy_op", OPS, ids=["max", "min"])
def test_uint32_single_tile_full_reduction(jax_op, numpy_op):
    x = np.array([[0x7FFFFFFF, 0x80000000, 1], [0xFFFFFFFE, 0, 0x80000001]], np.uint32)
    np.testing.assert_array_equal(run(lambda v: jax_op(v), x), numpy_op(x))


# The padding identity must not win: a padded slice whose maximum is INT32_MIN.
def test_int32_max_of_int32_min():
    x = np.full((2, 33), np.iinfo(np.int32).min, np.int32)
    np.testing.assert_array_equal(run(lambda v: jnp.max(v, axis=1), x), np.max(x, axis=1))


def test_uint32_ptp_and_nanmin():
    x = values(np.uint32, (4, 33), seed=1)
    np.testing.assert_array_equal(run(lambda v: jnp.ptp(v, axis=1), x), np.ptp(x, axis=1))
    np.testing.assert_array_equal(run(lambda v: jnp.nanmin(v, axis=0), x), np.nanmin(x, axis=0))
