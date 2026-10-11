"""Bitwise and/or/xor reductions of integers, and xor reductions of booleans,
run on ttnn's reductions (ttnn.reduce_bitwise_and/or/xor) like sum and max
(lax.reduce_and/or/xor used to fail to legalize).
"""

import jax
import numpy as np
import pytest
from jax import lax

OPS = {
    "and": (lax.reduce_and, np.bitwise_and.reduce),
    "or": (lax.reduce_or, np.bitwise_or.reduce),
    "xor": (lax.reduce_xor, np.bitwise_xor.reduce),
}


def reference(np_op, x, dims):
    for dim in sorted(dims, reverse=True):
        x = np_op(x, axis=dim)
    return x


def random_integers(dtype, shape, seed=0):
    info = np.iinfo(dtype)
    rng = np.random.default_rng(seed)
    return rng.integers(info.min, info.max, size=shape, dtype=np.int64, endpoint=True).astype(dtype)


def run(fn, x):
    return np.asarray(jax.jit(fn)(jax.device_put(x, jax.devices("tt")[0])))


@pytest.mark.parametrize("op", sorted(OPS))
@pytest.mark.parametrize("dtype", [np.int32, np.uint32], ids=["i32", "u32"])
# An outer dimension, the tile dimensions, and combinations of them.
@pytest.mark.parametrize("dims", [(0,), (1,), (2,), (1, 2), (0, 2), (0, 1, 2)], ids=["d0", "d1", "d2", "d12", "d02", "all"])
def test_integer_bitwise_reduce(op, dtype, dims):
    lax_op, np_op = OPS[op]
    x = random_integers(dtype, (3, 4, 5))
    np.testing.assert_array_equal(run(lambda v: lax_op(v, dims), x), reference(np_op, x, dims))


# Several tiles along each reduced dimension, with partial tiles, so the
# reduction folds tiles and must pad them with the identity.
@pytest.mark.parametrize("op", sorted(OPS))
@pytest.mark.parametrize("dims", [(0,), (1,), (2,), (1, 2)], ids=["d0", "d1", "d2", "d12"])
def test_multi_tile_bitwise_reduce(op, dims):
    lax_op, np_op = OPS[op]
    x = random_integers(np.uint32, (35, 70, 45), seed=1)
    if op == "and":
        # Random bits AND to zero over many elements; keep most bits set.
        x |= np.uint32(0xFFFF7FFE)
    np.testing.assert_array_equal(run(lambda v: lax_op(v, dims), x), reference(np_op, x, dims))


@pytest.mark.parametrize("op", sorted(OPS))
@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.int16, np.uint16])
def test_narrow_integer_bitwise_reduce(op, dtype):
    lax_op, np_op = OPS[op]
    x = random_integers(dtype, (3, 4, 5), seed=2)
    np.testing.assert_array_equal(run(lambda v: lax_op(v, (0, 2)), x), reference(np_op, x, (0, 2)))


@pytest.mark.parametrize("dims", [(0,), (1, 2), (0, 1, 2)], ids=["d0", "d12", "all"])
def test_boolean_xor_reduce(dims):
    x = np.random.default_rng(3).random((3, 4, 5)) < 0.5
    np.testing.assert_array_equal(run(lambda v: lax.reduce_xor(v, dims), x), reference(np.logical_xor.reduce, x, dims))


# A non-identity init value is combined into the result.
def test_bitwise_or_reduce_with_init():
    x = random_integers(np.uint32, (3, 4, 5), seed=4) & np.uint32(0x0F0F0F0F)
    got = run(lambda v: lax.reduce(v, np.uint32(0x10000001), lax.bitwise_or, (1,)), x)
    np.testing.assert_array_equal(got, np.bitwise_or.reduce(x, axis=1) | np.uint32(0x10000001))
