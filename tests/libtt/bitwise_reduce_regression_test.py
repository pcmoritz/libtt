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
# reduction folds tiles and must pad them with the identity. Random bits would
# AND to zero and OR to all ones, so most bits stay set for AND and clear for OR.
@pytest.mark.parametrize("op", sorted(OPS))
@pytest.mark.parametrize("dims", [(0,), (1,), (2,), (1, 2)], ids=["d0", "d1", "d2", "d12"])
def test_multi_tile_bitwise_reduce(op, dims):
    lax_op, np_op = OPS[op]
    x = random_integers(np.uint32, (35, 70, 45), seed=1)
    if op == "and":
        x |= np.uint32(0xFFFF7FFE)
    elif op == "or":
        x &= np.uint32(0x00008001)
    np.testing.assert_array_equal(run(lambda v: lax_op(v, dims), x), reference(np_op, x, dims))


# A single element decides the result, in a later tile of the reduced
# dimension (and of the partial last tile for d1).
@pytest.mark.parametrize("op", sorted(OPS))
@pytest.mark.parametrize("dim", [0, 1, 2], ids=["d0", "d1", "d2"])
def test_decisive_element_in_later_tile(op, dim):
    lax_op, np_op = OPS[op]
    x = np.full((35, 70, 45), 0xFFFFFFFF if op == "and" else 0, dtype=np.uint32)
    index = [2, 3, 4]
    index[dim] = x.shape[dim] - 1
    x[tuple(index)] ^= np.uint32(0x00400000)
    np.testing.assert_array_equal(run(lambda v: lax_op(v, (dim,)), x), np_op(x, axis=dim))


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
@pytest.mark.parametrize(
    "op, init",
    [("and", 0x0FF00FF0), ("or", 0x10000001), ("xor", 0x12345678)],
    ids=["and", "or", "xor"],
)
def test_bitwise_reduce_with_init(op, init):
    lax_op, np_op = OPS[op]
    x = random_integers(np.uint32, (3, 4, 5), seed=4)
    reducer = {"and": lax.bitwise_and, "or": lax.bitwise_or, "xor": lax.bitwise_xor}[op]
    got = run(lambda v: lax.reduce(v, np.uint32(init), reducer, (1,)), x)
    np.testing.assert_array_equal(got, np_op(np.concatenate([x, np.full((3, 1, 5), init, np.uint32)], axis=1), axis=1))


# Reducing an empty dimension gives the identity: all ones for AND.
@pytest.mark.parametrize("op", sorted(OPS))
@pytest.mark.parametrize("dtype", [np.int32, np.uint32], ids=["i32", "u32"])
def test_empty_bitwise_reduce(op, dtype):
    lax_op, np_op = OPS[op]
    x = np.zeros((3, 0, 5), dtype=dtype)
    np.testing.assert_array_equal(run(lambda v: lax_op(v, (1,)), x), np_op(x, axis=1))


# Only a body that is exactly the bitwise op lowers to the bitwise reduction:
# ~(a ^ b) is associative but is not xor.
def test_xnor_reducer_is_not_xor():
    x = np.zeros((3,), dtype=np.int32)
    xnor = lambda a, b: ~(a ^ b)
    try:
        got = run(lambda v: lax.reduce(v, np.int32(-1), xnor, (0,)), x)
    except Exception:
        return  # Not lowered, rather than miscompiled.
    np.testing.assert_array_equal(got, np.int32(0))
