"""Bitwise and/or/xor reductions of integers, and xor reductions of booleans,
have no ttnn reduction. They are lowered as a halving tree of elementwise ops
over each reduced dimension (lax.reduce_and/or/xor used to fail to legalize).
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


@pytest.mark.parametrize("op", sorted(OPS))
@pytest.mark.parametrize("dtype", [np.int32, np.uint32], ids=["i32", "u32"])
# Odd and even sizes, one and several dimensions.
@pytest.mark.parametrize("dims", [(0,), (1, 2), (0, 2), (0, 1, 2)], ids=["d0", "d12", "d02", "all"])
def test_integer_bitwise_reduce(op, dtype, dims):
    lax_op, np_op = OPS[op]
    info = np.iinfo(dtype)
    x = np.random.default_rng(0).integers(info.min, info.max, size=(3, 4, 5), dtype=np.int64, endpoint=True).astype(dtype)
    got = jax.jit(lambda v: lax_op(v, dims))(jax.device_put(x, jax.devices("tt")[0]))
    np.testing.assert_array_equal(np.asarray(got), reference(np_op, x, dims))


@pytest.mark.parametrize("dims", [(0,), (1, 2), (0, 1, 2)], ids=["d0", "d12", "all"])
def test_boolean_xor_reduce(dims):
    x = np.random.default_rng(1).random((3, 4, 5)) < 0.5
    got = jax.jit(lambda v: lax.reduce_xor(v, dims))(jax.device_put(x, jax.devices("tt")[0]))
    np.testing.assert_array_equal(np.asarray(got), reference(np.logical_xor.reduce, x, dims))
