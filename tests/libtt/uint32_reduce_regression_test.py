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


@pytest.mark.parametrize("op", [jnp.max, jnp.min], ids=["max", "min"])
@pytest.mark.parametrize("dtype", [np.uint32, np.int32], ids=["u32", "i32"])
# Several tiles with partial ones, so tile folding and padding identities matter.
@pytest.mark.parametrize("axis", [0, 1, 2, (1, 2), None], ids=["d0", "d1", "d2", "d12", "all"])
def test_integer_max_min(op, dtype, axis):
    x = values(dtype, (3, 40, 70))
    np.testing.assert_array_equal(run(lambda v: op(v, axis=axis), x), op(x, axis=axis))


# A single extreme value in a later tile decides each result.
@pytest.mark.parametrize("op", [jnp.max, jnp.min], ids=["max", "min"])
@pytest.mark.parametrize("axis", [1, 2], ids=["d1", "d2"])
def test_uint32_decisive_value(op, axis):
    x = np.full((2, 40, 70), 0x80000000 if op is jnp.max else 0x7FFFFFFF, np.uint32)
    x[1, 39, 69] = 0xFFFFFFFF if op is jnp.max else 0
    np.testing.assert_array_equal(run(lambda v: op(v, axis=axis), x), op(x, axis=axis))


def test_uint32_ptp_and_nanmin():
    x = values(np.uint32, (4, 33), seed=1)
    np.testing.assert_array_equal(run(lambda v: jnp.ptp(v, axis=1), x), np.ptp(x, axis=1))
    np.testing.assert_array_equal(run(lambda v: jnp.nanmin(v, axis=0), x), np.nanmin(x, axis=0))
