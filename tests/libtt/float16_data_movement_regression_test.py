"""Float16 data movement must be bit-exact. Gathers lower to ttnn.embedding,
which took bfloat16 tables, losing three mantissa bits, or to ttnn.gather, which
zeroed tiles with padding; the Float16 transpose kernels flushed subnormals and
-0 to +0. Embedding, gather, permute and transpose now move Float16 as raw
UInt16 bits.

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


def values(shape):
    x = np.random.default_rng(0).normal(size=shape).astype(np.float16)
    flat = x.reshape(-1)
    n = min(flat.size, SPECIAL.size)
    flat[:n] = SPECIAL[:n]
    return x


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
    assert_same_float16(got, x.T)
