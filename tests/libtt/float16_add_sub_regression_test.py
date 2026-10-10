"""Float16 addition and subtraction must not flush subnormal inputs to zero.

The FPU kernel of ttnn.add and ttnn.subtract flushes Float16 subnormals (below
6.1e-5) to zero as it unpacks its inputs, so 1e-5 - 0 was 0 and jnp.isclose of
tiny values was always true. Float16 now takes the SFPU kernel, which widens
both inputs to Float32 exactly; the Float16 result is then correctly rounded.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Every Float16 bit pattern, as a 256 x 256 array.
ALL_FLOAT16 = np.arange(1 << 16, dtype=np.uint16).view(np.float16).reshape(256, 256)

OPERANDS = [0.0, -0.0, 6e-8, 3.05e-5, -1e-5, 6.1e-5, 1.0, -65504.0, np.inf, np.nan]


def assert_same_float16(actual, expected):
    """Bit-for-bit equality, with every NaN counted as equal to every other."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.dtype == expected.dtype == np.float16
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    numbers = ~np.isnan(expected)
    np.testing.assert_array_equal(actual[numbers].view(np.uint16), expected[numbers].view(np.uint16))


@pytest.mark.parametrize("op", ["add", "subtract"])
@pytest.mark.parametrize("operand", OPERANDS)
def test_every_float16_against_operand(op, operand):
    fn = getattr(jnp, op)
    b = np.full_like(ALL_FLOAT16, operand)
    device = jax.devices("tt")[0]
    run = jax.jit(lambda x, y: (fn(x, y), fn(y, x)))
    forward, backward = run(jax.device_put(ALL_FLOAT16, device), jax.device_put(b, device))
    with np.errstate(over="ignore", invalid="ignore"):
        assert_same_float16(forward, getattr(np, op)(ALL_FLOAT16, b))
        assert_same_float16(backward, getattr(np, op)(b, ALL_FLOAT16))


def test_subnormal_literal():
    x = np.float16([1e-5, -2e-6, 6e-8, 0.0, 1.0, 7e-5])
    device = jax.devices("tt")[0]
    actual = jax.jit(lambda v: (v + jnp.float16(3e-5), v - jnp.float16(6e-8)))(jax.device_put(x, device))
    assert_same_float16(actual[0], x + np.float16(3e-5))
    assert_same_float16(actual[1], x - np.float16(6e-8))


def test_isclose_of_tiny_values():
    """jax's lax_numpy_operators_test testOp isclose_float16_int16: tiny values are not close to zero."""
    a = np.float16([1.1e-5, 1.4e-5, 2e-6, 6e-8])
    b = np.zeros(4, np.int16)
    device = jax.devices("tt")[0]
    actual = jax.jit(jnp.isclose)(jax.device_put(a, device), jax.device_put(b, device))
    np.testing.assert_array_equal(np.asarray(actual), np.isclose(a, b))
