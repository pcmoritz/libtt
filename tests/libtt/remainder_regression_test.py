"""StableHLO remainder truncates: the result takes the sign of the dividend.

tt-mlir lowers it to TTNN's fmod; TTNN's remainder floors instead, which
differs for operands of opposite signs (-5 rem 3 is -2, not 1). Unary ops
around a remainder are not fused into it as activations, so they cannot
switch it to the flooring kernel.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax


def operands(dtype, seed=0):
    rng = np.random.default_rng(seed)
    # Small integers keep every result exact; divisors of both signs.
    a = rng.integers(-20, 21, (64, 128))
    b = rng.integers(1, 8, (64, 128)) * rng.choice([-1, 1], (64, 128))
    return a.astype(dtype), b.astype(dtype)


@pytest.mark.parametrize("dtype", [np.float32, jnp.bfloat16, np.int32])
def test_remainder_truncates(dtype):
    a, b = operands(dtype)
    out = jax.jit(lax.rem)(*(jax.device_put(x, jax.devices("tt")[0]) for x in (a, b)))
    expected = np.fmod(a.astype(np.float64), b.astype(np.float64))
    np.testing.assert_array_equal(np.asarray(out).astype(np.float64), expected)


@pytest.mark.parametrize("dtype", [np.float32, jnp.bfloat16])
def test_remainder_between_unary_ops(dtype):
    # neg on an input and abs on the output are fusable unary ops.
    a, b = operands(dtype, seed=1)
    run = jax.jit(lambda x, y: jnp.abs(lax.rem(-x, y)), compiler_options={"optimization_level": "O1"})
    out = run(*(jax.device_put(x, jax.devices("tt")[0]) for x in (a, b)))
    expected = np.abs(np.fmod(-a.astype(np.float64), b.astype(np.float64)))
    np.testing.assert_array_equal(np.asarray(out).astype(np.float64), expected)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32])
def test_unsigned_remainder(dtype):
    rng = np.random.default_rng(2)
    a = rng.integers(0, 200, (64, 128)).astype(dtype)
    b = rng.integers(1, 9, (64, 128)).astype(dtype)
    out = jax.jit(lax.rem)(*(jax.device_put(x, jax.devices("tt")[0]) for x in (a, b)))
    np.testing.assert_array_equal(np.asarray(out), a % b)
