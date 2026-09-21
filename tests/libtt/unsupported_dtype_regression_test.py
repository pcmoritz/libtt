"""Unsupported dtypes must raise an error instead of crashing the compiler."""

import jax
import jax.numpy as jnp
import pytest


@pytest.mark.parametrize("dtype", [jnp.int4, jnp.uint4])
def test_unsupported_int4_comparison(dtype):
    # A boolean result allows int4 inputs to reach layout conversion.
    arg = jax.ShapeDtypeStruct((3,), dtype)
    with pytest.raises(jax.errors.JaxRuntimeError):
        jax.jit(jax.lax.ne).lower(arg, arg).compile()
