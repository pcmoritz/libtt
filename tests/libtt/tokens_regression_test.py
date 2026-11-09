"""Tokens reach libtt as the empty boolean arrays that JAX passes and returns."""

import jax
import jax.numpy as jnp
import numpy as np
from jax._src import core
from jax.experimental.overlap import program_order


def test_returned_token():
    token = jax.jit(jax.lax.create_token)()
    assert isinstance(token, core.Token)
    token.block_until_ready()


def test_token_argument_passes_through():
    token = jax.lax.create_token()
    out, out_token = jax.jit(lambda x, token: (x * 2, token))(jnp.arange(10.0), token)
    np.testing.assert_array_equal(out, np.arange(10.0) * 2)
    assert isinstance(out_token, core.Token)
    assert out_token is not token


# program_order puts a new token through an optimization barrier with the values.
def test_token_in_optimization_barrier():
    c = jnp.arange(8.0)

    @jax.jit
    @program_order(enforce=True)
    def f(x):
        @program_order(enforce=False)
        def add(x):
            return x + c

        @program_order(enforce=False)
        def multiply(y):
            return y * c

        return multiply(add(x))

    x = np.arange(8.0, dtype=np.float32)
    np.testing.assert_array_equal(f(x), (x + x) * x)
