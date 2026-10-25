"""A loop whose body returns a cached constant leaves the constant cached.

The runtime retains a loop's state while the body runs and afterwards restored
the retain flags by state slot. By then a slot can hold a different tensor: a
body that returns a constant hands back the cached copy, which got the flag of
the value the loop was entered with. The caller deallocated it, and the next
program to load that constant read the freed tensor. The cache is shared by all
programs with the same constant, so the second case here failed on its first
call, after the first case freed it.
"""

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("length", [1, 3])
def test_loop_returning_a_constant(length):
    def run(x):
        def body(carry, _):
            constant, total = carry
            return (jnp.full_like(constant, 2.0), total + constant), None

        (constant, total), _ = lax.scan(body, (x, x), None, length=length)
        return constant * 3.0, total

    x = np.ones((32, 32), np.float32)
    expected_total = x + 2.0 * (length - 1) + x
    fn = jax.jit(run)
    for call in range(3):
        constant, total = fn(x)
        np.testing.assert_array_equal(constant, np.full_like(x, 6.0))
        np.testing.assert_array_equal(total, expected_total)
