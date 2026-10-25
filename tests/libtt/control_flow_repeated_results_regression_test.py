"""Control flow that returns one tensor for two results keeps it for both.

A branch or loop body whose results fold to one value returns the same tensor
twice: a one-step scan's stacked output is its carry, and a body that computes
the same expression for two carries shares it after CSE. The caller compiled
each result as its own value and deallocated the first after its last use,
which freed the buffer the second one still named ("Tensor is not allocated").
Conditionals and multi-step loops failed this way on every build, one-step
scans since loop bodies stopped retaining their outputs (#404); it is why
jax.experimental.rnn.lstm_ref failed for one-step sequences.
"""

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import pytest


def _check(fn, args, expected):
    for _ in range(3):
        np.testing.assert_allclose(np.asarray(fn(*args)), expected, rtol=1e-5)


def test_one_step_scan_returning_its_carry():
    def run(h, xs):
        def body(h, x):
            h = jnp.tanh(h + x)
            return h, (h, h)

        _, (hs, ys) = lax.scan(body, h, xs)
        return jnp.sum(ys) + jnp.sum(hs[0])

    h = np.ones((1, 1), np.float32)
    xs = np.ones((1, 1, 1), np.float32)
    _check(jax.jit(run), (h, xs), 2 * np.tanh(2.0))


@pytest.mark.parametrize("predicate", [0, 1])
def test_cond_returning_one_value_twice(predicate):
    def run(p, x):
        a, b = lax.cond(p > 0, lambda x: (jnp.tanh(x), jnp.tanh(x)), lambda x: (x * 2, x * 2), x)
        return jnp.sum(a * 3) + jnp.sum(b.reshape(-1))

    x = np.full((32, 32), 0.5, np.float32)
    value = np.tanh(x) if predicate else x * 2
    _check(jax.jit(run), (np.int32(predicate), x), np.sum(value * 3) + np.sum(value))


def test_loop_returning_one_value_for_two_carries():
    def run(x):
        a, b = lax.fori_loop(0, 3, lambda i, ab: (ab[0] * 2, ab[0] * 2), (x, x))
        return jnp.sum(a) + jnp.sum(b.reshape(-1))

    x = np.full((32, 32), 0.5, np.float32)
    _check(jax.jit(run), (x,), 2 * np.sum(x * 8))
