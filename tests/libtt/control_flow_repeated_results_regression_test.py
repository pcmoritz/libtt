"""Control flow that returns one tensor for two results keeps it for both.

A branch or loop body whose results fold to one value returns the same tensor
twice: a one-step scan's stacked output is its carry, and a body that computes
the same expression for two carries shares it after CSE. The caller compiled
each result as its own value and deallocated the first after its last use,
which freed the buffer the second one still named ("Tensor is not allocated").
Eager conditionals and loops failed this way on every build, one-step scans
since loop bodies stopped retaining their outputs (#404); it is why
jax.experimental.rnn.lstm_ref failed for one-step sequences. With tracing, a
counted loop runs inside the capture and fails the same way.
"""

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import pytest

SOURCE = np.full((32, 32), 0.5, np.float32)


def _jit(fn, trace):
    options = {"optimization_level": "O1", "enable_trace": str(trace).lower()}
    return jax.jit(fn, compiler_options=options)


def _put(value):
    return jax.device_put(np.asarray(value), jax.devices("tt")[0])


@pytest.mark.parametrize("trace", [False, True])
def test_one_step_scan_returning_its_carry(trace):
    # No scanned input: the per-step slice of one cannot be traced yet.
    def run(h):
        def body(h, _):
            h = jnp.tanh(h + 1)
            return h, (h, h)

        _, (hs, ys) = lax.scan(body, h, None, length=1)
        return jnp.sum(ys) + jnp.sum(hs[0])

    run = _jit(run, trace)
    h = _put(SOURCE)
    for _ in range(3):
        np.testing.assert_allclose(np.asarray(run(h)), 2 * np.sum(np.tanh(SOURCE + 1)), rtol=1e-5)


@pytest.mark.parametrize("trace", [False, True])
def test_cond_returning_one_value_twice(trace):
    def run(predicate, x):
        a, b = lax.cond(predicate, lambda v: (jnp.tanh(v), jnp.tanh(v)), lambda v: (v * 2, v * 2), x)
        return jnp.sum(a * 3) + jnp.sum(b.reshape(-1))

    run = _jit(run, trace)
    x = _put(SOURCE)
    for predicate in (False, True) * 3:
        value = np.tanh(SOURCE) if predicate else SOURCE * 2
        result = run(_put(predicate), x)
        np.testing.assert_allclose(np.asarray(result), 4 * np.sum(value), rtol=1e-5)


@pytest.mark.parametrize("trace", [False, True])
def test_loop_returning_one_value_for_two_carries(trace):
    def run(x):
        a, b = lax.fori_loop(0, 3, lambda i, ab: (ab[0] * 2, ab[0] * 2), (x, x))
        return jnp.sum(a) + jnp.sum(b.reshape(-1))

    run = _jit(run, trace)
    x = _put(SOURCE)
    for _ in range(3):
        np.testing.assert_allclose(np.asarray(run(x)), 2 * np.sum(SOURCE * 8), rtol=1e-5)
