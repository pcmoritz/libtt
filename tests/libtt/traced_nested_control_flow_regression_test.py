"""Trace conditionals whose branches hold more control flow.

A trace capture cannot read device values. A counted loop inside a branch has
to keep its constant bounds, and a predicate forwarded to a nested conditional
has to select the replayed trace like a top-level predicate does. SGLang-JAX's
traced sampler needs both: its non-greedy branch masks top-k and top-p with
fori_loop searches and applies min-p under a nested cond.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

SOURCE = (np.arange(64 * 32, dtype=np.float32).reshape(64, 32) % 7) - 3


def _jit(fn, trace):
    return jax.jit(
        fn, compiler_options={"optimization_level": "O1", "enable_trace": str(trace).lower()}
    )


def _put(value):
    return jax.device_put(np.asarray(value), jax.devices("tt")[0])


@pytest.mark.parametrize("trace", [False, True])
def test_counted_loop_inside_conditional(trace):
    def run(predicate, x):
        return jax.lax.cond(
            predicate,
            lambda v: v + 1,
            lambda v: jax.lax.fori_loop(0, 5, lambda i, y: y * 2 + i, v),
            x,
        )

    run = _jit(run, trace)
    looped = SOURCE
    for i in range(5):
        looped = looped * 2 + i
    x = _put(SOURCE)
    for predicate in (False, True) * 3:
        result = run(_put(predicate), x)
        np.testing.assert_array_equal(np.asarray(result), SOURCE + 1 if predicate else looped)


@pytest.mark.parametrize("trace", [False, True])
def test_nested_conditional(trace):
    # cumsum lowers to a reduce_window, whose initial value has to stay a
    # constant two conditionals deep.
    def run(outer, inner, x):
        return jax.lax.cond(
            outer,
            lambda v: v + 1,
            lambda v: jax.lax.cond(inner, lambda w: w * 2, lambda w: jnp.cumsum(w, axis=-1), v),
            x,
        )

    run = _jit(run, trace)
    x = _put(SOURCE)
    # Interleaved so that replaying the trace of a different inner predicate
    # produces the wrong answer.
    for outer, inner in [(False, False), (False, True), (True, False), (False, True),
                         (True, True), (False, False)] * 2:
        expected = SOURCE + 1 if outer else (SOURCE * 2 if inner else np.cumsum(SOURCE, axis=-1))
        result = run(_put(outer), _put(inner), x)
        np.testing.assert_array_equal(np.asarray(result), expected)


@pytest.mark.parametrize("trace", [False, True])
def test_doubly_nested_conditional(trace):
    # Each predicate is forwarded unchanged to the conditional it selects, two
    # levels deep for the innermost one.
    def run(outer, first, second, x):
        innermost = lambda v: jax.lax.cond(second, lambda w: w * 2, lambda w: w - 1, v)
        middle = lambda v: jax.lax.cond(first, lambda w: w + 3, innermost, v)
        return jax.lax.cond(outer, lambda v: v + 1, middle, x)

    run = _jit(run, trace)
    x = _put(SOURCE)
    for outer, first, second in [(False, False, True), (False, False, False), (True, False, True),
                                 (False, True, False), (False, False, True)] * 2:
        result = run(_put(outer), _put(first), _put(second), x)
        if outer:
            expected = SOURCE + 1
        elif first:
            expected = SOURCE + 3
        else:
            expected = SOURCE * 2 if second else SOURCE - 1
        np.testing.assert_array_equal(np.asarray(result), expected)
