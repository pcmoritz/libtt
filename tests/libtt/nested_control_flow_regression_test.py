"""Control flow nested inside a conditional lowers like top-level control flow.

Canonicalization hoists constants out of lax.cond branches, and outlining a
branch into its own program turned those constants into program inputs: a
counted loop in the branch lost its constant bounds (which a trace capture
cannot read from the device), and a cumsum's reduce_window two conditionals
deep no longer legalized without a constant initial value. A conditional three
levels deep also failed to legalize, because dialect conversion does not
reapply the case pattern to the nested case it clones into a branch program.
And the layout pass kept a nested case's selector on host in the branch
program that reads it but not in its sibling, and a case whose branch programs
disagree gets no layout conversions on its inputs at all; a selector argument
of an outlined program now stays on device like its siblings' and the runtime
reads it from there. SGLang-JAX's sampler hits all of these at temperature > 0.
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


def test_nested_conditional():
    # cumsum lowers to a reduce_window, whose initial value has to stay a
    # constant two conditionals deep.
    def run(outer, inner, x):
        return jax.lax.cond(
            outer,
            lambda v: v + 1,
            lambda v: jax.lax.cond(inner, lambda w: w * 2, lambda w: jnp.cumsum(w, axis=-1), v),
            x,
        )

    run = _jit(run, trace=False)
    x = _put(SOURCE)
    for outer, inner in [(False, False), (False, True), (True, False), (True, True)] * 2:
        expected = SOURCE + 1 if outer else (SOURCE * 2 if inner else np.cumsum(SOURCE, axis=-1))
        result = run(_put(outer), _put(inner), x)
        np.testing.assert_array_equal(np.asarray(result), expected)


def test_doubly_nested_conditional():
    def run(outer, first, second, x):
        innermost = lambda v: jax.lax.cond(second, lambda w: w * 2, lambda w: w - 1, v)
        middle = lambda v: jax.lax.cond(first, lambda w: w + 3, innermost, v)
        return jax.lax.cond(outer, lambda v: v + 1, middle, x)

    run = _jit(run, trace=False)
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
