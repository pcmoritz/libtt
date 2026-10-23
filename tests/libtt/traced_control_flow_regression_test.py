"""Tracing never fails on control flow; it traces around what it cannot replay.

A trace holds straight-line device work and counted loops of it with constant
bounds. Every conditional, and every other loop, runs eagerly, and the trace
hoist traces around it instead of failing: the ops before and after it in the
same function, and the programs of its own branches or body, which are hoisted
in turn. Every
program here must therefore run correctly with tracing enabled, and the
exported IR shows where the traces went.
"""

import itertools
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

SOURCE = (np.arange(64 * 32, dtype=np.float32).reshape(64, 32) % 7) - 3


def _jit(fn, trace, export_path=None):
    options = {"optimization_level": "O1", "enable_trace": str(trace).lower()}
    if export_path is not None:
        options["export_path"] = str(export_path)
    return jax.jit(fn, compiler_options=options)


def _put(value):
    return jax.device_put(np.asarray(value), jax.devices("tt")[0])


def _ops_per_function(export_path):
    """Counts the trace ops and the eager conditionals of each function in the
    exported runtime IR."""
    (ir,) = (export_path / "irs").glob("ttnn_runtime_*.mlir")
    traces = {}
    cases = {}
    name = None
    for line in ir.read_text().splitlines():
        match = re.search(r"func\.func (?:private |public )?@([\w$.]+)\(", line)
        if match:
            name = match.group(1)
            traces.setdefault(name, 0)
            cases.setdefault(name, 0)
        elif "ttnn.capture_or_execute_trace" in line and name:
            traces[name] += 1
        elif "ttcore.case" in line and name:
            cases[name] += 1
    return traces, cases


@pytest.mark.parametrize("trace", [False, True])
def test_counted_loop_inside_conditional(trace, tmp_path):
    # The conditional runs eagerly; the branch with the counted loop is one
    # trace, since a loop with constant bounds replays from a capture.
    def run(predicate, x):
        return jax.lax.cond(
            predicate,
            lambda v: v + 1,
            lambda v: jax.lax.fori_loop(0, 5, lambda i, y: y * 2 + i, v),
            x,
        )

    run = _jit(run, trace, tmp_path)
    looped = SOURCE
    for i in range(5):
        looped = looped * 2 + i
    x = _put(SOURCE)
    for predicate in (False, True) * 3:
        result = run(_put(predicate), x)
        np.testing.assert_array_equal(np.asarray(result), SOURCE + 1 if predicate else looped)
    if trace:
        traces, cases = _ops_per_function(tmp_path)
        assert cases["main"] == 1
        # Each branch program is one trace; the loop body runs inside the
        # capture of its branch and has no trace of its own.
        assert traces["main_case_branch_0"] == 1
        assert traces["main_case_branch_1"] == 1
        assert traces["main_case_branch_0_counted_loop"] == 0


@pytest.mark.parametrize("trace", [False, True])
def test_nested_conditional(trace, tmp_path):
    # The nested conditional reads its predicate inside the branch, so the
    # outer conditional runs eagerly and its branches hold the traces.
    def run(outer, inner, x):
        return jax.lax.cond(
            outer,
            lambda v: v + 1,
            lambda v: jax.lax.cond(inner, lambda w: w * 2, lambda w: jnp.cumsum(w, axis=-1), v),
            x,
        )

    run = _jit(run, trace, tmp_path)
    x = _put(SOURCE)
    for outer, inner in [(False, False), (False, True), (True, False), (False, True),
                         (True, True), (False, False)] * 2:
        expected = SOURCE + 1 if outer else (SOURCE * 2 if inner else np.cumsum(SOURCE, axis=-1))
        result = run(_put(outer), _put(inner), x)
        np.testing.assert_array_equal(np.asarray(result), expected)
    if trace:
        traces, cases = _ops_per_function(tmp_path)
        assert cases["main"] == 1
        assert sum(count for name, count in traces.items() if name != "main") >= 2


@pytest.mark.parametrize("trace", [False, True])
def test_doubly_nested_conditional(trace):
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


@pytest.mark.parametrize("trace", [False, True])
def test_conditional_inside_counted_loop(trace):
    def run(predicate, x):
        body = lambda i, v: jax.lax.cond(predicate, lambda w: w * 2, lambda w: w - 1, v)
        return jax.lax.fori_loop(0, 3, body, x)

    run = _jit(run, trace)
    x = _put(SOURCE)
    for predicate in (True, False, False, True) * 2:
        result = run(_put(predicate), x)
        expected = SOURCE * 8 if predicate else SOURCE - 3
        np.testing.assert_array_equal(np.asarray(result), expected)


@pytest.mark.parametrize("trace", [False, True])
def test_conditional_inside_loop_inside_conditional(trace):
    def run(outer, inner, x):
        def branch(v):
            body = lambda i, y: jax.lax.cond(inner, lambda w: w + 1, lambda w: w * 2, y)
            return jax.lax.fori_loop(0, 3, body, v)

        return jax.lax.cond(outer, lambda v: v - 1, branch, x)

    run = _jit(run, trace)
    x = _put(SOURCE)
    for outer, inner in [(False, True), (False, False), (True, True), (False, True)]:
        result = run(_put(outer), _put(inner), x)
        expected = SOURCE - 1 if outer else (SOURCE + 3 if inner else SOURCE * 8)
        np.testing.assert_array_equal(np.asarray(result), expected)


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("trace", [False, True])
def test_nested_predicate_computed_in_a_branch(trace, depth):
    # The predicate at `depth` combines two forwarded flags, so the runtime
    # reads it from device; the conditional that holds it runs eagerly.
    def run(outer, first, second, x):
        computed = lambda v: jax.lax.cond(first ^ second, lambda w: w * 2, lambda w: w - 1, v)
        middle = lambda v: jax.lax.cond(first, lambda w: w + 3, computed, v)
        return jax.lax.cond(outer, lambda v: v + 1, middle if depth == 2 else computed, x)

    run = _jit(run, trace)
    x = _put(SOURCE)
    for outer, first, second in itertools.product([False, True], repeat=3):
        result = run(_put(outer), _put(first), _put(second), x)
        if outer:
            expected = SOURCE + 1
        elif depth == 2 and first:
            expected = SOURCE + 3
        else:
            expected = SOURCE * 2 if first != second else SOURCE - 1
        np.testing.assert_array_equal(np.asarray(result), expected)


@pytest.mark.parametrize("trace", [False, True])
def test_predicate_computed_from_data(trace, tmp_path):
    # A predicate computed on device splits the function into a trace before
    # the conditional, the eager conditional whose branches hold their own
    # traces, and a trace after it.
    def run(x):
        y = jnp.sin(x) * 3 + 1
        z = jax.lax.cond(jnp.sum(y) > 0, lambda v: jnp.cos(v) + 1, lambda v: v * v - 1, y)
        return jnp.exp(z / 8) + 2

    run = _jit(run, trace, tmp_path)
    for scale in (1.0, -1.0, 1.0, 1.0, -1.0):
        x = SOURCE * scale
        y = np.sin(x) * 3 + 1
        z = np.cos(y) + 1 if y.sum() > 0 else y * y - 1
        expected = np.exp(z / 8) + 2
        result = run(_put(x))
        np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-2, atol=1e-2)
    if trace:
        traces, cases = _ops_per_function(tmp_path)
        assert cases["main"] == 1
        assert traces["main"] == 2
        assert sum(count for name, count in traces.items() if name != "main") == 2


@pytest.mark.parametrize("trace", [False, True])
def test_data_dependent_loops(trace):
    # A while loop whose condition the runtime evaluates on device, and a
    # counted loop whose bound is an argument, both run eagerly with traced
    # bodies.
    def run(limit, x):
        def while_body(state):
            value, step = state
            return value * 2 - 1, step + 1

        value, _ = jax.lax.while_loop(lambda state: state[1] < limit, while_body, (x, jnp.int32(0)))
        return jax.lax.fori_loop(0, limit, lambda i, v: v + i, value)

    run = _jit(run, trace)
    x = _put(SOURCE)
    for limit in (3, 0, 2, 3, 1):
        expected = SOURCE
        for _ in range(limit):
            expected = expected * 2 - 1
        for i in range(limit):
            expected = expected + i
        result = run(_put(np.int32(limit)), x)
        np.testing.assert_array_equal(np.asarray(result), expected)


@pytest.mark.parametrize("trace", [False, True])
def test_selector_also_used_as_data(trace):
    # The host predicate selects a branch and is also added to the data. The
    # arithmetic on it runs eagerly; the conditional still replays from a
    # trace per predicate value.
    def run(predicate, x):
        y = jax.lax.cond(predicate, lambda v: v + 1, lambda v: v - 1, x)
        return y + predicate.astype(jnp.float32)

    run = _jit(run, trace)
    x = _put(SOURCE)
    for predicate in (True, False, True, True, False):
        result = run(_put(predicate), x)
        expected = SOURCE + 2 if predicate else SOURCE - 1
        np.testing.assert_array_equal(np.asarray(result), expected)
