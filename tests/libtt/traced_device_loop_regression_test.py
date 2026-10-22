"""A traced loop can run for a trip count the device decides.

The trace captures the predicate computation and the body once and loops on
the predicate at replay (docs/trace-branches.md), so a while_loop with a
device condition, or a fori_loop whose bound is an argument, stays inside one
trace. Counted loops with constant bounds are unrolled at capture as before.
"""

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
    (ir,) = (export_path / "irs").glob("ttnn_runtime_*.mlir")
    traces = {}
    loops = {}
    name = None
    for line in ir.read_text().splitlines():
        match = re.search(r"func\.func (?:private |public )?@([\w$.]+)\(", line)
        if match:
            name = match.group(1)
            traces.setdefault(name, 0)
            loops.setdefault(name, 0)
        elif "ttnn.capture_or_execute_trace" in line and name:
            traces[name] += 1
        elif "ttcore.while_loop" in line and name:
            loops[name] += 1
    return traces, loops


@pytest.mark.parametrize("trace", [False, True])
def test_while_loop_with_device_condition(trace, tmp_path):
    def run(limit, x):
        def body(state):
            value, step = state
            return value * 2 - 1, step + 1

        value, _ = jax.lax.while_loop(lambda state: state[1] < limit, body, (x, jnp.int32(0)))
        return value + 1

    run = _jit(run, trace, tmp_path)
    x = _put(SOURCE)
    for limit in (3, 0, 2, 5, 1, 3):
        expected = SOURCE
        for _ in range(limit):
            expected = expected * 2 - 1
        result = run(_put(np.int32(limit)), x)
        np.testing.assert_array_equal(np.asarray(result), expected + 1)
    if trace:
        traces, loops = _ops_per_function(tmp_path)
        assert traces["main"] == 1
        assert loops["main"] == 0


@pytest.mark.parametrize("trace", [False, True])
def test_fori_loop_with_argument_bound(trace):
    def run(count, x):
        return jax.lax.fori_loop(0, count, lambda i, v: v + i, x)

    run = _jit(run, trace)
    x = _put(SOURCE)
    for count in (4, 0, 1, 9, 2):
        expected = SOURCE
        for i in range(count):
            expected = expected + i
        result = run(_put(np.int32(count)), x)
        np.testing.assert_array_equal(np.asarray(result), expected)


@pytest.mark.parametrize("trace", [False, True])
def test_loop_then_branch(trace, tmp_path):
    # A loop and a conditional, both controlled by device values, in one trace.
    def run(limit, x):
        def body(state):
            value, step = state
            return value + 3, step + 1

        value, _ = jax.lax.while_loop(lambda state: state[1] < limit, body, (x, jnp.int32(0)))
        return jax.lax.cond(jnp.sum(value) > 0, lambda v: v * 2, lambda v: v - 1, value)

    run = _jit(run, trace, tmp_path)
    for limit, scale in [(1, 1.0), (0, -1.0), (2, 1.0), (0, 1.0), (3, -1.0)]:
        x = SOURCE * scale
        value = x + 3 * limit
        expected = value * 2 if value.sum() > 0 else value - 1
        result = run(_put(np.int32(limit)), _put(x))
        np.testing.assert_array_equal(np.asarray(result), expected)
    if trace:
        traces, loops = _ops_per_function(tmp_path)
        assert traces["main"] == 1
