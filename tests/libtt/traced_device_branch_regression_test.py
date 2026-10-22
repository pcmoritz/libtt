"""A traced conditional can branch on a value the device computed.

The trace keeps both branches and the prefetcher picks one at replay from
the predicate's buffer (docs/trace-branches.md), so the function stays one
trace with no eager conditional in it. Both branches run once when the trace
is captured, so the results of that first call come from the replay.
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
    """Counts the trace ops and the conditionals left in each function of the
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
def test_predicate_computed_from_data(trace, tmp_path):
    def run(x):
        y = jnp.sin(x) * 3 + 1
        z = jax.lax.cond(jnp.sum(y) > 0, lambda v: jnp.cos(v) + 1, lambda v: v * v - 1, y)
        return jnp.exp(z / 8) + 2

    run = _jit(run, trace, tmp_path)
    for scale in (1.0, -1.0, 1.0, 1.0, -1.0, -1.0):
        x = SOURCE * scale
        y = np.sin(x) * 3 + 1
        z = np.cos(y) + 1 if y.sum() > 0 else y * y - 1
        expected = np.exp(z / 8) + 2
        result = run(_put(x))
        np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-2, atol=1e-2)
    if trace:
        traces, cases = _ops_per_function(tmp_path)
        # One trace holds the whole function; the conditional is inside it.
        assert traces["main"] == 1
        assert cases["main"] == 0


@pytest.mark.parametrize("trace", [False, True])
def test_branches_of_unequal_size(trace):
    # The branch with more kernels is padded so that the device is in the
    # same state after either branch.
    def run(x):
        heavy = lambda v: jnp.tanh(jnp.cos(jnp.sin(v) * 2) + v) * 3
        return jax.lax.cond(x[0, 0] > 0, heavy, lambda v: v - 1, x) + 1

    run = _jit(run, trace)
    for sign in (1.0, -1.0, -1.0, 1.0, 1.0):
        x = SOURCE.copy()
        x[0, 0] = sign
        expected = (np.tanh(np.cos(np.sin(x) * 2) + x) * 3 if sign > 0 else x - 1) + 1
        result = run(_put(x))
        np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("trace", [False, True])
def test_device_predicate_with_host_predicate(trace):
    # A host predicate still selects the trace; the device predicate branches
    # inside it. Every combination replays correctly.
    def run(flag, x):
        y = jax.lax.cond(flag, lambda v: v * 2, lambda v: v + 10, x)
        return jax.lax.cond(jnp.sum(y) > 1000, lambda v: v - 1, lambda v: v + 1, y)

    run = _jit(run, trace)
    for flag, scale in [(True, 1.0), (False, 1.0), (True, -1.0), (False, -1.0), (True, 1.0)]:
        x = SOURCE * scale
        y = x * 2 if flag else x + 10
        expected = y - 1 if y.sum() > 1000 else y + 1
        result = run(_put(flag), _put(x))
        np.testing.assert_array_equal(np.asarray(result), expected)
