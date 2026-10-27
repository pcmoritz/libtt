"""A traced function keeps its conditionals inside its trace.

A conditional ran eagerly between traces: the runtime read its index on host,
which for an index the device computes waited for all the work queued before
it, and the values crossing the trace boundaries were copied. Now the trace
holds every branch, and the device reads the index when the replay gets there
and runs only the branch it selects (tt-metal's record_case_begin,
trace_case.patch). An index the caller passes from host is staged into a slot
like any other input.
"""

import re
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

SOURCE = (np.arange(64 * 32, dtype=np.float32).reshape(64, 32) % 7) - 3


def _jit(fn, trace=True, export_path=None):
    options = {"optimization_level": "O1", "enable_trace": str(trace).lower()}
    if export_path is not None:
        options["export_path"] = str(export_path)
    return jax.jit(fn, compiler_options=options)


def _put(value):
    return jax.device_put(np.asarray(value), jax.devices("tt")[0])


def _ops_per_function(export_path):
    """Counts the trace ops and the conditionals of each function in the
    exported runtime IR."""
    (ir,) = (export_path / "irs").glob("ttnn_runtime_[0-9]*.mlir")
    traces, cases = {}, {}
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


def data_dependent(x):
    y = x * 2
    y = lax.cond(y.sum() > 0, lambda v: v + 1, lambda v: v * 3, y)
    # A switch whose index the device computes, with branches of different
    # lengths, an identity branch and a value returned twice.
    index = jnp.clip(jnp.round(y.mean()).astype(jnp.int32), -1, 3)
    y, z = lax.switch(
        index,
        [
            lambda v: (v, v),
            lambda v: (jnp.tanh(v) * 4, v - 1),
            lambda v: (lax.cond(v.max() > 20, lambda u: u - 20, lambda u: u + 20, v), v),
        ],
        y,
    )
    return y + 0.5 * z


def data_dependent_reference(x):
    y = x * 2
    y = y + 1 if y.sum() > 0 else y * 3
    index = min(max(int(np.round(y.mean())), 0), 2)
    if index == 0:
        y, z = y, y
    elif index == 1:
        y, z = np.tanh(y) * 4, y - 1
    else:
        y, z = (y - 20 if y.max() > 20 else y + 20), y
    return y + 0.5 * z


@pytest.mark.parametrize("trace", [False, True])
def test_conditionals_on_computed_values(trace):
    run = _jit(data_dependent, trace)
    # Inputs that take every branch, in an order that replays each one after
    # the others ran.
    for offset in (0.0, -10.0, 1.0, 0.1, 8.0, -10.0, 0.0, 8.0, 1.0):
        x = SOURCE + offset
        np.testing.assert_allclose(
            np.asarray(run(_put(x))), data_dependent_reference(x), rtol=2e-2, atol=2e-2, err_msg=str(offset)
        )


def test_counted_loop_of_conditionals():
    def run(x):
        return lax.fori_loop(0, 4, lambda i, v: lax.cond(v.sum() > 0, lambda u: u - 3, lambda u: u + 2, v), x)

    run = _jit(run)
    for offset in (0.0, 2.0, -1.0, 0.0):
        x = SOURCE + offset
        expected = x
        for _ in range(4):
            expected = expected - 3 if expected.sum() > 0 else expected + 2
        np.testing.assert_array_equal(np.asarray(run(_put(x))), expected)


def test_the_function_stays_one_trace(tmp_path):
    run = _jit(data_dependent, export_path=tmp_path)
    np.testing.assert_allclose(np.asarray(run(_put(SOURCE))), data_dependent_reference(SOURCE), rtol=2e-2, atol=2e-2)
    traces, cases = _ops_per_function(tmp_path)
    assert traces["main"] == 1, traces
    # The conditionals moved into the trace function; no branch program has a
    # trace of its own.
    assert cases["main"] == 0, cases
    assert sum(count for name, count in cases.items() if name.startswith("trace_")) == 2, cases
    assert sum(count for name, count in traces.items() if name != "main") == 0, traces


def test_host_runs_ahead_of_the_device():
    # Device work that takes a while, then conditionals on what it computed.
    # Reading their indices on host would hold the call until that work is done.
    def heavy(x, w):
        for _ in range(24):
            x = jnp.tanh(x @ w)
        return x[:64, :32]

    rng = np.random.default_rng(0)
    w = _put((rng.standard_normal((2048, 2048)) * 0.02).astype(np.float32))
    x = _put(rng.standard_normal((2048, 2048)).astype(np.float32))
    heavy = _jit(heavy)
    run = _jit(data_dependent)
    for _ in range(3):
        np.asarray(run(heavy(x, w)))

    start = time.perf_counter()
    result = run(heavy(x, w))
    dispatched = time.perf_counter() - start
    result = np.asarray(result)
    finished = time.perf_counter() - start
    assert dispatched < 0.5 * finished, f"the call returned after {dispatched * 1e3:.1f} of {finished * 1e3:.1f} ms"


@pytest.mark.parametrize("computed", [False, True])
def test_results_passed_through_from_dead_and_live_inputs(computed):
    # A branch that returns its input unchanged, whether nothing reads the
    # input after the case, it is still read, or the caller passed it: the
    # case's result must take the input's value, and the input keep it.
    def run(x, flag):
        pred = (x.sum() > 0) if computed else flag
        y = x * 2
        dead = lax.cond(pred, lambda v: v * 3, lambda v: v, y)
        z = x - 1
        live = lax.cond(pred, lambda v: v + 10, lambda v: v, z)
        passed = lax.cond(pred, lambda v: v * 5, lambda v: v, x)
        return dead + 1, live + z, passed

    run = _jit(run)
    flags = {value: _put(np.bool_(value)) for value in (False, True)}
    calls = []
    for k, take in enumerate([True, False, False, True, True, False]):
        x = SOURCE + (k + 1 if take else -10 - k)
        calls.append((x, run(_put(x), flags[take])))
    # Checked only after the last call: each result must have outlived the
    # replays after it, which rewrite the trace's buffers.
    for x, (dead, live, passed) in calls:
        take = bool(x.sum() > 0)
        np.testing.assert_array_equal(np.asarray(dead), (x * 6 if take else x * 2) + 1)
        np.testing.assert_array_equal(np.asarray(live), (x + 9 if take else x - 1) + x - 1)
        np.testing.assert_array_equal(np.asarray(passed), x * 5 if take else x)


def test_only_the_selected_branch_writes_in_place(tmp_path):
    # A branch that writes into the cache in place. The call that captures the
    # trace prepares every branch first, to compile it; the ones the flag does
    # not select must not write into the cache then. The write before the case
    # keeps the cache in its own buffer on device, as a model's KV cache,
    # rather than staging it into a slot.
    device = jax.devices("tt")[0]
    rng = np.random.default_rng(0)
    pages = np.tile(np.arange(16, dtype=np.int32)[::-1], (16, 1))
    first, second = np.arange(16, dtype=np.int32) + 24, np.arange(16, dtype=np.int32) + 64

    def write(cache, values, positions):
        return jax.ffi.ffi_call(
            "tt.paged_update_cache",
            jax.ShapeDtypeStruct(cache.shape, cache.dtype),
            input_output_aliases={0: 0},
        )(cache, values, _put(positions), _put(pages), share_cache=True)

    def step(cache, values, flag):
        cache = write(cache, values, first)
        return lax.cond(flag, lambda c: write(c, values * 2, second), lambda c: c, cache)

    step = jax.jit(
        step,
        donate_argnums=(0,),
        compiler_options={"optimization_level": "O1", "enable_trace": "true", "export_path": str(tmp_path)},
    )
    flags = {value: _put(np.bool_(value)) for value in (False, True)}
    expected = np.asarray(rng.normal(size=(17, 8, 32, 128)), dtype=jnp.bfloat16)
    cache = _put(expected.copy())
    for take in [False, True, False, True]:
        values = np.asarray(rng.normal(size=(1, 16, 8, 128)), dtype=jnp.bfloat16)
        cache = step(cache, _put(values), flags[take])
        for positions, scale in [(first, 1)] + [(second, 2)] * take:
            for row, index in enumerate(positions):
                expected[pages[row, index // 32], :, index % 32, :] = values[0, row] * scale
        np.testing.assert_array_equal(np.asarray(cache), expected, err_msg=str(take))
    traces, cases = _ops_per_function(tmp_path)
    assert traces["main"] == 1 and cases["main"] == 0, (traces, cases)


def test_constants_a_branch_computes(tmp_path):
    # A branch computes a mask from constants on device. The warmup before
    # the capture only prepares the branch's workloads, but the constants it
    # loads are computed for real, on the call that captures the trace.
    def run(x, flag):
        def masked(v):
            columns = jnp.arange(v.shape[-1], dtype=jnp.int32)
            return jnp.where((columns % 3 == 0) | (columns > 27), v, -v)

        return lax.cond(flag, masked, lambda v: v + 1, x)

    run = _jit(run, export_path=tmp_path)
    columns = np.arange(SOURCE.shape[-1])
    for take in [True, False, True]:
        expected = np.where((columns % 3 == 0) | (columns > 27), SOURCE, -SOURCE) if take else SOURCE + 1
        np.testing.assert_array_equal(np.asarray(run(_put(SOURCE), _put(np.bool_(take)))), expected, err_msg=str(take))
    traces, cases = _ops_per_function(tmp_path)
    assert traces["main"] == 1 and cases["main"] == 0, (traces, cases)


def test_a_branch_returning_a_cached_constant(tmp_path):
    # Branch 0 returns a constant the program caches. The other branch copies
    # its result into the case's result, so that must not be the constant's
    # buffer: a later call taking branch 0 runs no work that would restore it.
    def run(x, flag):
        return lax.cond(flag, lambda v: v * 2, lambda v: jnp.full_like(v, 7.0), x)

    run = _jit(run, export_path=tmp_path)
    for take in [False, True, False, True, False]:
        expected = SOURCE * 2 if take else np.full_like(SOURCE, 7.0)
        np.testing.assert_array_equal(np.asarray(run(_put(SOURCE), _put(np.bool_(take)))), expected, err_msg=str(take))
    traces, cases = _ops_per_function(tmp_path)
    assert traces["main"] == 1 and cases["main"] == 0, (traces, cases)
