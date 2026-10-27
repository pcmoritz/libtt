"""A conditional on a value the device computed runs without the host reading it.

With tracing, each branch of a two-way conditional whose selector is on device
is captured as a conditional trace and replayed with enqueue_trace_if: the
device's command prefetcher reads the predicate and runs only the branch it
selects, while the host goes on. The branches compile in a dry warmup that runs
none of their kernels, and publish their results in the same slots.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

SOURCE = (np.arange(64 * 32, dtype=np.float32).reshape(64, 32) % 7) - 3


def _jit(fn):
    return jax.jit(fn, compiler_options={"optimization_level": "O1", "enable_trace": "true"})


def _put(value):
    return jax.device_put(np.asarray(value), jax.devices("tt")[0])


def _branch_on_sum(x):
    y = jnp.sin(x) * 3 + x
    z = jax.lax.cond(jnp.sum(y) > 0, lambda v: jnp.cos(v) + 1, lambda v: v * v - 1, y)
    return jnp.exp(z / 8) + 2


def _expected(x):
    y = np.sin(x) * 3 + x
    z = np.cos(y) + 1 if y.sum() > 0 else y * y - 1
    return np.exp(z / 8) + 2


def test_runs_the_branch_the_device_selects():
    run = _jit(_branch_on_sum)
    # The offsets put the sum on both sides of zero, so both branches run, in
    # either order after the first call captured both.
    for offset in (1.0, -1.0, -1.0, 1.0, 1.0, -1.0):
        x = SOURCE + offset
        np.testing.assert_allclose(np.asarray(run(_put(x))), _expected(x), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("taken", [True, False])
def test_a_branch_never_taken_does_not_run(taken):
    # Only one branch is ever selected; the other is captured but never run.
    def run(x):
        return jax.lax.cond(jnp.sum(x) > -1e9 if taken else jnp.sum(x) < -1e9,
                            lambda v: v + 7, lambda v: v * 0 - 7, x)

    run = _jit(run)
    for _ in range(3):
        np.testing.assert_allclose(np.asarray(run(_put(SOURCE))), SOURCE + 7 if taken else SOURCE * 0 - 7)


def test_the_call_does_not_wait_for_the_device():
    run = _jit(_branch_on_sum)
    busy = jax.jit(lambda m: jnp.linalg.matrix_power(m, 24) * 1e-6)
    matrix = _put(np.eye(2048, dtype=np.float32))
    x = _put(SOURCE + 1.0)
    np.asarray(run(x))
    np.asarray(run(_put(SOURCE - 1.0)))
    np.asarray(busy(matrix))

    start = time.perf_counter()
    np.asarray(busy(matrix))
    busy_seconds = time.perf_counter() - start

    returns = []
    for _ in range(3):
        pending = busy(matrix)
        start = time.perf_counter()
        result = run(x)
        returns.append(time.perf_counter() - start)
        np.asarray(result)
        np.asarray(pending)
    # Reading the predicate on host would wait for the busy work ahead of it.
    assert min(returns) < 0.25 * busy_seconds, (returns, busy_seconds)
