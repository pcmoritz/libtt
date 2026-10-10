"""While-loop bodies are captured into a trace and replayed per iteration.

Numerical checks alone pass even if every iteration silently falls back to
op-by-op execution, so the tests read the runtime's per-loop iteration and
replay counts (TT_RUNTIME_LOG_LOOP_TRACES=1).
"""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

FALLBACK = "Running while loop body op by op"
LOOP_TRACE = re.compile(r"While loop trace \((\w+) loop\): (\d+) iterations, (\d+) replays")


def device_put(x):
    return jax.device_put(x, jax.devices("tt")[0])


@pytest.fixture
def loop_traces(capfd, monkeypatch):
    """Returns a function that yields (kind, iterations, replays) per loop run."""
    monkeypatch.setenv("TT_RUNTIME_LOG_LOOP_TRACES", "1")
    capfd.readouterr()

    def read():
        output = "".join(capfd.readouterr())
        assert FALLBACK not in output
        return [(k, int(n), int(r)) for k, n, r in LOOP_TRACE.findall(output)]

    return read


# A counted loop captures its body during the second iteration when at least
# two replays follow, so every later iteration is a replay.
@pytest.mark.parametrize("iterations,replays", [(3, 0), (4, 2), (50, 48)])
def test_counted_loop_replays_trace(iterations, replays, loop_traces):
    # x counts the iterations exactly, so a missed replay changes the result.
    run = jax.jit(lambda x: jax.lax.fori_loop(0, iterations, lambda i, x: x + 1.0, x))
    x = np.random.default_rng(0).integers(-4, 4, (32, 64)).astype(np.float32)
    for _ in range(2):
        np.testing.assert_array_equal(np.asarray(run(device_put(x))), x + iterations)
    assert loop_traces() == [("counted", iterations, replays)] * 2


def test_condition_loop_short_and_long_invocations(loop_traces):
    # The predicate depends on the evolving tensor, so the loop keeps its
    # condition program. Its trip count is only known at run time: the body is
    # captured during the fourth iteration, and short invocations never are.
    def loop(x, n):
        return jax.lax.while_loop(lambda x: jnp.any(x < n), lambda x: x + 1.0, x)

    run = jax.jit(loop)
    x = np.zeros((32, 64), np.float32)
    x[0, 0] = -1.0
    counts = [2, 20, 3, 30, 9, 1, 4, 5]
    for n in counts:
        got = np.asarray(run(device_put(x), np.float32(n)))
        np.testing.assert_array_equal(got, x + n + 1)
    assert loop_traces() == [("condition", n + 1, max(0, n - 3)) for n in counts]


@pytest.mark.parametrize("third_dtype", [np.float32, np.float16])
def test_swapping_body_with_another_state_value(third_dtype):
    # The body swaps two state values, which alias each other's slots, and
    # updates a third. If writing the third value into its slot fails after the
    # swap has been written, the loop must still continue from the swapped
    # values.
    iterations = 11
    run = jax.jit(
        lambda a, b, c: jax.lax.fori_loop(
            0, iterations, lambda i, s: (s[1], s[0], s[2] + 1), (a, b, c)
        )
    )
    rng = np.random.default_rng(2)
    a = rng.uniform(-4, 4, (32, 64)).astype(np.float32)
    b = rng.uniform(-4, 4, (32, 64)).astype(np.float32)
    c = np.zeros((32, 64), third_dtype)
    got_a, got_b, got_c = run(device_put(a), device_put(b), device_put(c))
    np.testing.assert_array_equal(np.asarray(got_a), b)
    np.testing.assert_array_equal(np.asarray(got_b), a)
    np.testing.assert_array_equal(
        np.asarray(got_c), np.full((32, 64), iterations, third_dtype)
    )

