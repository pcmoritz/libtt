"""While-loop bodies are captured into a trace and replayed per iteration.

Numerical checks alone pass even if every iteration silently falls back to
op-by-op execution, so the tests also check the runtime's fallback warning.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

FALLBACK = "Running while loop body op by op"


def device_put(x):
    return jax.device_put(x, jax.devices("tt")[0])


@pytest.mark.parametrize("iterations", [3, 50])
def test_counted_loop_replays_trace(iterations, capfd):
    run = jax.jit(
        lambda x: jax.lax.fori_loop(0, iterations, lambda i, x: x * 0.5 + 1.0, x)
    )
    x = np.random.default_rng(0).uniform(-4, 4, (32, 64)).astype(np.float32)
    expected = x.copy()
    for _ in range(iterations):
        expected = expected * 0.5 + 1.0
    for _ in range(2):
        np.testing.assert_allclose(
            np.asarray(run(device_put(x))), expected, rtol=1e-6, atol=1e-6
        )
    assert FALLBACK not in "".join(capfd.readouterr())


def test_condition_loop_short_and_long_invocations(capfd):
    # The trip count is only known at run time; short invocations run op by op
    # and long ones are traced, in any order.
    def loop(x, n):
        return jax.lax.while_loop(
            lambda s: s[0] < n, lambda s: (s[0] + 1, s[1] * 0.5 + 1.0), (0, x)
        )[1]

    run = jax.jit(loop)
    x = np.random.default_rng(1).uniform(-4, 4, (32, 64)).astype(np.float32)
    for n in [2, 20, 3, 30, 9, 1]:
        expected = x.copy()
        for _ in range(n):
            expected = expected * 0.5 + 1.0
        np.testing.assert_allclose(
            np.asarray(run(device_put(x), n)), expected, rtol=1e-6, atol=1e-6
        )
    assert FALLBACK not in "".join(capfd.readouterr())


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

