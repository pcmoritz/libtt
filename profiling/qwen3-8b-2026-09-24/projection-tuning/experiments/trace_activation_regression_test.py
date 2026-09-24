"""Device activations crossing trace boundaries must not become stale."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_trace_activation_buffers_and_lifetimes(dtype):
    device = jax.devices("tt")[0]
    options = {"enable_trace": "true", "optimization_level": "O1"}
    producers = [
        jax.jit(lambda x: x + 1, compiler_options=options),
        jax.jit(lambda x: x - 1, compiler_options=options),
    ]
    consume = jax.jit(lambda x: x * 2, compiler_options=options)
    saved = []
    for step in range(24):
        host = ((np.arange(4096).reshape(32, 128) + step) % 31).astype(dtype)
        value = producers[step % 2](jax.device_put(host, device))
        expected = (host.astype(np.float32) + (1 if step % 2 == 0 else -1)) * 2
        np.testing.assert_array_equal(np.asarray(consume(value)), expected)
        # Keep older outputs alive across captures and cache eviction.
        if step % 4 == 0:
            saved.append((value, expected))
    for value, expected in reversed(saved):
        np.testing.assert_array_equal(np.asarray(consume(value)), expected)


def test_trace_activation_conditional_replay():
    device = jax.devices("tt")[0]
    options = {"enable_trace": "true", "optimization_level": "O1"}
    produce = jax.jit(lambda x: x + 2, compiler_options=options)
    consume = jax.jit(
        lambda p, x: jax.lax.cond(p, lambda v: v + 1, lambda v: v - 1, x),
        compiler_options=options,
    )
    for step in range(12):
        host = np.full((32, 128), step, dtype=jnp.bfloat16)
        value = produce(jax.device_put(host, device))
        predicate = jax.device_put(np.asarray(step % 2 == 0), device)
        expected = host.astype(np.float32) + (3 if step % 2 == 0 else 1)
        np.testing.assert_array_equal(np.asarray(consume(predicate, value)), expected)
