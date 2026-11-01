"""Feed one traced executable's device output into another traced executable.

The consumer's trace copies the input into its slot on device. The producer's
output is itself a trace output slot that every replay overwrites in place, so
the consumer must pick up the new values on every call rather than reuse the
first one.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("shape", [(1, 32), (1, 4748), (32, 96)])
def test_traced_output_feeds_traced_input(shape):
    options = {"optimization_level": "O1", "enable_trace": "true"}
    produce = jax.jit(lambda x, y: x * 2 + y, compiler_options=options)
    consume = jax.jit(
        lambda logits: jnp.argmax(logits, axis=-1),
        compiler_options=options,
    )
    device = jax.devices("tt")[0]
    rng = np.random.default_rng(3)
    for _ in range(6):
        x = rng.integers(-8, 8, shape).astype(jnp.bfloat16)
        y = rng.integers(-8, 8, shape).astype(jnp.bfloat16)
        logits = produce(jax.device_put(x, device), jax.device_put(y, device))
        index = consume(logits)
        expected = x.astype(np.float32) * 2 + y.astype(np.float32)
        np.testing.assert_array_equal(np.asarray(logits, dtype=np.float32), expected)
        np.testing.assert_array_equal(
            np.take_along_axis(expected, np.asarray(index)[:, None], axis=-1)[:, 0],
            expected.max(axis=-1),
        )


def _ternary(rng, shape):
    """Values in {-1, 0, 1}, whose bf16 products and sums here are exact."""
    return rng.integers(-1, 2, shape).astype(jnp.bfloat16)


def _f32(array):
    return np.asarray(array, dtype=np.float32)


def test_traced_inputs_in_other_buffers():
    """A traced program reads an input that is on device in place, captured by
    address. When the input later comes in another buffer, the runtime copies
    it into a slot of its own before every replay, and the first buffer may then
    be freed and its memory reused."""
    options = {"optimization_level": "O1", "enable_trace": "true"}
    step = jax.jit(lambda w, x: x @ w + 1, compiler_options=options)
    device = jax.devices("tt")[0]
    rng = np.random.default_rng(5)
    weights = [_ternary(rng, (64, 64)) for _ in range(3)]
    first, second = (jax.device_put(w, device) for w in weights[:2])

    def check(w_buffer, w):
        x = _ternary(rng, (32, 64))
        out = step(w_buffer, jax.device_put(x, device))
        np.testing.assert_array_equal(_f32(out), _f32(x) @ _f32(w) + 1)

    for _ in range(3):
        check(first, weights[0])
    check(second, weights[1])
    check(first, weights[0])
    check(second, weights[1])
    first.delete()
    third = jax.device_put(weights[2], device)
    check(second, weights[1])
    check(third, weights[2])
    check(second, weights[1])


def test_traced_donated_updates():
    """A traced program that updates a donated input gets a new buffer every
    call, its previous result."""
    options = {"optimization_level": "O1", "enable_trace": "true"}
    update = jax.jit(lambda cache, x: cache + x, donate_argnums=0, compiler_options=options)
    device = jax.devices("tt")[0]
    rng = np.random.default_rng(6)
    cache = jax.device_put(np.zeros((32, 64), jnp.bfloat16), device)
    total = np.zeros((32, 64), np.float32)
    for _ in range(8):
        x = _ternary(rng, (32, 64))
        cache = update(cache, jax.device_put(x, device))
        total += _f32(x)
        np.testing.assert_array_equal(_f32(cache), total)
