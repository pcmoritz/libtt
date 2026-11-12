"""A program frees a donated input once it no longer needs it.

Before, a donated input's memory went only after the program, when its buffer
was deleted, so a program held a donated input and the values computed from it
at once. An optimizer step that updates all of a model's parameters and
optimizer states in one program, as torch_tpu's fused AdamW does, needed twice
their memory: Qwen3-1.7B's update ran out of memory on one Blackhole.
"""

import jax
import jax.numpy as jnp
import numpy as np


def _fourth_power(x):
    # x is last used by the first product and y by the second, so a program
    # that frees donated inputs holds two of x, y and the result at a time,
    # and one that doesn't holds all three.
    y = x * x
    return y * y


def _small_integers():
    # Values whose fourth powers are exact at any multiplier precision.
    return jnp.arange(64 * 128, dtype=jnp.float32).reshape(64, 128) % 4


def test_donated_input_is_freed_after_its_last_use():
    device = jax.devices()[0]
    stats = device.memory_stats()
    free = stats["bytes_limit"] - stats["bytes_in_use"]
    # Each array takes 40% of the free memory: two fit, three don't.
    cols = 32 * 1024
    rows = int(0.4 * free) // (4 * cols) // 32 * 32
    # The value is an input: a program of constants keeps its result cached
    # (const-eval) and returns a copy, which would take another 40%.
    x = jax.jit(lambda v: jnp.full((rows, cols), v))(jnp.float32(1.5))
    z = jax.jit(_fourth_power, donate_argnums=0)(x)
    assert x.is_deleted()
    corner = jax.jit(lambda a: a[-32:, -32:])(z)
    np.testing.assert_array_equal(np.asarray(corner), np.full((32, 32), 5.0625, np.float32))


def test_donated_input_returned_unchanged_stays_valid():
    x = _small_integers()
    expected = np.asarray(x)
    same, doubled = jax.jit(lambda a: (a, a * 2), donate_argnums=0)(x)
    np.testing.assert_array_equal(np.asarray(same), expected)
    np.testing.assert_array_equal(np.asarray(doubled), 2 * expected)


def test_donating_an_input_that_another_array_shares_keeps_that_array():
    # The double transpose folds away, so the first program returns its input
    # and its output shares the input's device tensor: there is no copy.
    # Donating the input must not free what the output still holds.
    x = _small_integers()
    expected = np.asarray(x)
    shared = jax.jit(lambda a: (a.T.T, a + 1))(x)[0]
    powered = jax.jit(_fourth_power, donate_argnums=0)(x)
    # New arrays take any memory the donation freed.
    fills = [jax.jit(lambda: jnp.full((64, 128), -1.0, jnp.float32))() for _ in range(4)]
    jax.block_until_ready(fills)
    np.testing.assert_array_equal(np.asarray(shared), expected)
    np.testing.assert_array_equal(np.asarray(powered), expected**4)
