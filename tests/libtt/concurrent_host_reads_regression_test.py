"""Blocking host reads must not stall other threads' device work.

A blocking read waits for every program queued ahead of it. It used to hold the
mesh device's API lock while it waited, so a second thread could not upload
inputs or queue programs until the device drained (SGLang-JAX's overlap
scheduler reads one step's tokens while it queues the next step). The read now
waits outside the lock.
"""

import threading
import time

import jax
import jax.numpy as jnp
import numpy as np


def _slow(a, w):
    # About half a second of device work on one Blackhole chip.
    for _ in range(16):
        a = jnp.tanh(a @ w)
    return a[:1, :8]


def test_dispatch_proceeds_while_a_read_waits():
    device = jax.devices("tt")[0]
    rng = np.random.default_rng(0)
    slow = jax.jit(_slow)
    add_one = jax.jit(lambda y: y + 1)
    a = jax.device_put((rng.standard_normal((8192, 8192)) / 64).astype(jnp.bfloat16), device)
    w = jax.device_put((rng.standard_normal((8192, 8192)) / 64).astype(jnp.bfloat16), device)
    y = np.arange(32 * 32, dtype=np.float32).reshape(32, 32)
    expected_slow = np.asarray(slow(a, w), dtype=np.float32)
    np.asarray(add_one(jax.device_put(y, device)))

    pending = slow(a, w)
    read_done = threading.Event()
    result = {}

    def read():
        result["slow"] = np.asarray(pending, dtype=np.float32)
        read_done.set()

    reader = threading.Thread(target=read)
    reader.start()
    time.sleep(0.05)
    assert not read_done.is_set(), "the slow program finished too early to test anything"

    # Uploads an input and queues a program while the reader waits on the device.
    queued = add_one(jax.device_put(y, device))
    queued_while_reading = not read_done.is_set()

    reader.join(timeout=60)
    assert not reader.is_alive(), "the read hung"
    assert queued_while_reading, "queueing work waited for another thread's read"
    np.testing.assert_allclose(result["slow"], expected_slow)
    np.testing.assert_array_equal(np.asarray(queued), y + 1)


def test_reads_and_dispatch_from_three_threads():
    device = jax.devices("tt")[0]
    # Traced outputs alias slots that every replay overwrites, so share an untraced program.
    step = jax.jit(lambda x, w: jnp.tanh(x @ w) @ w.T)
    w = (np.random.default_rng(0).standard_normal((512, 512)) / 32).astype(jnp.bfloat16)
    w_dev = jax.device_put(w, device)
    w32 = w.astype(np.float32)
    errors = []

    def worker(seed):
        rng = np.random.default_rng(seed)
        try:
            for _ in range(20):
                x = rng.standard_normal((32, 512)).astype(jnp.bfloat16)
                out = np.asarray(step(jax.device_put(x, device), w_dev), dtype=np.float32)
                np.testing.assert_allclose(out, np.tanh(x.astype(np.float32) @ w32) @ w32.T, atol=0.1, rtol=0.05)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(seed,)) for seed in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=300)
    assert not any(thread.is_alive() for thread in threads), "a thread hung"
    assert not errors, errors[0]
