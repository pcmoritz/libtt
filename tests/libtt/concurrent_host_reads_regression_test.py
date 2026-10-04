"""Blocking host reads must not stall other threads' device work.

A blocking read waits for every program queued ahead of it. It used to hold the
mesh device's API lock while it waited, so a second thread could not upload
inputs or queue programs until the device drained (SGLang-JAX's overlap
scheduler reads one step's tokens while it queues the next step). The read now
waits outside the lock; reads and dispatch from several threads must stay
correct. tests/tt_metal/test_mesh_command_queue_reads.cpp checks that a write
queues while a read waits.
"""

import threading

import jax
import jax.numpy as jnp
import numpy as np


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
