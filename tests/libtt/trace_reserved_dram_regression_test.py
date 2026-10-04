"""Buffers allocated between trace replays must not land in a trace's transients.

A captured trace reuses the DRAM its intermediates occupied during capture on
every replay. Once the capture freed them, a first-fit allocator handed that
region to the next small buffer, so the trace was no longer safe to replay and
was captured again on every call (SGLang-JAX's overlap scheduler allocates a few
small tensors between steps). Captured traces now keep their transient DRAM
reserved, and buffers allocated between replays must keep their values.
"""

import jax
import jax.numpy as jnp
import numpy as np

OPTIONS = {"optimization_level": "O1", "enable_trace": "true"}


def _step(x, w):
    h = jnp.tanh(x @ w) * 2 + 1
    return jnp.argmax(h @ w.T + x, axis=-1)


def _expected(x, w):
    h = np.tanh(x @ w) * 2 + 1
    return h @ w.T + x


def test_buffers_allocated_between_replays_keep_values():
    device = jax.devices("tt")[0]
    rng = np.random.default_rng(0)
    step = jax.jit(_step, compiler_options=OPTIONS)
    other = jax.jit(lambda x: jnp.sum(jnp.exp(x * 0.5) - x, axis=-1), compiler_options=OPTIONS)
    w = rng.standard_normal((256, 512)).astype(np.float32) / 16
    w_dev = jax.device_put(w.astype(jnp.bfloat16), device)

    kept = []
    for i in range(6):
        x = rng.standard_normal((32, 256)).astype(jnp.bfloat16)
        x_dev = jax.device_put(x, device)
        index = np.asarray(step(x_dev, w_dev))
        logits = _expected(x.astype(np.float32), w.astype(jnp.bfloat16).astype(np.float32))
        chosen = np.take_along_axis(logits, index[:, None], axis=-1)[:, 0]
        np.testing.assert_allclose(chosen, logits.max(axis=-1), atol=0.25)

        totals = np.asarray(other(x_dev))
        np.testing.assert_allclose(
            totals,
            np.sum(np.exp(x.astype(np.float32) * 0.5) - x.astype(np.float32), axis=-1),
            rtol=0.05,
            atol=1.0,
        )

        # Small buffers between steps, like the overlap scheduler's token ids.
        kept.append((i, jax.device_put(np.full((1, 32), i, np.int32), device)))
        kept.append((-i, jax.device_put(np.full((32, 32), -i, np.int32), device)))
        for value, buffer in kept:
            np.testing.assert_array_equal(np.asarray(buffer), value)
