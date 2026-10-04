"""Buffers allocated between trace replays must not land in a trace's transients.

A captured trace reuses the DRAM its intermediates occupied during capture on
every replay. Once the capture freed them, a first-fit allocator handed that
region to the next small buffer, so the trace was no longer safe to replay and
was captured again on every call (SGLang-JAX's overlap scheduler allocates a few
small tensors between steps). Captured traces now keep their transient DRAM
reserved: small buffers allocated between replays must not cause a capture,
and must keep their values.
"""

import ctypes
from pathlib import Path

import jax
import jax.numpy as jnp
import jax_plugins.libtt
import numpy as np

OPTIONS = {"optimization_level": "O1", "enable_trace": "true"}


def _hook(name):
    # Test hooks exported by the plugin; the library is already loaded by JAX.
    lib = ctypes.CDLL(str(Path(jax_plugins.libtt.__file__).with_name("libtt.so")))
    getattr(lib, name).restype = ctypes.c_uint64
    return getattr(lib, name)()


def _mesh_trace_captures():
    return _hook("libtt_mesh_trace_captures")


def _trace_reserved_dram_bytes():
    return _hook("libtt_trace_reserved_dram_bytes")


def _step(x, w):
    h = jnp.tanh(x @ w) * 2 + 1
    return jnp.argmax(h @ w.T + x, axis=-1)


def _expected(x, w):
    h = np.tanh(x @ w) * 2 + 1
    return h @ w.T + x


def _check_step(index, x, w):
    logits = _expected(x.astype(np.float32), w.astype(np.float32))
    chosen = np.take_along_axis(logits, np.asarray(index)[:, None], axis=-1)[:, 0]
    np.testing.assert_allclose(chosen, logits.max(axis=-1), atol=0.25)


# An untraced program, so its outputs are allocated on device (device_put
# alone keeps a host copy until a program consumes it).
_copy = jax.jit(lambda v: v + 0)


def _small_buffers(device, i):
    # Small device buffers between steps, like the overlap scheduler's token ids.
    # Enough of them to fill older free holes and reach the trace's transients.
    return [
        (i * 100 + j, _copy(jax.device_put(np.full((1 + j, 32), i * 100 + j, np.int32), device)))
        for j in range(16)
    ]


def test_buffers_allocated_between_replays_do_not_recapture():
    device = jax.devices("tt")[0]
    rng = np.random.default_rng(0)
    step = jax.jit(_step, compiler_options=OPTIONS)
    w = (rng.standard_normal((256, 512)) / 16).astype(jnp.bfloat16)
    w_dev = jax.device_put(w, device)

    x = rng.standard_normal((32, 256)).astype(jnp.bfloat16)
    _check_step(step(jax.device_put(x, device), w_dev), x, w)
    captures = _mesh_trace_captures()

    kept = []
    for i in range(1, 7):
        kept += _small_buffers(device, i)
        x = rng.standard_normal((32, 256)).astype(jnp.bfloat16)
        _check_step(step(jax.device_put(x, device), w_dev), x, w)
        for value, buffer in kept:
            np.testing.assert_array_equal(np.asarray(buffer), value)
    assert _mesh_trace_captures() == captures


def test_second_capture_keeps_first_trace_reserved():
    # Capturing another trace frees the shared pool while it runs, then must
    # reserve it again for both traces.
    device = jax.devices("tt")[0]
    rng = np.random.default_rng(1)
    first = jax.jit(_step, compiler_options=OPTIONS)
    second = jax.jit(lambda x: jnp.sum(jnp.exp(x * 0.5) - x, axis=-1), compiler_options=OPTIONS)
    w = (rng.standard_normal((256, 512)) / 16).astype(jnp.bfloat16)
    w_dev = jax.device_put(w, device)
    x = rng.standard_normal((32, 256)).astype(jnp.bfloat16)
    x_dev = jax.device_put(x, device)

    _check_step(first(x_dev, w_dev), x, w)
    second(x_dev).block_until_ready()
    assert _trace_reserved_dram_bytes() > 0
    captures = _mesh_trace_captures()

    kept = []
    for i in range(1, 5):
        kept += _small_buffers(device, i)
        _check_step(first(x_dev, w_dev), x, w)
        np.testing.assert_allclose(
            np.asarray(second(x_dev)),
            np.sum(np.exp(x.astype(np.float32) * 0.5) - x.astype(np.float32), axis=-1),
            rtol=0.05,
            atol=1.0,
        )
    for value, buffer in kept:
        np.testing.assert_array_equal(np.asarray(buffer), value)
    assert _mesh_trace_captures() == captures
