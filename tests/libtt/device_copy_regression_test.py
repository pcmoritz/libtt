"""Copies of device data with the same values on every chip stay on the device.

A program over one chip runs on every chip of the mesh in use, so a result it
computes from replicated or host data is on every chip, as is a replicated
whole-mesh result. JAX copies such values to other devices whenever a
whole-mesh program takes them, for example the result of an eager op on the
default device. Those copies share the device data instead of reading it back
through the host, which waited for every program queued before.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


@pytest.fixture
def mesh():
    devices = jax.devices("tt")
    if len(devices) < 2:
        pytest.skip("needs several chips")
    return Mesh(np.array(devices), ("x",))


def test_copies_of_one_chip_results(mesh):
    n = mesh.devices.size
    replicated = NamedSharding(mesh, P())
    scale = jax.jit(lambda a, s: a * s)
    x = jax.device_put(np.arange(8 * n, dtype=np.float32).reshape(2 * n, 4), NamedSharding(mesh, P("x")))
    for k in range(4):
        s = jnp.asarray(k + 2, dtype=jnp.float32)  # one chip, no inputs
        np.testing.assert_allclose(np.asarray(scale(x, s)), np.asarray(x) * (k + 2))
        made = jnp.arange(4, dtype=jnp.float32) + k  # one chip, from another one-chip result
        moved = jax.device_put(made, replicated)
        for shard in moved.addressable_shards:
            np.testing.assert_allclose(np.asarray(shard.data), np.arange(4) + k)
        on_last = jax.device_put(made, mesh.devices.flat[-1])
        np.testing.assert_allclose(np.asarray(on_last), np.arange(4) + k)
    # A replicated whole-mesh result copied to one device, and back.
    doubled = jax.jit(lambda a: a * 2, out_shardings=replicated)(x)
    single = jax.device_put(doubled, mesh.devices.flat[1])
    np.testing.assert_allclose(np.asarray(single), np.asarray(x) * 2)


def test_donated_copies_keep_their_source(mesh):
    replicated = NamedSharding(mesh, P())
    made = jnp.arange(8, dtype=jnp.float32) * 3
    copy = jax.device_put(made, replicated)
    bump = jax.jit(lambda a: a + 1, donate_argnums=0)
    bumped = bump(copy)
    np.testing.assert_allclose(np.asarray(bumped), np.arange(8) * 3 + 1)
    np.testing.assert_allclose(np.asarray(made), np.arange(8) * 3)
    again = jax.device_put(made, replicated)
    np.testing.assert_allclose(np.asarray(jax.jit(lambda a: a * 2)(again)), np.arange(8) * 6)


def test_copies_do_not_wait_for_the_device(mesh):
    """Taking an eager result into a whole-mesh program while a long one runs
    returns before that program finishes."""
    replicated = NamedSharding(mesh, P())
    # Large enough for the device, not the host's dispatch, to take the time.
    w = jax.device_put(np.eye(4096, dtype=np.float32), NamedSharding(mesh, P(None, "x")))

    @jax.jit
    def long(v):
        for _ in range(20):
            v = jax.sharding.reshard(jnp.tanh(v @ w), replicated)
        return v

    v = jax.device_put(np.ones((4096, 4096), np.float32), replicated)
    use = jax.jit(lambda a, s: a * s)
    two = lambda: jnp.ones(()) * 2  # one chip, run behind what is queued
    for _ in range(2):
        np.asarray(use(long(v), two())[0, 0])
        np.asarray(long(v)[0, 0])
    start = time.monotonic()
    np.asarray(long(v)[0, 0])
    run = time.monotonic() - start
    out = long(v)
    s = two()
    start = time.monotonic()
    use(out, s)
    queued = time.monotonic() - start
    np.asarray(out[0, 0])
    assert queued < run / 2, f"queueing took {queued * 1e3:.1f} ms behind a {run * 1e3:.1f} ms program"
