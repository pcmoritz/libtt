"""Copies of results of programs over one device stay on the device.

A program over one device runs on every chip of the mesh in use, so its results
hold the same data on every chip. JAX copies such a result to other devices
whenever a whole-mesh program takes it, for example the result of an eager op on
the default device. Those copies are made on the device instead of reading the
result back through the host, which waited for every program queued before.

The first call of each program over one device compiles it, which still opens a
mesh of its own (tenstorrent/tt-xla#4809), so the tests run their programs more
than once.
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


def test_copies_of_one_device_results(mesh):
    n = mesh.devices.size
    replicated = NamedSharding(mesh, P())
    scale = jax.jit(lambda a, s: a * s)
    x = jax.device_put(np.arange(8 * n, dtype=np.float32).reshape(2 * n, 4), NamedSharding(mesh, P("x")))
    for k in range(4):
        s = jnp.asarray(k + 2, dtype=jnp.float32)
        np.testing.assert_allclose(np.asarray(scale(x, s)), np.asarray(x) * (k + 2))
        made = jnp.arange(4, dtype=jnp.float32) + k
        moved = jax.device_put(made, replicated)
        for shard in moved.addressable_shards:
            np.testing.assert_allclose(np.asarray(shard.data), np.arange(4) + k)
        np.testing.assert_allclose(np.asarray(scale(x, moved)), np.asarray(x) * (np.arange(4) + k))
        # A replicated input whose first shard holds its data on every chip and
        # whose other shards are host data.
        others = [jax.device_put(np.arange(4, dtype=np.float32) + k, d) for d in mesh.devices.flat[1:]]
        mixed = jax.make_array_from_single_device_arrays((4,), replicated, [made, *others])
        np.testing.assert_allclose(np.asarray(scale(x, mixed)), np.asarray(x) * (np.arange(4) + k))
        on_last = jax.device_put(made, mesh.devices.flat[-1])
        np.testing.assert_allclose(np.asarray(on_last), np.arange(4) + k)
    # A replicated whole-mesh result copied to one device.
    doubled = jax.jit(lambda a: a * 2, out_shardings=replicated)(x)
    single = jax.device_put(doubled, mesh.devices.flat[1])
    np.testing.assert_allclose(np.asarray(single), np.asarray(x) * 2)


def test_donated_copies_keep_their_source(mesh):
    """A program may update a donated input in place; a copy has its own data,
    so its source keeps its values."""
    shape, dtype = (2, 128, 3), jnp.bfloat16
    device = mesh.devices.flat[1]

    def update(state, indices, values):
        return jax.ffi.ffi_call(
            "tt.state_pool_update",
            jax.ShapeDtypeStruct(state.shape, state.dtype),
            input_output_aliases={0: 0},
            vmap_method="sequential",
        )(state, indices, values)

    update = jax.jit(update, donate_argnums=(0,))
    make = jax.jit(lambda: jnp.full(shape, 3, dtype))
    step = jax.jit(lambda a: a + 1)
    x = jax.device_put(np.zeros((8, 4), np.float32), NamedSharding(mesh, P("x")))
    indices = jax.device_put(np.array([1], np.int32), device)
    values = jax.device_put(np.full((1, *shape[1:]), 7, np.dtype(dtype)), device)
    expected = np.full(shape, 3, np.dtype(dtype))
    expected[1] = 7
    for _ in range(3):
        x = step(x)  # back to the whole mesh
        made = make()
        updated = update(jax.device_put(made, device), indices, values)
        np.testing.assert_array_equal(np.asarray(updated), expected)
        np.testing.assert_array_equal(np.asarray(made), 3)


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
    two = lambda: jnp.ones(()) * 2  # an eager op, queued behind the long program
    for _ in range(2):
        np.asarray(use(long(v), two())[0, 0])
        np.asarray(long(v)[0, 0])
    start = time.monotonic()
    np.asarray(long(v)[0, 0])
    run = time.monotonic() - start
    out = long(v)
    start = time.monotonic()
    s = two()
    used = use(out, s)
    queued = time.monotonic() - start
    np.testing.assert_allclose(np.asarray(used[0, 0]), 2 * np.asarray(out[0, 0]))
    assert queued < run / 2, f"queueing took {queued * 1e3:.1f} ms behind a {run * 1e3:.1f} ms program"


def test_copies_that_fail_raise(mesh):
    """A copy can fail, here for want of device memory, and returns its error
    through PJRT: JAX raises a JaxRuntimeError rather than the plugin's own
    exception passing through the C API."""
    device, other = mesh.devices.flat[0], mesh.devices.flat[1]
    x = jax.device_put(np.ones((mesh.devices.size, 4), np.float32), NamedSharding(mesh, P("x")))
    step = jax.jit(lambda a: a + 1)
    # n GiB on every chip.
    fill = jax.jit(lambda s, n: jnp.broadcast_to(s, (4 * n, 8192, 8192)), static_argnums=1)
    s = jax.device_put(np.float32(0.5), device)
    for _ in range(2):
        np.asarray(step(x))
        jax.block_until_ready(fill(s, 4))
        jax.block_until_ready(fill(s, 1))
    np.asarray(step(x))

    big = fill(s, 4)
    held = []
    with pytest.raises(Exception, match="Out of Memory"):
        for _ in range(64):
            held.append(fill(s, 1))
    with pytest.raises(jax.errors.JaxRuntimeError, match="Out of Memory"):
        jax.device_put(big, other)
    del held, big
    np.testing.assert_allclose(np.asarray(jax.device_put(s, other)), 0.5)
    np.testing.assert_allclose(np.asarray(step(x)), 2.0)
