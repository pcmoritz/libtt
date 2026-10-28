"""Programs may run on some of the chips.

Opening a mesh over only some chips starts fabric routers whose ethernet
partners never answer, so libtt then opens a mesh over every chip that runs
nothing itself and runs every program on a submesh of it. Switching submeshes
must hand over the chips' command queues and leave nothing of the submesh being
left on them, as each allocates device memory on its own.
"""

import time

import jax
import numpy as np
import pytest
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


@pytest.fixture
def devices():
    devices = jax.devices("tt")
    if len(devices) != 4:
        pytest.skip("needs exactly four chips")
    return devices


def _run(devices, ids):
    mesh = Mesh(np.array([devices[i] for i in ids]), ("x",))
    x = np.arange(16, dtype=np.float32).reshape(8, 2)
    sharded = jax.device_put(x, NamedSharding(mesh, P("x")))
    out = jax.jit(lambda a: a * 2 + 1)(sharded)
    np.testing.assert_allclose(np.asarray(out), x * 2 + 1)
    total = jax.jit(
        jax.shard_map(lambda a: jax.lax.psum(a, "x"), mesh=mesh, in_specs=P("x"), out_specs=P())
    )(sharded)
    np.testing.assert_allclose(np.asarray(total), x.reshape(len(ids), -1, 2).sum(0))


def test_switching_between_submeshes_and_the_whole_mesh(devices):
    mesh = Mesh(np.array(devices[:2]), ("x",))
    x = np.arange(16, dtype=np.float32).reshape(8, 2)
    double = jax.jit(lambda a: a * 2)
    # Left on the chips, unread, while programs run on other meshes.
    kept = double(jax.device_put(x, NamedSharding(mesh, P("x"))))
    _run(devices, [2, 3])
    _run(devices, [0, 1, 2, 3])
    _run(devices, [0, 1])
    single = jax.jit(lambda a: a + 1)(jax.device_put(np.ones((4, 4), np.float32), devices[3]))
    np.testing.assert_allclose(np.asarray(single), 2)
    _run(devices, [2, 3])
    _run(devices, [0, 1, 2, 3])
    np.testing.assert_allclose(np.asarray(double(kept)), x * 4)
    np.testing.assert_allclose(np.asarray(kept), x * 2)


def test_whole_mesh_after_a_submesh_program_without_inputs(devices):
    """The whole mesh's cached program and const-eval results survive."""
    _run(devices, [0, 1, 2, 3])
    mesh = Mesh(np.array(devices[:2]), ("x",))
    zeros = jax.jit(
        lambda: jnp.arange(8, dtype=jnp.float32).reshape(4, 2) + 1,
        out_shardings=NamedSharding(mesh, P("x")),
    )()
    np.testing.assert_allclose(np.asarray(zeros), np.arange(8).reshape(4, 2) + 1)
    _run(devices, [0, 1, 2, 3])


def test_gather_on_the_whole_mesh_after_programs_on_both(devices):
    """Nothing a whole-mesh program left behind meets a later submesh."""
    mesh = Mesh(np.array(devices).reshape(2, 2), ("x", "y"))
    zeros = jax.jit(
        lambda: jax.lax.with_sharding_constraint(
            jnp.zeros((64, 64), jnp.bfloat16), NamedSharding(mesh, P("x", "y"))
        )
    )()
    del zeros
    _run(devices, [2, 3])
    x = np.arange(16).reshape(8, 2)
    gathered = jax.jit(
        lambda v: jax.lax.with_sharding_constraint(v, NamedSharding(mesh, P("x"))) * 2
    )(jax.device_put(x, NamedSharding(mesh, P("x", "y"))))
    np.testing.assert_array_equal(np.asarray(gathered), x * 2)


def test_one_chip_programs_run_on_the_mesh_in_use(devices):
    """A program over one chip has no collectives, so it runs on every chip of
    the mesh in use, each chip on its own inputs, instead of on a mesh of its
    own. Its inputs may be shards of whole-mesh results or results of other
    one-chip programs, and its results feed both kinds of program."""
    mesh = Mesh(np.array(devices), ("x",))
    n = len(devices)
    x = np.arange(16 * n, dtype=np.float32).reshape(4 * n, 4)
    rows = lambda k: x[4 * k : 4 * k + 4]
    sharded = jax.device_put(x, NamedSharding(mesh, P("x")))
    replicated = jax.device_put(x, NamedSharding(mesh, P()))
    # A whole-mesh cache, updated in place between one-chip programs.
    step = jax.jit(lambda cache, v: cache + v, donate_argnums=0)
    cache = jax.device_put(np.zeros_like(x), NamedSharding(mesh, P("x")))
    triple = jax.jit(lambda a: a * 3)
    parts = []
    for k, device in enumerate(devices):
        cache = step(cache, sharded)
        shard = triple(sharded.addressable_shards[k].data)
        again = triple(shard)
        whole = triple(replicated.addressable_shards[k].data)
        host = triple(jax.device_put(x[k], device))
        made = jnp.arange(4, dtype=jnp.float32) + k
        np.testing.assert_allclose(np.asarray(again), rows(k) * 9)
        np.testing.assert_allclose(np.asarray(shard), rows(k) * 3)
        np.testing.assert_allclose(np.asarray(whole), x * 3)
        np.testing.assert_allclose(np.asarray(host), x[k] * 3)
        np.testing.assert_allclose(np.asarray(made), np.arange(4) + k)
        parts.append(triple(sharded.addressable_shards[k].data))
    np.testing.assert_allclose(np.asarray(cache), x * n)
    # One-chip results, still on the mesh, as the shards of a whole-mesh input.
    combined = jax.make_array_from_single_device_arrays(
        x.shape, NamedSharding(mesh, P("x")), parts
    )
    np.testing.assert_allclose(np.asarray(jax.jit(lambda a: a + 1)(combined)), x * 3 + 1)
    moved = jax.device_put(triple(sharded.addressable_shards[1].data), NamedSharding(mesh, P()))
    np.testing.assert_allclose(np.asarray(jax.jit(lambda a: a * 2)(moved)), rows(1) * 6)


def test_one_chip_programs_keep_the_mesh_open(devices):
    """Alternating whole-mesh and one-chip programs neither reopens the mesh
    nor moves the whole-mesh tensors to the host, which took about a second a
    switch."""
    mesh = Mesh(np.array(devices), ("x",))
    x = np.ones((8 * len(devices), 128), np.float32)
    step = jax.jit(lambda cache: cache * 2 - 1, donate_argnums=0)
    cache = jax.device_put(x, NamedSharding(mesh, P("x")))
    add = jax.jit(lambda a: a + 1)
    for _ in range(2):
        cache = step(cache)
        np.asarray(add(jnp.zeros(4)))
    start = time.monotonic()
    for _ in range(20):
        cache = step(cache)
        np.asarray(add(jnp.zeros(4)))
    elapsed = time.monotonic() - start
    np.testing.assert_allclose(np.asarray(cache), x)
    assert elapsed < 2, f"20 switches took {elapsed:.1f} s"
