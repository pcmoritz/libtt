"""Programs may run on some of the chips.

Opening a mesh over only some chips starts fabric routers whose ethernet
partners never answer, so libtt then opens a mesh over every chip that runs
nothing itself and runs every program on a submesh of it. Switching submeshes
must hand over the chips' command queues and leave nothing of the submesh being
left on them, as each allocates device memory on its own.
"""

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
