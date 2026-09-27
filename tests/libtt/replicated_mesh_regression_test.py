"""Reductions and replicated values on a mesh.

Covers JAX's empty placeholder mesh, per-device slice bounds, and the
cross-device combine of reductions over sharded dimensions.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def _mesh():
    devices = jax.devices("tt")
    if len(devices) < 2:
        pytest.skip("needs at least two devices")
    return Mesh(np.array(devices), ("x",))


def test_dynamic_slice_with_replicated_index():
    # The replicated index lives on the empty mesh next to the real one, and
    # the slice bounds reach the runtime as one tensor per device.
    mesh = _mesh()
    n = mesh.devices.size
    x = np.arange(8 * n, dtype=np.float32)
    f = jax.shard_map(
        lambda b: jax.lax.dynamic_slice_in_dim(b, jnp.array(1, dtype=np.int32), 2),
        mesh=mesh,
        in_specs=P("x"),
        out_specs=P("x"),
    )
    out = np.asarray(jax.jit(f)(jax.device_put(x, NamedSharding(mesh, P("x")))))
    np.testing.assert_array_equal(out, x.reshape(n, 8)[:, 1:3].reshape(-1))


def test_explicit_mesh_replicated_values():
    # Under an explicit mesh, JAX puts replicated values on an empty placeholder
    # mesh, next to the real mesh or as the only one; the latter programs run
    # replicated over all partitions.
    devices = jax.devices("tt")
    if len(devices) < 2:
        pytest.skip("needs at least two devices")
    mesh = jax.make_mesh((len(devices),), ("x",), (jax.sharding.AxisType.Explicit,), devices=devices)
    values = np.arange(8 * len(devices) * 4, dtype=np.int32).reshape(8 * len(devices), 4)
    with jax.set_mesh(mesh):
        edges = jnp.histogram_bin_edges(jax.device_put(values, P("x")), bins=100)
    np.testing.assert_allclose(np.asarray(edges), np.histogram_bin_edges(values, bins=100), rtol=1e-6)


@pytest.mark.parametrize("reduce", [jnp.min, jnp.max, jnp.sum])
@pytest.mark.parametrize("axis", [None, 0])
def test_reduce_over_sharded_axis(reduce, axis):
    # Shardy combines partial results with the reduction's own computation.
    mesh = _mesh()
    values = np.arange(8 * mesh.devices.size * 4, dtype=np.int32).reshape(-1, 4)
    x = jax.device_put(values, NamedSharding(mesh, P("x")))
    out = jax.jit(lambda v: reduce(v, axis=axis))(x)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(reduce(values, axis=axis)))
