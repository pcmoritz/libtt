"""Collectives beyond psum: ppermute, pmax/pmin and axis_index."""

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


def _run(mesh, body, x):
    f = jax.shard_map(body, mesh=mesh, in_specs=P("x"), out_specs=P("x"))
    return np.asarray(jax.jit(f)(jax.device_put(x, NamedSharding(mesh, P("x")))))


def test_ppermute_shifts_shards():
    mesh = _mesh()
    n = mesh.devices.size
    x = np.arange(n * 32, dtype=np.float32).reshape(n, 32)
    shift = [(i, (i + 1) % n) for i in range(n)]
    out = _run(mesh, lambda b: jax.lax.ppermute(b, "x", shift), x)
    np.testing.assert_array_equal(out, np.roll(x, 1, axis=0))


@pytest.mark.parametrize("collective,reduce", [(jax.lax.pmax, np.max), (jax.lax.pmin, np.min)])
def test_pmax_pmin(collective, reduce):
    mesh = _mesh()
    n = mesh.devices.size
    x = np.random.default_rng(0).standard_normal((n, 32)).astype(np.float32)
    out = _run(mesh, lambda b: collective(b, "x"), x)
    np.testing.assert_array_equal(out, np.broadcast_to(reduce(x, axis=0), x.shape))


def test_axis_index():
    mesh = _mesh()
    n = mesh.devices.size
    x = np.zeros((n, 1), dtype=np.int32)
    out = _run(mesh, lambda b: b + jax.lax.axis_index("x"), x)
    np.testing.assert_array_equal(out, np.arange(n, dtype=np.int32).reshape(n, 1))
