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


# Per-device shards: one row takes the direct all-reduce (for 16-bit floats),
# large shards reduce-scatter (along columns, or rows for one tile column),
# and single-tile shards gather and reduce locally.
@pytest.mark.parametrize("rows,cols", [(1, 1024), (256, 1024), (256, 32), (3, 32)])
@pytest.mark.parametrize("dtype", [np.float32, jnp.bfloat16, np.int32])
@pytest.mark.parametrize("collective,reduce", [(jax.lax.pmax, np.max), (jax.lax.pmin, np.min)])
def test_pmax_pmin(collective, reduce, dtype, rows, cols):
    mesh = _mesh()
    n = mesh.devices.size
    values = np.random.default_rng(0).standard_normal((n * rows, cols)) * 100
    x = values.astype(dtype)
    out = _run(mesh, lambda b: collective(b, "x"), x)
    expected = reduce(x.reshape(n, rows, cols), axis=0)
    np.testing.assert_array_equal(out, np.tile(expected, (n, 1)))


def test_axis_index():
    mesh = _mesh()
    n = mesh.devices.size
    x = np.zeros((n, 1), dtype=np.int32)
    out = _run(mesh, lambda b: b + jax.lax.axis_index("x"), x)
    np.testing.assert_array_equal(out, np.arange(n, dtype=np.int32).reshape(n, 1))


# Reducing one axis of a 2x2 mesh runs the line reduce-scatter.
@pytest.mark.parametrize("cols", [1024, 32])
@pytest.mark.parametrize("collective,reduce", [(jax.lax.pmax, np.max), (jax.lax.pmin, np.min)])
def test_pmax_pmin_2d_mesh(collective, reduce, cols):
    devices = jax.devices("tt")
    if len(devices) != 4:
        pytest.skip("needs four devices")
    mesh = Mesh(np.array(devices).reshape(2, 2), ("x", "y"))
    x = (np.random.default_rng(0).standard_normal((2, 2 * 256, cols)) * 100).astype(jnp.bfloat16)
    f = jax.shard_map(lambda b: collective(b, "y"), mesh=mesh, in_specs=P("x", "y"), out_specs=P("x", "y"))
    out = np.asarray(jax.jit(f)(jax.device_put(x, NamedSharding(mesh, P("x", "y")))))
    expected = reduce(x.reshape(2, 2, 256, cols), axis=1)
    np.testing.assert_array_equal(out, np.tile(expected, (1, 2, 1)))
