"""Programs may use several meshes over the same devices.

JAX lets every shard_map or sharding choose its own mesh, e.g. a layer that
arranges the model's (data, tensor) devices as (expert, tensor). tt-mlir
compiles a program for one mesh, so libtt rewrites shardings on the other
meshes in terms of one of them, splitting axes into sub-axes where the shapes
differ.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


@pytest.fixture
def devices():
    devices = jax.devices("tt")
    if len(devices) != 4:
        pytest.skip("needs exactly four chips")
    return np.array(devices)


def test_reduction_on_renamed_mesh(devices):
    """A layer's own (expert, model) mesh next to the model's (data, tensor)."""
    outer = Mesh(devices.reshape(1, 4), ("data", "tensor"))
    inner = Mesh(devices.reshape(1, 4), ("expert", "model"))

    @jax.jit
    def run(x, w, bias):
        partial = jax.shard_map(
            lambda x, w: jax.lax.psum(x @ w, "model"),
            mesh=inner,
            in_specs=(P(None, "model"), P("model", None)),
            out_specs=P(),
        )(x, w)
        return jax.lax.with_sharding_constraint(partial + bias, NamedSharding(outer, P()))

    rng = np.random.default_rng(0)
    x = rng.standard_normal((8, 64)).astype(np.float32)
    w = rng.standard_normal((64, 32)).astype(np.float32)
    bias = rng.standard_normal((8, 32)).astype(np.float32)
    out = run(
        jax.device_put(x, NamedSharding(outer, P())),
        jax.device_put(w, NamedSharding(inner, P("model", None))),
        jax.device_put(bias, NamedSharding(outer, P())),
    )
    np.testing.assert_allclose(np.asarray(out), x @ w + bias, rtol=2e-2, atol=2e-1)


@pytest.mark.parametrize(
    "outer_shape,inner_shape,inner_axes",
    [
        ((2, 2), (2, 2), ("expert", "model")),  # Renamed axes.
        ((1, 4), (2, 2), ("expert", "model")),  # Sub-axes of one outer axis.
        ((2, 2), (1, 4), "model"),  # One inner axis spans both outer axes.
    ],
)
def test_sharding_on_reshaped_mesh(devices, outer_shape, inner_shape, inner_axes):
    outer = Mesh(devices.reshape(outer_shape), ("data", "tensor"))
    inner = Mesh(devices.reshape(inner_shape), ("expert", "model"))

    @jax.jit
    def run(x, y):
        z = jax.shard_map(
            lambda x: jnp.tanh(x) * 2,
            mesh=inner,
            in_specs=P(None, inner_axes),
            out_specs=P(None, inner_axes),
        )(x)
        return z + y

    rng = np.random.default_rng(1)
    x = rng.standard_normal((8, 64)).astype(np.float32)
    y = rng.standard_normal((8, 64)).astype(np.float32)
    out = run(
        jax.device_put(x, NamedSharding(inner, P(None, inner_axes))),
        jax.device_put(y, NamedSharding(outer, P())),
    )
    np.testing.assert_allclose(np.asarray(out), np.tanh(x) * 2 + y, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("shape", [(1, 4, 1), (1, 2, 2)])
def test_mesh_with_size_one_axes(devices, shape):
    """Meshes of more than two axes, all but two of size one, become 2D."""
    mesh = Mesh(devices.reshape(shape), ("a", "b", "c"))
    spec = P("a", "b", "c")
    x = np.arange(2 * 16 * 4, dtype=np.float32).reshape(2, 16, 4)
    run = jax.jit(jax.shard_map(lambda x: x * x + 2, mesh=mesh, in_specs=spec, out_specs=spec))
    out = run(jax.device_put(x, NamedSharding(mesh, spec)))
    np.testing.assert_allclose(np.asarray(out), x * x + 2, rtol=1e-2)


def test_axis_index_without_operands_on_size_one_axis(devices):
    """A shard_map without operands still names the dropped axis as manual."""
    mesh = Mesh(devices.reshape(1, 2, 2), ("x", "y", "z"))

    def indices():
        return jnp.array([jax.lax.axis_index(name) for name in mesh.axis_names])

    out = jax.jit(jax.shard_map(indices, mesh=mesh, in_specs=(), out_specs=P(mesh.axis_names)))()
    expected = [[0, y, z] for y in range(2) for z in range(2)]
    np.testing.assert_array_equal(np.asarray(out), np.ravel(expected))


@pytest.mark.parametrize("rows", [4, 2])
def test_mesh_with_trailing_size_one_axis(devices, rows):
    """tt-mlir takes (1, n) meshes; an (n, 1) mesh is the same devices."""
    mesh = Mesh(devices[:rows].reshape(rows, 1), ("x", "y"))
    x = np.arange(8 * 4, dtype=np.float32).reshape(8, 4)
    run = jax.jit(
        jax.shard_map(
            lambda x: jax.lax.psum(x, "x"), mesh=mesh, in_specs=P("x", "y"), out_specs=P(None, "y")
        )
    )
    out = run(jax.device_put(x, NamedSharding(mesh, P("x", "y"))))
    np.testing.assert_allclose(np.asarray(out), x.reshape(rows, -1, 4).sum(0))
