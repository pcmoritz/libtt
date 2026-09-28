"""Inputs that JAX already placed on every device of a mesh.

An unsharded argument of a multi-device program, e.g. a slice's start index
or a NumPy array next to a sharded one, arrives replicated on each device.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, NamedSharding, PartitionSpec as P


def _mesh(axis_type=AxisType.Auto):
    devices = jax.devices("tt")
    if len(devices) < 2:
        pytest.skip("needs at least two devices")
    return jax.make_mesh((1, len(devices)), ("data", "tensor"), (axis_type, axis_type), devices=devices)


@pytest.mark.parametrize("axis_type", [AxisType.Auto, AxisType.Explicit])
def test_eager_indexing_of_replicated_array(axis_type):
    # Loading weights slices and gathers replicated arrays eagerly.
    mesh = _mesh(axis_type)
    x = np.arange(64 * 32, dtype=np.float32).reshape(64, 32)
    xs = jax.device_put(x, NamedSharding(mesh, P()))
    np.testing.assert_array_equal(np.asarray(xs[:, 4:9]), x[:, 4:9])
    np.testing.assert_array_equal(np.asarray(xs[jnp.array([3, 1, 5])]), x[[3, 1, 5]])


def test_numpy_argument_next_to_sharded_argument():
    mesh = _mesh()
    x = (np.arange(8 * 64, dtype=np.float32).reshape(8, 64) % 7) / 4
    w = (np.arange(64 * 64, dtype=np.float32).reshape(64, 64) % 5) / 4
    ws = jax.device_put(w, NamedSharding(mesh, P(None, "tensor")))
    out = jax.jit(lambda a, b: a @ b)(x, ws)
    np.testing.assert_allclose(np.asarray(out), x @ w, rtol=1e-2)


def test_replicated_result_as_argument():
    # A replicated result of one program is already a single tensor spread
    # over the mesh when the next program takes it.
    mesh = _mesh()
    x = np.arange(64 * 32, dtype=np.float32).reshape(64, 32)
    xs = jax.device_put(x, NamedSharding(mesh, P()))
    doubled = jax.jit(lambda a: a * 2, out_shardings=NamedSharding(mesh, P()))(xs)
    np.testing.assert_array_equal(np.asarray(doubled[:, 4:9]), 2 * x[:, 4:9])
