"""tt-xla reads an argument sharding left open for propagation as closed.

JAX may hand over an argument sharding with an open dimension, e.g.
[{"i", ?}]. The argument is laid out by the axes it names so far.
"""

import jax
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh


def test_smap_of_an_unmapped_input():
    devices = jax.devices("tt")
    if len(devices) < 4:
        pytest.skip("needs four chips")
    mesh = Mesh(np.array(devices[:4]).reshape(2, 2), ("i", "j"), axis_types=(AxisType.Auto,) * 2)
    x = np.arange(6, dtype=np.float32).reshape(1, 3, 2) / 6 - 0.5
    with jax.set_mesh(mesh):
        out = jax.jit(jax.smap(jax.nn.relu, in_axes=None, out_axes=1, axis_name="i"))(x)
    np.testing.assert_allclose(np.asarray(out), np.concatenate([np.maximum(x, 0)] * 2, axis=1))
