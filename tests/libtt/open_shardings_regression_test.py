"""tt-xla reads an argument sharding left open for propagation as closed.

JAX may hand over an argument sharding with an open dimension, e.g.
[{"i", ?}]. The argument is laid out by the axes it names so far.
"""

import jax
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh


@pytest.mark.parametrize(
    "in_axes, shape, expected",
    [
        # [{?}, {?}, {?}]
        (None, (1, 3, 2), lambda x: np.concatenate([np.maximum(x, 0)] * 2, axis=1)),
        # [{?}, {"i", ?}, {?}]
        (1, (1, 6, 2), lambda x: np.maximum(x, 0)),
    ],
)
def test_smap_input_with_open_sharding(in_axes, shape, expected):
    devices = jax.devices("tt")
    if len(devices) < 4:
        pytest.skip("needs four chips")
    mesh = Mesh(np.array(devices[:4]).reshape(2, 2), ("i", "j"), axis_types=(AxisType.Auto,) * 2)
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / np.prod(shape) - 0.5
    with jax.set_mesh(mesh):
        out = jax.jit(jax.smap(jax.nn.relu, in_axes=in_axes, out_axes=1, axis_name="i"))(x)
    np.testing.assert_allclose(np.asarray(out), expected(x))
