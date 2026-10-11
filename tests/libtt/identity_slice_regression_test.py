"""A slice of a whole tensor with unit steps folds to its input. Unfolded, it
could be commuted past a reshape of a scalar onto the rank-0 tensor, which
ttnn.slice cannot take ('ttnn.slice_static' op Input must be at least a 1D
tensor), as in lu_pivots_to_permutation under vmap and Rotation.as_euler.
"""

import jax
import numpy as np
import pytest
from jax import lax
from jax.scipy.spatial.transform import Rotation


def on_tt(f, *args):
    return jax.jit(f)(*(jax.device_put(a, jax.devices("tt")[0]) for a in args))


def on_cpu(f, *args):
    return jax.jit(f)(*(jax.device_put(a, jax.devices("cpu")[0]) for a in args))


@pytest.mark.parametrize("batch", [1, 3])
def test_lu_pivots_to_permutation_vmap(batch):
    pivots = np.random.default_rng(0).integers(0, 5, size=(batch, 4)).astype(np.int32)
    f = jax.vmap(lambda p: lax.linalg.lu_pivots_to_permutation(p, 5))
    np.testing.assert_array_equal(np.asarray(on_tt(f, pivots)), np.asarray(on_cpu(f, pivots)))


@pytest.mark.parametrize("seq", ["xyz", "zyx", "XYZ", "zxz", "ZXZ"])
def test_rotation_as_euler(seq):
    quat = np.random.default_rng(1).normal(size=(3, 4)).astype(np.float32)
    f = lambda q: Rotation.from_quat(q).as_euler(seq)
    np.testing.assert_allclose(np.asarray(on_tt(f, quat)), np.asarray(on_cpu(f, quat)), rtol=1e-4, atol=1e-4)
