"""Permuting 32-bit integer tensors must keep values that float32 cannot hold."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("dtype", [jnp.int32, jnp.uint32])
@pytest.mark.parametrize("perm", [(1, 0, 2), (2, 1, 0), (0, 2, 1), (1, 2, 0)])
def test_transpose_int32(dtype, perm):
    rng = np.random.default_rng(0)
    info = np.iinfo(dtype)
    x = rng.integers(info.min, info.max, size=(3, 40, 70), dtype=dtype)
    np.testing.assert_array_equal(jax.jit(lambda a: jnp.transpose(a, perm))(x), np.transpose(x, perm))
