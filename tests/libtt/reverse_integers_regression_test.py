"""Reversing integer tensors must keep their exact values."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("dtype", [jnp.int16, jnp.uint16, jnp.int32, jnp.uint32])
@pytest.mark.parametrize("axis", [0, 1, (0, 1)])
def test_flip_integers(dtype, axis):
    rng = np.random.default_rng(0)
    info = np.iinfo(dtype)
    x = rng.integers(info.min, info.max, size=(5, 70), dtype=dtype)
    np.testing.assert_array_equal(jax.jit(lambda a: jnp.flip(a, axis))(x), np.flip(x, axis))
