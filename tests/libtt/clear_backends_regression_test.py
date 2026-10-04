"""Clearing the backends must leave a working client that owns the device."""

import jax
from jax._src import api
import jax.numpy as jnp
import numpy as np


def test_compute_after_clear_backends():
    x = jnp.arange(8, dtype=jnp.float32)
    np.testing.assert_allclose(jax.jit(lambda v: v + 1)(x), np.arange(8) + 1)

    # JAX creates the new client before the old one is destroyed.
    api.clear_backends()
    assert jax.devices()[0].platform == "tt"

    y = jnp.arange(8, dtype=jnp.float32)
    np.testing.assert_allclose(jax.jit(lambda v: v * 2)(y), np.arange(8) * 2)
