"""A multi-device input may be assembled from shards still on a chip.

tt-xla builds a multi-device input from host shards, so shards left on a chip
by an earlier program must move to the host first. Checking second-order
forward-mode gradients of a shard_map hands over such shards.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jax.test_util import check_grads


def test_second_order_forward_gradients_of_a_shard_map():
    devices = jax.devices("tt")
    if len(devices) < 4:
        pytest.skip("needs four chips")
    mesh = Mesh(np.array(devices[:4]).reshape(2, 2), ("x", "y"))

    @jax.jit
    def f(x, y):
        return jax.shard_map(
            lambda x, y: jnp.sin(x) + 3 + jnp.tan(2.0) * jnp.cos(x) + y,
            mesh=mesh,
            in_specs=(P("x"), P(None)),
            out_specs=P("x"),
        )(x, y)

    x = jnp.arange(8.0) / 10
    y = jnp.arange(4.0) / 10
    check_grads(f, (x, y), modes=["fwd"], order=2, atol=5e-2, rtol=5e-2)
