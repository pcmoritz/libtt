"""Sharding constraints are no-ops and shard_map collectives compile and run."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def test_sharding_constraint_is_identity():
    device = jax.devices("tt")[0]
    x = jnp.arange(8 * 64, dtype=jnp.float32).reshape(8, 64)
    constrained = jax.jit(
        lambda v: jax.lax.with_sharding_constraint(
            v * 2, jax.sharding.SingleDeviceSharding(device)
        )
        + 1
    )
    np.testing.assert_array_equal(np.asarray(constrained(x)), np.asarray(x) * 2 + 1)


def test_shard_map_all_gather():
    devices = jax.devices("tt")
    if len(devices) < 2:
        pytest.skip("needs at least two devices")
    mesh = Mesh(np.array(devices), ("x",))
    x = np.arange(8 * len(devices), dtype=np.float32)
    gather = jax.shard_map(
        lambda b: jax.lax.all_gather(b, "x", tiled=True),
        mesh=mesh,
        in_specs=P("x"),
        out_specs=P(),
        check_vma=False,
    )
    out = jax.jit(gather)(jax.device_put(x, NamedSharding(mesh, P("x"))))
    np.testing.assert_array_equal(np.asarray(out), x)
