"""A multi-device input may be assembled from shards still on their chips.

tt-xla builds a multi-device input from host shards, so a shard that is the
result of an earlier single-chip program must move to the host first.
"""

import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


def test_sharded_input_from_single_chip_results():
    devices = jax.devices("tt")
    if len(devices) < 2:
        pytest.skip("needs two chips")
    devices = devices[:2]
    x = np.arange(16, dtype=np.float32).reshape(8, 2)
    add_one = jax.jit(lambda a: a + 1)
    shards = [add_one(jax.device_put(x[4 * i : 4 * i + 4], d)) for i, d in enumerate(devices)]
    sharding = NamedSharding(Mesh(np.array(devices), ("x",)), P("x"))
    arr = jax.make_array_from_single_device_arrays(x.shape, sharding, shards)
    out = jax.jit(lambda a: a * 2)(arr)
    np.testing.assert_allclose(np.asarray(out), (x + 1) * 2)
