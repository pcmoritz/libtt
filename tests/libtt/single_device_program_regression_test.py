"""Single-device programs must not reopen a multi-chip mesh.

JAX runs eager ops on uncommitted arrays (for example an index built from a
Python int) as single-device programs. They used to make the plugin close the
multi-chip mesh, move every device tensor to host, and reopen the mesh at
shape (1, 1), and then back again for the next multi-chip program: about half
a second per switch, and seconds per request in SGLang-JAX.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P


def _mesh_and_sharding():
    devices = jax.devices("tt")
    if len(devices) < 2:
        pytest.skip("needs at least two devices")
    mesh = jax.make_mesh((len(devices),), ("x",), devices=devices)
    return devices, NamedSharding(mesh, P("x"))


def test_alternating_single_and_multi_device_programs():
    devices, sharded = _mesh_and_sharding()
    values = np.arange(len(devices) * 64, dtype=np.float32).reshape(len(devices), 64)
    x = jax.device_put(values, sharded)
    step = jax.jit(lambda a: a * 2.0 + 1.0, out_shardings=sharded)

    def alternate(i):
        n = jnp.asarray(i, dtype=jnp.int32) + 1  # Single-device programs.
        y = step(x)  # A program on every device.
        return n, y

    for i in range(3):
        alternate(i)
    start = time.perf_counter()
    for i in range(20):
        n, y = alternate(i)
        assert int(n) == i + 1
    jax.block_until_ready(y)
    elapsed = time.perf_counter() - start

    np.testing.assert_array_equal(np.asarray(y), values * 2.0 + 1.0)
    # Reopening the mesh took about 0.45 s per alternation; without it an
    # alternation takes about a millisecond.
    assert elapsed < 4.0, f"20 alternations took {elapsed:.2f} s"


def test_single_device_program_reads_mesh_array():
    devices, sharded = _mesh_and_sharding()
    values = np.arange(len(devices) * 32, dtype=np.float32).reshape(len(devices), 32)
    x = jax.device_put(values, sharded)
    # A shard of the mesh array, read by a single-device program.
    first = x.addressable_shards[0].data
    np.testing.assert_array_equal(np.asarray(first * 3.0), values[:1] * 3.0)
    # The mesh array stays usable by programs on every device.
    double = jax.jit(lambda a: a * 2.0, out_shardings=sharded)
    np.testing.assert_array_equal(np.asarray(double(x)), values * 2.0)
