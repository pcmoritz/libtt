"""Transposes of host data stay on the host.

A program that only transposes its argument runs on the host when the argument
is in host memory, as a weight being loaded from a checkpoint is: the result is
a host array, like one the caller put, and takes no device memory. An argument
on the device is transposed there.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P


def device_bytes(devices):
    return [d.memory_stats()["bytes_in_use"] for d in devices]


def transpose_and_device_bytes(x, permutation):
    devices = x.sharding.device_set
    before = device_bytes(devices)
    y = jnp.transpose(x, permutation).block_until_ready()
    return y, before, device_bytes(devices)


@pytest.mark.parametrize(
    "shape,permutation,dtype",
    [
        ((256, 512), (1, 0), jnp.bfloat16),
        ((4, 64, 96), (0, 2, 1), jnp.bfloat16),
        ((64, 32), (1, 0), np.float32),
    ],
)
def test_transpose_of_host_data_stays_on_host(shape, permutation, dtype):
    device = jax.devices("tt")[0]
    host = np.random.default_rng(0).standard_normal(shape).astype(dtype)
    x = jax.device_put(host, device)
    y, before, after = transpose_and_device_bytes(x, permutation)
    assert after == before
    np.testing.assert_array_equal(np.asarray(y), np.transpose(host, permutation))
    # The result is an ordinary input of later programs.
    z = jax.jit(lambda a: a * 2)(y)
    np.testing.assert_array_equal(np.asarray(z), np.transpose(host, permutation) * 2)


def test_transpose_of_a_strided_host_view():
    # A transposed numpy view is borrowed with its strides; transposing it back
    # reads the caller's memory in its original order.
    device = jax.devices("tt")[0]
    host = np.random.default_rng(1).standard_normal((512, 256)).astype(jnp.bfloat16)
    x = jax.device_put(host.T, device)
    y = jnp.transpose(x, (1, 0))
    np.testing.assert_array_equal(np.asarray(y), host)


def test_transpose_of_sharded_host_data_stays_on_host():
    devices = jax.devices("tt")
    if len(devices) < 2:
        pytest.skip("needs two devices")
    mesh = jax.sharding.Mesh(
        np.array(devices[:2]).reshape(1, 2),
        ("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,
    )
    host = np.random.default_rng(2).standard_normal((512, 256)).astype(jnp.bfloat16)
    x = jax.device_put(host, NamedSharding(mesh, P("tensor", None)))
    with jax.sharding.set_mesh(mesh):
        y, before, after = transpose_and_device_bytes(x, (1, 0))
    assert after == before
    assert y.sharding.spec == P(None, "tensor")
    np.testing.assert_array_equal(np.asarray(y), host.T)


def test_transpose_of_device_data_runs_on_the_device():
    device = jax.devices("tt")[0]
    host = np.random.default_rng(3).standard_normal((128, 64)).astype(jnp.bfloat16)
    x = jax.jit(lambda a: a + 1)(jax.device_put(host, device))
    y = jnp.transpose(x, (1, 0))
    np.testing.assert_array_equal(np.asarray(y), np.asarray(x).T)
