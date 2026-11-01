"""Reading device results back gives their exact values.

Results are untilized on the device before they are read, as on other
architectures, instead of being read tile by tile and untilized on the host.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


def _values(dtype, shape, rng):
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return rng.integers(info.min, info.max, size=shape, dtype=dtype, endpoint=True)
    x = rng.standard_normal(shape).astype(dtype)
    flat = x.reshape(-1)
    special = np.array([np.inf, -np.inf, 0.0, -0.0, 1e30, -1e-30], dtype)
    flat[: min(flat.size, special.size)] = special[: flat.size]
    return x


@pytest.mark.parametrize("dtype", [np.float32, jnp.bfloat16, np.float16, np.int32, np.uint32, np.uint16, np.uint8])
@pytest.mark.parametrize("shape", [(1,), (7, 1000), (3, 5, 33), (6, 248320)])
def test_readback_is_exact(dtype, shape):
    dtype = np.dtype(dtype)
    x = _values(dtype, shape, np.random.default_rng(0))
    # A transpose moves the data on the device, so the result is a device tensor.
    out = jax.jit(lambda a: jnp.swapaxes(a, 0, -1))(jax.device_put(x, jax.devices("tt")[0]))
    np.testing.assert_array_equal(np.asarray(out), np.swapaxes(x, 0, -1))


def test_sharded_and_replicated_readback_is_exact():
    devices = jax.devices("tt")
    mesh = Mesh(np.array(devices), ("x",))
    x = np.random.default_rng(1).standard_normal((6, 2048 * len(devices))).astype(np.float32)
    for spec in (P(None, "x"), P()):
        out = jax.jit(lambda a: a * 1.0, out_shardings=NamedSharding(mesh, spec))(x)
        np.testing.assert_array_equal(np.asarray(out), x)
        for shard in out.addressable_shards:
            np.testing.assert_array_equal(np.asarray(shard.data), x[shard.index])


def test_a_later_shard_read_first():
    """Reading a shard moves the whole result to the host and gives the other
    shards host tensors of their own: a read of a later shard first gets its
    rows, not the first shard's."""
    devices = jax.devices("tt")
    if len(devices) < 2:
        pytest.skip("needs at least two devices")
    mesh = Mesh(np.array(devices), ("x",))
    rows = NamedSharding(mesh, P("x"))
    x = np.arange(len(devices) * 4 * 32, dtype=np.float32).reshape(len(devices) * 4, 32)
    out = jax.jit(lambda a: a + 1, out_shardings=rows)(jax.device_put(x, rows))
    np.testing.assert_array_equal(np.asarray(out.addressable_shards[1].data), x[4:8] + 1)
    np.testing.assert_array_equal(np.asarray(out), x + 1)
