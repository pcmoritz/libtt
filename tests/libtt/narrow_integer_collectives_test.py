"""Collectives and sums of 8-, 16- and 32-bit unsigned integers.

TT-NN's sum (moreh_sum) takes BFLOAT16 and INT32 only, so a psum of uint8 or
uint16 failed ("moreh_sum input only supports specific data types"), and
tt-mlir summed other integers through Float32, so a sum of uint16 saturated
at 65535 instead of wrapping. torch.distributed's object collectives send
pickled bytes as uint8 tensors, and torch_tpu's broadcast sums them. Both now
sum on INT32 and cast back, which wraps as the narrow type would.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

DTYPES = [np.int8, np.uint8, np.int16, np.uint16, np.uint32]


def mesh():
    devices = jax.devices("tt")
    if len(devices) < 2:
        pytest.skip("needs at least two devices")
    return Mesh(np.array(devices), ("x",))


def values(dtype, shape, seed):
    info = np.iinfo(dtype)
    return np.random.default_rng(seed).integers(info.min, info.max, shape, endpoint=True, dtype=dtype)


def sharded(m, x):
    return jax.device_put(x, NamedSharding(m, P("x")))


@pytest.mark.parametrize("dtype", DTYPES)
def test_all_gather(dtype):
    m = mesh()
    x = values(dtype, (len(m.devices) * 32,), 0)
    gather = jax.shard_map(
        lambda b: jax.lax.all_gather(b, "x", tiled=True),
        mesh=m,
        in_specs=P("x"),
        out_specs=P(),
        check_vma=False,
    )
    out = jax.jit(gather)(sharded(m, x))
    assert out.dtype == dtype
    np.testing.assert_array_equal(np.asarray(out), x)


@pytest.mark.parametrize("dtype", DTYPES)
def test_psum_wraps(dtype):
    m = mesh()
    n = len(m.devices)
    x = values(dtype, (n, 32), 1)
    total = jax.shard_map(
        lambda b: jax.lax.psum(b, "x"), mesh=m, in_specs=P("x"), out_specs=P(), check_vma=False
    )
    out = jax.jit(total)(sharded(m, x))
    assert out.dtype == dtype
    np.testing.assert_array_equal(np.asarray(out)[0], x.sum(axis=0, dtype=dtype))


@pytest.mark.parametrize("dtype", DTYPES)
def test_sum_wraps(dtype):
    x = values(dtype, (8, 64), 2)
    out = jax.jit(lambda v: jnp.sum(v, axis=1, dtype=dtype))(x)
    assert out.dtype == dtype
    np.testing.assert_array_equal(np.asarray(out), x.sum(axis=1, dtype=dtype))


@pytest.mark.parametrize("dtype", [np.int8, np.uint8])
def test_max_keeps_its_type(dtype):
    """Reductions that compare values are left alone."""
    x = values(dtype, (8, 64), 3)
    out = jax.jit(lambda v: jnp.max(v, axis=1))(x)
    np.testing.assert_array_equal(np.asarray(out), x.max(axis=1))
