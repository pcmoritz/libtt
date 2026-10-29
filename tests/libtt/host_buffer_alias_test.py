"""Contiguous host arrays of up to 32 MiB divided by the number of chips are
uploaded from the caller's memory, without first copying them into a new
buffer. Larger ones are still gathered when read, so that no write of over
32 MiB, which tt-metal would pin, reads from the caller's memory.
"""

import numpy as np
import pytest

import jax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


def limit():
    return (32 << 20) // len(jax.devices("tt"))


@pytest.mark.parametrize("nbytes", [(1 << 20) + 4096, "limit", "over"])
@pytest.mark.parametrize("sharding", ["one", "replicated", "sharded"])
def test_contiguous_upload(nbytes, sharding):
    devices = jax.devices("tt")
    nbytes = {"limit": limit(), "over": limit() + 4096}.get(nbytes, nbytes)
    x = (np.arange(nbytes // 4) % 1009).astype(np.float32).reshape(-1, 1024)
    mesh = Mesh(np.array(devices), ("x",))
    target = {
        "one": devices[0],
        "replicated": NamedSharding(mesh, P()),
        "sharded": NamedSharding(mesh, P(None, "x")),
    }[sharding]
    y = jax.device_put(x, target)
    np.testing.assert_array_equal(np.asarray(jax.jit(lambda a: a * 2)(y)), x * 2)


def test_mapped_file_replicated_to_every_chip(tmp_path):
    """A copy-on-write file mapping at the limit, written to every chip in one
    write of at most 32 MiB, which tt-metal copies rather than pins."""
    shape = (limit() // 4096, 1024)
    path = tmp_path / "values.bin"
    values = np.memmap(path, np.float32, mode="w+", shape=shape)
    values[:] = (np.arange(np.prod(shape)) % 1009).astype(np.float32).reshape(shape)
    values.flush()
    del values
    mapped = np.memmap(path, np.float32, mode="c", shape=shape)
    mesh = Mesh(np.array(jax.devices("tt")), ("x",))
    y = jax.device_put(mapped, NamedSharding(mesh, P()))
    np.testing.assert_array_equal(np.asarray(jax.jit(lambda a: a + 1)(y)), np.asarray(mapped) + 1)
