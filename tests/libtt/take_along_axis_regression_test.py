"""Picking one element per row must not flatten the operand.

tt-mlir lowered x[arange(n), ids] by flattening x to [n * width, 1] for an
embedding lookup, which takes 64 bytes per element on device: selecting each
prompt token's logprob over a 248320-entry vocabulary needed an 8.1 GB buffer
and ran out of DRAM. take_along_axis instead lowered to ttnn.gather, which is
exact but took 2.4 s for 512 such rows. A single index per row now compares the
index with the positions along the row and sums the selected element.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("dtype", [np.float32, jnp.bfloat16])
@pytest.mark.parametrize("rows, width", [(512, 248320), (7, 1000), (64, 151936)])
def test_take_along_axis_one_index_per_row(rows, width, dtype):
    device = jax.devices("tt")[0]
    rng = np.random.default_rng(0)
    x = rng.standard_normal((rows, width)).astype(dtype)
    x[0, :] = -np.inf  # Logprobs of masked tokens.
    ids = rng.integers(0, width, rows).astype(np.int32)
    ids[:3] = [0, width - 1, 0][: min(3, rows)]
    pick = jax.jit(lambda x, ids: jnp.take_along_axis(x, ids[:, None], axis=1)[:, 0])
    out = np.asarray(pick(jax.device_put(x, device), jax.device_put(ids, device)))
    np.testing.assert_array_equal(out, x[np.arange(rows), ids])


def test_take_along_axis_one_index_per_row_is_fast():
    device = jax.devices("tt")[0]
    x = jax.device_put(np.zeros((512, 248320), np.float32), device)
    ids = jax.device_put(np.arange(512, dtype=np.int32), device)
    pick = jax.jit(lambda x, ids: jnp.take_along_axis(x, ids[:, None], axis=1)[:, 0])
    pick(x, ids).block_until_ready()
    start = time.perf_counter()
    pick(x, ids).block_until_ready()
    # About 11 ms; ttnn.gather took 2.4 s.
    assert time.perf_counter() - start < 0.5


def test_take_along_axis_batched_rows():
    device = jax.devices("tt")[0]
    rng = np.random.default_rng(1)
    x = rng.standard_normal((3, 64, 1000)).astype(np.float32)
    ids = rng.integers(0, 1000, (3, 64, 1)).astype(np.int32)
    pick = jax.jit(lambda x, ids: jnp.take_along_axis(x, ids, axis=2))
    out = np.asarray(pick(jax.device_put(x, device), jax.device_put(ids, device)))
    np.testing.assert_array_equal(out, np.take_along_axis(x, ids, axis=2))


def test_take_along_axis_several_indices_per_row():
    # More than one index per row still uses ttnn.gather.
    device = jax.devices("tt")[0]
    rng = np.random.default_rng(2)
    x = rng.standard_normal((32, 1000)).astype(np.float32)
    ids = rng.integers(0, 1000, (32, 4)).astype(np.int32)
    pick = jax.jit(lambda x, ids: jnp.take_along_axis(x, ids, axis=1))
    out = np.asarray(pick(jax.device_put(x, device), jax.device_put(ids, device)))
    np.testing.assert_array_equal(out, np.take_along_axis(x, ids, axis=1))
