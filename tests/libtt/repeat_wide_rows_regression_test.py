"""Repeating rows wider than L1 must copy them in chunks.

A row-major repeat used to stage each whole row in L1 twice, so broadcasting a
row of a 248320-entry vocabulary (Qwen3.5) failed to allocate its circular
buffers. A vocab-sharded argmax builds such a broadcast for its index offsets.
"""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P


@pytest.mark.parametrize("trace", [False, True])
@pytest.mark.parametrize(
    "dtype, width",
    [
        (np.int32, 248320),  # One 970 KiB row: 15 full 64 KiB chunks and a remainder.
        (np.int32, 100003),  # A row that is not a multiple of 16 bytes.
        (np.float32, 16384),  # Exactly one chunk.
        (jnp.bfloat16, 262147),
        (np.int32, 8),  # Short row: one chunk, smaller than the read alignment.
    ],
)
def test_broadcast_wide_row(dtype, width, trace):
    device = jax.devices("tt")[0]
    rows = 3
    x = (np.arange(width) % 251 - 125).astype(dtype)[None, :]
    run = jax.jit(
        lambda v: jnp.broadcast_to(v, (rows, width)),
        compiler_options={"enable_trace": str(trace).lower()},
    )
    out = np.asarray(run(jax.device_put(x, device)))
    np.testing.assert_array_equal(out, np.broadcast_to(x, (rows, width)))


@pytest.mark.parametrize("trace", [False, True])
@pytest.mark.parametrize("rows", [2, 3])
def test_vocab_sharded_argmax(rows, trace, tmp_path):
    devices = jax.devices("tt")
    if len(devices) < 2:
        pytest.skip("needs at least two devices")
    width = 248320
    shard = width // len(devices)
    mesh = jax.make_mesh((len(devices),), ("tensor",), devices=devices)
    x = np.random.default_rng(0).standard_normal((rows, width)).astype(np.float32)
    x[0, -1] = 10  # The maximum of a row in the last shard.
    # Equal maxima: the first one wins, across shards (row 0), across groups
    # of 32 in one shard (row 1) and within one group (row 2).
    x[0, [shard - 1, shard]] = 11
    x[1, [shard + 40, shard + 4000]] = 11
    if rows > 2:
        x[2, [shard + 67, shard + 73]] = 11
    run = jax.jit(
        lambda v: jnp.argmax(v, axis=-1),
        out_shardings=NamedSharding(mesh, P()),
        compiler_options={"enable_trace": str(trace).lower(), "export_path": str(tmp_path)},
    )
    out = np.asarray(run(jax.device_put(x, NamedSharding(mesh, P(None, "tensor")))))
    np.testing.assert_array_equal(out, x.argmax(-1))

    # Each shard picks its maximum's value with a sum over the shard, which
    # tt-mlir splits into groups of 32 (wide_row_reduction_regression_test).
    irs = [path.read_text() for path in (tmp_path / "irs").glob("ttnn_runtime_*.mlir")]
    assert irs, "no IR was exported"
    sums = re.findall(r'"ttnn\.sum"\(%\w+\).*?: \(tensor<([0-9x]+)x[a-z]', "\n".join(irs))
    assert f"{rows}x{shard // 32}x32" in sums, sums
    assert f"{rows}x{shard // 32}" in sums, sums
