"""Exercise decode-sized all-reduce across widths, collectives and trace inputs.

The mesh spans every visible device, so run once with two and once with four
chips visible to cover both exchange/reduce ring sizes.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


@pytest.fixture
def full_mesh():
    devices = jax.local_devices(backend="tt")
    if len(devices) not in (2, 4):
        pytest.skip("Requires two or four local TT devices")
    previous = jax.config.jax_use_shardy_partitioner
    jax.config.update("jax_use_shardy_partitioner", True)
    try:
        yield Mesh(np.array(devices), ("tp",))
    finally:
        jax.config.update("jax_use_shardy_partitioner", previous)


@pytest.mark.parametrize("trace", [False, True])
@pytest.mark.parametrize(
    "rows,width,dtype",
    [
        # One tile, an odd tile count, a typical hidden size, and a prime
        # tile count (257); several and all 32 rows of the tile.
        (1, 32, jnp.bfloat16),
        (1, 96, jnp.bfloat16),
        (1, 5120, jnp.bfloat16),
        (1, 8224, jnp.bfloat16),
        (2, 5120, jnp.bfloat16),
        (32, 5120, jnp.bfloat16),
        # Falls back to reduce-scatter plus all-gather.
        (1, 5120, jnp.float32),
    ],
)
def test_all_reduce(full_mesh, trace, rows, width, dtype):
    ranks = full_mesh.size
    sharding = NamedSharding(full_mesh, P("tp", None))

    def sums(x, y):
        return jax.lax.psum(x, "tp"), jax.lax.psum(y, "tp")

    run = jax.jit(
        jax.shard_map(
            sums,
            mesh=full_mesh,
            in_specs=(P("tp", None), P("tp", None)),
            out_specs=(P(), P()),
            check_vma=False,
        ),
        compiler_options={"optimization_level": "O1", "enable_trace": str(trace).lower()},
    )
    rng = np.random.default_rng(324)
    for _ in range(8):
        # Every partial sum of up to four ranks has at most eight significant
        # bits, so the result is exact in BF16 for any reduction order.
        x = (rng.integers(-64, 64, (ranks * rows, width)) / 8).astype(dtype)
        y = (rng.integers(-64, 64, (ranks * rows, width)) / 32).astype(dtype)
        outputs = run(jax.device_put(x, sharding), jax.device_put(y, sharding))
        for source, actual in zip((x, y), outputs):
            expected = (
                source.astype(np.float32).reshape(ranks, rows, width).sum(axis=0)
                .astype(dtype).astype(np.float32)
            )
            np.testing.assert_array_equal(np.asarray(actual, dtype=np.float32), expected)
            for shard in actual.addressable_shards:
                np.testing.assert_array_equal(np.asarray(shard.data, dtype=np.float32), expected)


@pytest.mark.parametrize("trace", [False, True])
def test_leading_dimension_padding_falls_back(full_mesh, trace):
    # Two logical rows in separate leading batches occupy two padded tile rows.
    # They must not use the single-tile-row decode specialization.
    ranks = full_mesh.size
    spec = P("tp", None, None)
    run = jax.jit(
        jax.shard_map(
            lambda x: jax.lax.psum(x, "tp"),
            mesh=full_mesh,
            in_specs=spec,
            out_specs=P(),
            check_vma=False,
        ),
        compiler_options={"optimization_level": "O1", "enable_trace": str(trace).lower()},
    )
    rng = np.random.default_rng(567)
    for _ in range(4):
        x = (rng.integers(-64, 64, (2 * ranks, 1, 5120)) / 8).astype(jnp.bfloat16)
        actual = run(jax.device_put(x, NamedSharding(full_mesh, spec)))
        expected = x.astype(np.float32).reshape(ranks, 2, 1, 5120).sum(axis=0).astype(jnp.bfloat16)
        np.testing.assert_array_equal(np.asarray(actual), expected)
