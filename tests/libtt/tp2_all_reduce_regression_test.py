"""Exercise two-rank sums, multiple collectives, and changing trace inputs."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P
from local_mesh_regression_test import local_mesh


@pytest.mark.parametrize('trace', [False, True])
@pytest.mark.parametrize('rows,width,dtype', [
    (1, 1024, jnp.bfloat16), (1, 4096, jnp.bfloat16),
    (1, 5120, jnp.bfloat16), (2, 5120, jnp.bfloat16),
    (1, 8192, jnp.bfloat16), (1, 8224, jnp.bfloat16),
    (32, 5120, jnp.bfloat16), (1, 5120, jnp.float32),
])
def test_two_rank_all_reduce(local_mesh, trace, rows, width, dtype):
    sharding = NamedSharding(local_mesh, P('tp', None))
    def sums(x, y):
        return jax.lax.psum(x, 'tp'), jax.lax.psum(y, 'tp')
    run = jax.jit(
        jax.shard_map(sums, mesh=local_mesh, in_specs=(P('tp', None), P('tp', None)),
                      out_specs=(P(), P()), check_vma=False),
        compiler_options={'optimization_level': 'O1', 'enable_trace': str(trace).lower()},
    )
    rng = np.random.default_rng(324)
    for step in range(8):
        x = (rng.integers(-128, 128, (2 * rows, width)) / 16).astype(dtype)
        y = (rng.integers(-64, 64, (2 * rows, width)) / 32).astype(dtype)
        for source, actual in zip((x, y), run(jax.device_put(x, sharding), jax.device_put(y, sharding))):
            expected = source.astype(np.float32).reshape(2, rows, width).sum(axis=0).astype(dtype).astype(np.float32)
            np.testing.assert_array_equal(np.asarray(actual, dtype=np.float32), expected)
            for shard in actual.addressable_shards:
                np.testing.assert_array_equal(np.asarray(shard.data, dtype=np.float32), expected)


@pytest.mark.parametrize('trace', [False, True])
def test_leading_dimension_padding_falls_back(local_mesh, trace):
    # Two logical rows in separate leading batches occupy two padded tile rows.
    # They must not use the single-tile-row decode specialization.
    spec = P('tp', None, None)
    run = jax.jit(
        jax.shard_map(lambda x: jax.lax.psum(x, 'tp'), mesh=local_mesh,
                      in_specs=spec, out_specs=P(), check_vma=False),
        compiler_options={'optimization_level': 'O1', 'enable_trace': str(trace).lower()},
    )
    rng = np.random.default_rng(567)
    for _ in range(4):
        x = (rng.integers(-128, 128, (4, 1, 5120)) / 16).astype(jnp.bfloat16)
        actual = run(jax.device_put(x, NamedSharding(local_mesh, spec)))
        expected = x.astype(np.float32).reshape(2, 2, 1, 5120).sum(axis=0).astype(jnp.bfloat16)
        np.testing.assert_array_equal(np.asarray(actual), expected)
