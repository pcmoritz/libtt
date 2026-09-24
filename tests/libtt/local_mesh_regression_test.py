"""Keep PJRT shards and replicated trace predicates valid on a local mesh."""

import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


@pytest.fixture
def local_mesh():
    devices = jax.local_devices(backend="tt")
    if len(devices) < 2:
        pytest.skip("Requires two local TT devices")
    # This integration uses Shardy; the general test runner defaults to GSPMD.
    previous = jax.config.jax_use_shardy_partitioner
    jax.config.update("jax_use_shardy_partitioner", True)
    try:
        yield Mesh(np.array(devices[:2]), ("tp",))
    finally:
        jax.config.update("jax_use_shardy_partitioner", previous)


@pytest.mark.parametrize("explicit", [False, True])
def test_local_mesh_outputs_remain_shard_local(local_mesh, explicit):
    sharding = NamedSharding(local_mesh, P("tp", None))
    source = np.arange(64 * 32, dtype=np.float32).reshape(64, 32)
    value = jax.device_put(source, sharding)

    def add_one(x):
        return x + 1

    if explicit:
        add_one = jax.shard_map(
            add_one,
            mesh=local_mesh,
            in_specs=P("tp", None),
            out_specs=P("tp", None),
            check_vma=False,
        )
    result = jax.jit(add_one)(value)
    np.testing.assert_array_equal(np.asarray(result), source + 1)
    for shard in result.addressable_shards:
        np.testing.assert_array_equal(np.asarray(shard.data), (source + 1)[shard.index])


@pytest.mark.parametrize("trace", [False, True])
def test_local_mesh_replicated_conditional(local_mesh, trace):
    sharding = NamedSharding(local_mesh, P("tp", None))
    replicated = NamedSharding(local_mesh, P())
    source = np.arange(64 * 32, dtype=np.float32).reshape(64, 32)
    value = jax.device_put(source, sharding)

    def choose(predicate, x):
        return jax.lax.cond(predicate, lambda v: v + 1, lambda v: v - 1, x)

    choose = jax.jit(
        choose,
        in_shardings=(replicated, sharding),
        out_shardings=sharding,
        compiler_options={"enable_trace": str(trace).lower(), "optimization_level": "O1"},
    )
    # Each predicate exercises warmup, capture, and replay, interleaved so that
    # reusing a trace for the wrong replicated scalar produces the wrong answer.
    for predicate in (True, False) * 4:
        flag = jax.device_put(np.asarray(predicate), replicated)
        result = choose(flag, value)
        np.testing.assert_array_equal(np.asarray(result), source + (1 if predicate else -1))


@pytest.mark.parametrize("width", [32, 4096, 8224])
@pytest.mark.parametrize("trace", [False, True])
def test_local_mesh_collective_sum(local_mesh, width, trace):
    """Different rank inputs must sum correctly, including after trace replay."""
    import jax.numpy as jnp

    sharding = NamedSharding(local_mesh, P("tp", None))
    reduce_sum = jax.shard_map(
        lambda x: jax.lax.psum(x, "tp"),
        mesh=local_mesh,
        in_specs=P("tp", None),
        out_specs=P(),
        check_vma=False,
    )
    reduce_sum = jax.jit(
        reduce_sum,
        compiler_options={"enable_trace": str(trace).lower(), "optimization_level": "O1"},
    )
    for step in range(5):
        source = ((np.arange(2 * width).reshape(2, width) + step) % 47 - 23).astype(
            jnp.bfloat16
        )
        result = reduce_sum(jax.device_put(source, sharding))
        expected = source.astype(np.float32).sum(axis=0, keepdims=True)
        np.testing.assert_array_equal(np.asarray(result, dtype=np.float32), expected)
        for shard in result.addressable_shards:
            np.testing.assert_array_equal(np.asarray(shard.data, dtype=np.float32), expected)
