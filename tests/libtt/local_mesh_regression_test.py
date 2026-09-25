"""Keep PJRT shards and replicated trace predicates valid on a local mesh."""

import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


@pytest.fixture
def local_mesh():
    devices = jax.local_devices(backend="tt")
    # A two-chip sub-mesh of a larger visible system is not supported yet.
    if len(devices) != 2:
        pytest.skip("Requires exactly two visible TT devices")
    # This integration uses Shardy; the general test runner defaults to GSPMD.
    previous = jax.config.jax_use_shardy_partitioner
    jax.config.update("jax_use_shardy_partitioner", True)
    try:
        yield Mesh(np.array(devices), ("tp",))
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


@pytest.mark.parametrize("spec", [P("tp", None), P(None, "tp")])
def test_repartition_replicated_weights(local_mesh, spec):
    replicated = NamedSharding(local_mesh, P())
    partitioned = NamedSharding(local_mesh, spec)
    for offset in (0, 100):
        source = np.arange(64 * 64, dtype=np.float32).reshape(64, 64) + offset
        result = jax.device_put(jax.device_put(source, replicated), partitioned)
        for shard in result.addressable_shards:
            np.testing.assert_array_equal(np.asarray(shard.data), source[shard.index])


def test_explicit_mesh_gather_and_control_flow(local_mesh):
    mesh = Mesh(
        local_mesh.devices.reshape(1, 2), ("data", "tp"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,
    )
    with jax.set_mesh(mesh):
        spec = P("data", "tp")
        sharding = NamedSharding(mesh, spec)
        replicated = NamedSharding(mesh, P())
        source = (np.arange(32 * 64).reshape(32, 64) % 17).astype(np.float32)
        x = jax.device_put(source, sharding)
        delta = jax.device_put(np.ones_like(source), sharding)

        def captured(predicate, x, delta):
            value = x + x
            selected = jax.lax.cond(predicate, lambda: value + delta, lambda: value)
            return jax.lax.fori_loop(0, 2, lambda i, y: y + delta, selected)

        run = jax.jit(
            jax.shard_map(captured, mesh=mesh, in_specs=(P(), spec, spec),
                          out_specs=spec, check_vma=False),
            compiler_options={"optimization_level": "O1", "enable_trace": "true"},
        )
        for flag in (False, True) * 3:
            actual = run(jax.device_put(np.array(flag), replicated), x, delta)
            np.testing.assert_array_equal(np.asarray(actual), 2 * source + int(flag) + 2)

        # Indexed vocabulary dimensions must be replicated before gather;
        # indices cross both halves of the sharded table.
        table = (np.arange(64 * 32).reshape(64, 32) % 19).astype(np.float32)
        indices = np.array([63, 0, 32, 31], dtype=np.int32)
        table_sharding = NamedSharding(mesh, P("tp", None))
        lookup = jax.jit(lambda t, i: t.at[i].get(out_sharding=replicated))
        actual = lookup(jax.device_put(table, table_sharding),
                        jax.device_put(indices, replicated))
        np.testing.assert_array_equal(np.asarray(actual), table[indices])


def test_large_all_gather_slice_bounds(local_mesh):
    import jax.numpy as jnp

    # Each input shard has more than 65535 tiles, exposing overflow in the
    # original all-gather page-count multiplication used by large weights.
    shape = (2 * 16416, 4096)
    sharding = NamedSharding(local_mesh, P("tp", None))

    def values(index):
        rows, cols = index
        r = np.arange(*rows.indices(shape[0]), dtype=np.int32)[:, None]
        c = np.arange(*cols.indices(shape[1]), dtype=np.int32)[None, :]
        return ((r % 127 + c % 127) / 128).astype(jnp.bfloat16)

    gather = jax.jit(jax.shard_map(
        lambda x: jax.lax.all_gather(x, "tp", axis=0, tiled=True),
        mesh=local_mesh, in_specs=P("tp", None), out_specs=P(), check_vma=False,
    ))
    result = gather(jax.make_array_from_callback(shape, sharding, values))
    for shard in result.addressable_shards:
        actual = np.asarray(shard.data)
        for start in range(0, shape[0], 256):
            index = (slice(start, min(start + 256, shape[0])), slice(None))
            np.testing.assert_array_equal(actual[index], values(index))
