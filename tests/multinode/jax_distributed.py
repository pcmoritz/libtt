"""Two-process JAX/PJRT smoke test; run once on each Ethernet-connected host."""

import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinator", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--enumeration-only", action="store_true")
    args = parser.parse_args()
    jax.distributed.initialize(
        coordinator_address=args.coordinator,
        num_processes=2,
        process_id=args.rank,
        initialization_timeout=120,
    )
    devices = jax.devices("tt")
    local = jax.local_devices(backend="tt")
    assert jax.process_index("tt") == args.rank
    assert jax.process_count("tt") == 2
    assert len(devices) == 2 and len(local) == 1
    assert sorted(d.process_index for d in devices) == [0, 1]
    assert local[0].process_index == args.rank
    print(
        json.dumps(
            {
                "rank": args.rank,
                "devices": [
                    {"id": d.id, "process": d.process_index, "platform": d.platform}
                    for d in devices
                ],
                "local_devices": [d.id for d in local],
            }
        ),
        flush=True,
    )
    print("PASS: multi-process TT device enumeration", flush=True)
    if args.enumeration_only:
        return

    local_step = jax.jit(lambda x: x * 2 + 1)

    def check_local_jit():
        # Unequal call counts catch accidental rendezvous for local execution.
        expected = np.full((32, 32), args.rank, dtype=np.float32)
        value = jax.device_put(expected, local[0])
        for _ in range(args.rank + 1):
            value = local_step(value)
            expected = expected * 2 + 1
        assert value.devices() == {local[0]}
        np.testing.assert_array_equal(np.asarray(value), expected)
        print("PASS: independent local JIT", flush=True)

    check_local_jit()

    mesh = Mesh(np.array(devices), ("x",))
    sharding = NamedSharding(mesh, P("x", None))

    def compile_local(fn, out_spec):
        return jax.jit(
            jax.shard_map(
                fn,
                mesh=mesh,
                in_specs=P("x", None),
                out_specs=out_spec,
                check_vma=False,
            )
        )

    # Eager shard_map asks PJRT to infer how to split an unsharded input.
    eager_psum = jax.shard_map(
        lambda x: lax.psum(x, "x"),
        mesh=mesh,
        in_specs=P("x", None),
        out_specs=P(),
        check_vma=False,
    )
    full_input = np.arange(128 * 32, dtype=np.float32).reshape(128, 32) % 17
    result = eager_psum(full_input)
    np.testing.assert_array_equal(
        np.asarray(result.addressable_shards[0].data),
        full_input[:64] + full_input[64:],
    )
    print("PASS: inferred input sharding", flush=True)

    pointwise = compile_local(lambda x: x * 2 + 1, P("x", None))
    psum = compile_local(lambda x: lax.psum(x, "x"), P())
    gather = compile_local(
        lambda x: lax.all_gather(x, "x", axis=0, tiled=True),
        P(),
    )
    scatter = compile_local(
        lambda x: lax.psum_scatter(x, "x", scatter_dimension=0, tiled=True),
        P("x", None),
    )

    # Exercise both the tiled path and wide rows eligible for direct all-gather.
    for iteration, width in enumerate((32, 1024, 1024)):
        # Different data on every host and iteration catches accidental
        # replication and stale cached collective outputs.
        shards = [
            np.arange(64 * width, dtype=np.float32).reshape(64, width) % 17
            + 20 * (rank + iteration)
            for rank in range(2)
        ]
        # Exercise local-to-global mesh transfer with a computed device array.
        local_input = local_step(jax.device_put(shards[args.rank], local[0]))
        shards = [shard * 2 + 1 for shard in shards]
        x = jax.make_array_from_process_local_data(sharding, local_input)
        cases = [
            ("pointwise", pointwise, shards[args.rank] * 2 + 1),
            ("psum", psum, shards[0] + shards[1]),
            ("all_gather", gather, np.concatenate(shards, axis=0)),
            (
                "psum_scatter",
                scatter,
                (shards[0] + shards[1])[args.rank * 32 : (args.rank + 1) * 32],
            ),
        ]
        for name, fn, expected in cases:
            print(f"RUN: {name}, iteration {iteration}", flush=True)
            result = fn(x)
            assert len(result.addressable_shards) == 1
            np.testing.assert_allclose(
                np.asarray(result.addressable_shards[0].data),
                expected,
                rtol=1e-5,
                atol=1e-5,
            )
            print(f"PASS: {name}, iteration {iteration}", flush=True)
        # The local shard of a collective result must work in a local JIT too.
        np.testing.assert_array_equal(
            np.asarray(local_step(result.addressable_shards[0].data)),
            expected * 2 + 1,
        )
    # More than 65535 input tiles exercises all-gather's slice arithmetic
    # beyond the old uint32 page-count multiplication limit.
    shape = (2 * 16416, 4096)

    def large_values(index):
        rows, cols = index
        r = np.arange(*rows.indices(shape[0]), dtype=np.int32)[:, None]
        c = np.arange(*cols.indices(shape[1]), dtype=np.int32)[None, :]
        return ((r % 127 + c % 127) / 128).astype(jnp.bfloat16)

    large_input = jax.make_array_from_callback(shape, sharding, large_values)
    result = gather(large_input)
    actual = np.asarray(result.addressable_shards[0].data)
    for start in range(0, shape[0], 256):
        index = (slice(start, min(start + 256, shape[0])), slice(None))
        np.testing.assert_array_equal(actual[index], large_values(index))
    print("PASS: large all-gather slice bounds", flush=True)
    # Partition replicated weights on-device, as model loaders do after a
    # transpose. Changing the values also checks cached slice arguments.
    for spec in (P("x", None), P(None, "x")):
        target = NamedSharding(mesh, spec)
        for offset in (0, 100):
            full = np.arange(64 * 64, dtype=np.float32).reshape(64, 64) + offset
            replicated = jax.make_array_from_callback(
                full.shape, NamedSharding(mesh, P()), lambda index: full[index]
            )
            result = jax.device_put(replicated, target)
            shard = result.addressable_shards[0]
            np.testing.assert_array_equal(np.asarray(shard.data), full[shard.index])
    print("PASS: on-device weight partitioning", flush=True)
    # A size-one data axis must not create a reduction when gathering rows.
    explicit_mesh = Mesh(
        np.array(devices).reshape(1, 2),
        ("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,
    )
    with jax.set_mesh(explicit_mesh):
        # Inlining a manual sharding region must preserve captured arguments
        # when outlining nested conditionals and loops.
        capture_sharding = NamedSharding(explicit_mesh, P("data", "tensor"))
        full = (np.arange(32 * 64).reshape(32, 64) % 17).astype(np.float32)
        x = jax.make_array_from_callback(full.shape, capture_sharding, lambda i: full[i])
        delta = jax.make_array_from_callback(
            full.shape, capture_sharding, lambda i: np.ones(full.shape, np.float32)[i]
        )

        def captured_control_flow(predicate, x, delta):
            value = x + x
            selected = lax.cond(predicate, lambda: value + delta, lambda: value)
            looped = lax.fori_loop(0, 2, lambda i, y: y + delta, selected)
            return selected, looped

        captured_control_flow = jax.jit(
            jax.shard_map(
                captured_control_flow,
                mesh=explicit_mesh,
                in_specs=(P(), P("data", "tensor"), P("data", "tensor")),
                out_specs=(P("data", "tensor"), P("data", "tensor")),
                check_vma=False,
            )
        )
        for flag in (False, True, False):
            predicate = jax.make_array_from_callback(
                (), NamedSharding(explicit_mesh, P()), lambda i: np.array(flag)
            )
            for result, extra in zip(captured_control_flow(predicate, x, delta), (0, 2)):
                shard = result.addressable_shards[0]
                np.testing.assert_array_equal(
                    np.asarray(shard.data), (2 * full + int(flag) + extra)[shard.index]
                )
        print("PASS: sharded control-flow captures", flush=True)
        full = (np.arange(64 * 8 * 32).reshape(64, 8, 32) % 13).astype(np.float32)
        indices = np.arange(63, -1, -1, dtype=np.int32).reshape(2, 32)
        values = jax.make_array_from_callback(
            full.shape,
            NamedSharding(explicit_mesh, P("data", "tensor", None)),
            lambda index: full[index],
        )
        rows = jax.make_array_from_callback(
            indices.shape,
            NamedSharding(explicit_mesh, P("data", None)),
            lambda index: indices[index],
        )
        output_sharding = NamedSharding(explicit_mesh, P("data", None, "tensor", None))
        result = jax.jit(lambda x, i: x.at[i].get(out_sharding=output_sharding))(values, rows)
        shard = result.addressable_shards[0]
        np.testing.assert_array_equal(np.asarray(shard.data), full[indices][shard.index])
        # Indexed dimensions need global rows, not an unmasked sum of local
        # lookups. Exercise tokens on both sides of the vocabulary boundary.
        output_sharding = NamedSharding(explicit_mesh, P("data", None))
        embedding = jax.jit(
            lambda w, i: w.at[i].get(out_sharding=output_sharding),
            compiler_options={"optimization_level": "O1", "enable_trace": "true"},
        )
        full = (np.arange(32 * 64).reshape(32, 64) % 127).astype(jnp.bfloat16)
        weights = jax.make_array_from_callback(
            full.shape, NamedSharding(explicit_mesh, P("tensor", None)), lambda i: full[i]
        )
        for rows in ([0, 15, 16, 31], [31, 16, 15, 0], [1, 17, 2, 18]):
            indices = np.array(rows, dtype=np.int32)
            ids = jax.make_array_from_callback(
                indices.shape, NamedSharding(explicit_mesh, P("data")), lambda i: indices[i]
            )
            result = embedding(weights, ids)
            np.testing.assert_array_equal(
                np.asarray(result.addressable_shards[0].data), full[indices]
            )
    print("PASS: gather with size-one mesh axis", flush=True)
    print("PASS: vocabulary-sharded embedding", flush=True)
    check_local_jit()
    print("PASS: multi-process TT arrays and collectives", flush=True)


if __name__ == "__main__":
    main()
