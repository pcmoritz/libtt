"""Two-process JAX/PJRT smoke test; run once on each Ethernet-connected host."""

import argparse
import json

import jax
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

    for iteration in range(3):
        # Different data on every host and iteration catches accidental
        # replication and stale cached collective outputs.
        shards = [
            np.arange(64 * 32, dtype=np.float32).reshape(64, 32) % 17
            + 20 * (rank + iteration)
            for rank in range(2)
        ]
        x = jax.make_array_from_process_local_data(sharding, shards[args.rank])
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
    print("PASS: multi-process TT arrays and collectives", flush=True)


if __name__ == "__main__":
    main()
