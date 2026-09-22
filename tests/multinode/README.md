# Two-host Blackhole diagnostics

This experimental diagnostic uses JAX's coordinator for TT-Metal host metadata
and runs the upstream mesh-socket test over the cards' Ethernet links. It does
not require MPI. A separate PJRT test exercises JAX device enumeration,
sharded arrays, and device collectives.

The supplied topology describes one P150 on each host, connected by two Ethernet
channels (distinct from physical cables). Both cards must use matching firmware.
Each host needs its normal TT-Metal device access and hugepage setup, and
Python with JAX 0.11.2.

Build the diagnostic and run its CPU coordination tests:

```bash
bazel test -c opt //tests:key_value_context_test
bazel build -c opt //tests/multinode:fabric_probe.so
```

Copy the resulting shared library and this directory to both hosts. Also make
these two kernel files from the pinned TT-Metal sources available under the same
relative paths beneath `KERNEL_ROOT`:

- `tests/tt_metal/tt_metal/test_kernels/misc/socket/fabric_sender.cpp`
- `tests/tt_metal/tt_metal/test_kernels/misc/socket/fabric_receiver_worker.cpp`

Start the following on both hosts, setting `RANK` to 0 or 1 and `COORDINATOR` to
host 0's reachable IP and an unused TCP port. Use absolute paths for the file
arguments. `JAX_PLATFORMS=cpu` keeps PJRT device initialization out of the control
process; the C++ test opens and executes kernels on the Blackhole cards.

```bash
JAX_PLATFORMS=cpu TT_METAL_OPERATION_TIMEOUT_SECONDS=60 \
python discover.py \
  --coordinator "$COORDINATOR" --rank "$RANK" \
  --library "$LIBRARY" --output "$OUTPUT_YAML" \
  --test-config "$TEST_DIR/transfer.yaml" \
  --mesh-descriptor "$TEST_DIR/dual_blackhole.textproto" \
  --kernel-root "$KERNEL_ROOT"
```

Omit the last three arguments to run discovery alone. A successful discovery
can still report no connected Ethernet links; inspect `global_eth_connections`
in its YAML output. The payload test verifies changing 64 KiB and 256 KiB buffers
in both directions (80 transfers, 8 MiB total) and fails on any data mismatch.

## JAX/PJRT integration test

Build and install the same release wheel on both hosts:

```bash
bazel build -c opt //:jax_tt_plugin_wheel
pip install --force-reinstall --no-deps bazel-bin/jax_tt_plugin-*.whl
```

Run the following on each host, with `RANK=0` or `RANK=1` and the same reachable
`COORDINATOR` address. This topology uses one P150 per host in a single global
1-by-2 mesh. The test uses `jax.distributed.initialize()` and the TT backend;
no MPI launcher is required.

```bash
JAX_PLATFORMS=tt,cpu \
TT_MESH_ID=0 TT_MESH_HOST_RANK="$RANK" \
TT_MESH_GRAPH_DESC_PATH="$TEST_DIR/dual_blackhole_mesh.textproto" \
TT_METAL_OPERATION_TIMEOUT_SECONDS=60 \
python "$TEST_DIR/jax_distributed.py" \
  --coordinator "$COORDINATOR" --rank "$RANK"
```

Use `--enumeration-only` to stop after checking global device IDs and local
ownership. The full test checks a sharded pointwise operation, `psum`,
`all_gather`, and `psum_scatter` against NumPy, with distinct inputs on each
process and three iterations to exercise cached execution.

The full test passes on two hosts with one P150a each, JAX 0.11.2, and firmware
19.13.1. It uses TTNN device collectives over the cards' Ethernet connection;
the JAX coordinator carries host metadata. This remains experimental: more
than one local card, other topologies, process restarts, and explicit coordinator
shutdown before backend teardown are not validated.
