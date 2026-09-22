# Two-host Blackhole fabric diagnostic

This experimental diagnostic uses JAX's coordinator for TT-Metal host metadata
and runs the upstream mesh-socket test over the cards' Ethernet links. It does
not require MPI. It is not yet wired into libtt's PJRT client: this does not enable
multi-process JAX arrays or collectives.

The supplied topology describes one P150 on each host, connected by two Ethernet
channels. Both cards must use matching firmware. Each host needs its normal
TT-Metal device access and hugepage setup, and Python with JAX 0.11.2.

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
