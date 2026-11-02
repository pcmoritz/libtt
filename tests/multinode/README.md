# Multi-process JAX on Blackhole

`jax_distributed.py` runs JAX across two processes with one Blackhole chip
each, on two hosts or on two chips of one host. The processes coordinate
through JAX's coordinator (`jax.distributed.initialize()`), which carries only
host metadata; tensors move over the chips' Ethernet links with TTNN's
collectives. No MPI launcher is needed.

Build the plugin wheel and install it, with JAX 0.11.2, where the processes
run:

```bash
bazel build -c opt //:jax_tt_plugin_wheel
pip install --force-reinstall --no-deps bazel-bin/jax_tt_plugin-*.whl
```

`dual_blackhole_mesh.textproto` describes the two chips as one 1-by-2 mesh.
Start one process per chip, with `RANK` 0 and 1 and the same `COORDINATOR`, a
host and unused port that both processes reach:

```bash
JAX_PLATFORMS=tt,cpu \
TT_MESH_ID=0 TT_MESH_HOST_RANK="$RANK" \
TT_MESH_GRAPH_DESC_PATH="$TEST_DIR/dual_blackhole_mesh.textproto" \
TT_METAL_OPERATION_TIMEOUT_SECONDS=60 \
python "$TEST_DIR/jax_distributed.py" \
  --coordinator "$COORDINATOR" --rank "$RANK"
```

On one host, also give each process its chip, such as
`TT_VISIBLE_DEVICES=0000:01:00.0` and `TT_VISIBLE_DEVICES=0000:04:00.0` for two
Ethernet-connected chips of a QuietBox.

Use `--enumeration-only` to stop after checking global device IDs and local
ownership. The full test checks independent local JIT execution, a sharded
pointwise operation, `psum`, `all_gather`, and `psum_scatter` against NumPy. It
uses distinct inputs on each process, checks inferred input sharding, passes
device arrays between local and distributed computations, and runs three
iterations to exercise cached execution. Additional cases cover on-device weight
partitioning, captured control flow, vocabulary-sharded embedding lookup, and
all-gathers exceeding 65,535 input tiles.

It passes on two hosts with one P150a each and on two chips of one QuietBox.
This remains experimental: more than one chip per process and other topologies
are not validated. A process that is killed can leave the chips' Ethernet links
down; `tt-smi -r` resets them.
