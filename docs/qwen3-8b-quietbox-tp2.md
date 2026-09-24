# Qwen3-8B on two local QuietBox chips

Run one SGLang-JAX process with a two-device JAX mesh. Weights and attention
heads are partitioned over the `tensor` axis; TT-Fabric handles the on-box
collectives. No JAX distributed initialization, MPI launcher, coordinator,
rank binding, or multi-host mesh descriptor is required.

## Build and launch

Build and install the plugin into the Python environment used by SGLang-JAX:

```bash
bazel build -c opt //:jax_tt_plugin_wheel
python -m pip install --force-reinstall --no-deps \
  bazel-bin/jax_tt_plugin-0.1.0-py3-none-linux_x86_64.whl
```

Select the two connected chips by PCI address. These are the addresses on the
validated QuietBox; substitute the addresses on your machine. Remove any
single-chip mesh descriptor from the environment so fabric discovery can see
both chips.

```bash
unset TT_MESH_GRAPH_DESC_PATH TT_METAL_RUNTIME_ROOT TT_RUNTIME_ENABLE_DISTRIBUTED
export TT_VISIBLE_DEVICES=0000:01:00.0,0000:04:00.0
export JAX_PLATFORMS=tt
export JAX_USE_SHARDY_PARTITIONER=true

python -m sgl_jax.launch_server \
  --model-path Qwen/Qwen3-8B \
  --host 127.0.0.1 --port 31000 \
  --device tt --dtype bfloat16 --attention-backend tt --tp-size 2 \
  --max-running-requests 2 --max-total-tokens 1024 \
  --max-prefill-tokens 256 --chunked-prefill-size 256 --page-size 32 \
  --watchdog-timeout 1200 --disable-precompile --skip-server-warmup \
  --disable-overlap-schedule --disable-radix-cache --stream-interval 1
```

These conservative cache and scheduling settings match validation. The first
request compiles the model; allow it to finish before measuring throughput.

```bash
curl --fail http://127.0.0.1:31000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"The capital of France is","sampling_params":{"temperature":0,"max_new_tokens":128,"ignore_eos":true}}'
```

## SGLang-JAX compatibility

Use a SGLang-JAX version with TT tensor-parallel attention and sampler support.
Validation uses the same unmodified integration revision as the preceding
TP2 experiments: `pcmoritz/sglang-jax` commit
`6ca38400e6130bc242b215af52b8984bc859036e`, with JAX/jaxlib 0.11.1, Flax 0.12.9
and Transformers 4.57.6. Despite that integration revision's multi-host commit
title, this recipe uses only its single-process TP path. This libtt branch
contains no multi-host JAX support or changes to SGLang-JAX. Arbitrary upstream
SGLang-JAX revisions have not been validated with this branch.

The TT backend uses `optimization_level="O1"`, BF8 weights, BF16 activations,
and traced decode. An older backend spelling of `optimization_level="1"`
needs updating in SGLang-JAX.

## Implementation scope

- Link upstream fabric discovery, routing, tensor partitioning and CCL instead
  of the single-chip stubs. The shared native fabric library needs its upstream
  descriptor schemas and topology solver even for an entirely local mesh.
- Give local PJRT devices unique IDs and preserve input/output shard metadata.
  A single-device executable must not inherit a previously opened TP mesh.
- Normalize Shardy meshes and collective axes, replicate indexed gather
  dimensions, and preserve layouts for operations without layout models.
- Preserve control-flow captures inside `shard_map`, and validate replicated
  scalar predicates before choosing a branch or replaying a trace.
- Discover links using the actual physical neighbor, handle single-page
  all-gather packet headers and large-transfer arithmetic, and cleanly stop
  fabric routers.
- Keep logical rotary head counts and support the 6,144-wide TP2 SwiGLU
  projection with full FP32 accumulation. Unsupported fusion shapes retain
  their generic implementation. Decode matmuls retain the proven eight-tile K
  blocks for narrow TP2 projections, with the existing divisibility and L1 checks.

There is no multi-process PJRT initialization, key-value transport, remote
host discovery, mesh-socket launcher, or multi-node test harness. The branch
also excludes the experimental fused TP2 all-reduce, 14B tail-tile support,
wide-reduction tuning and benchmark archives from the earlier branch.

## Regression checks

With both chips visible, run the tests together to also exercise a transition
from sharded execution to single-device execution in one process:

```bash
bazel test -c opt //tests:libtt_test_suite \
  --test_arg=local_mesh_regression_test.py \
  --test_arg=rotary_decode_regression_test.py \
  --test_arg=swiglu_regression_test.py --test_arg=-xq \
  --test_env=TT_VISIBLE_DEVICES=0000:01:00.0,0000:04:00.0 \
  --test_env=TT_METAL_OPERATION_TIMEOUT_SECONDS=30 \
  --test_timeout=300 --test_output=errors
```

Coverage includes inferred and explicit sharding, changing collective inputs,
trace replay, repartitioned weights, vocabulary lookup, nested control-flow
captures, all-gathers larger than 65,535 tiles, rotary head padding, and fused
and unfused projection widths.

## Validation on September 24, 2026

The branch is based directly on upstream libtt `f93eadb`. The release wheel
built successfully and all 30 combined local-mesh, rotary and SwiGLU
regression cases and all 17 single-chip JAX smoke tests passed. Test execution uses the repository's JAX 0.11.2
environment; model execution uses the serving versions listed above.

Qwen3-8B completed 21 requests: short and long prompts with three warmups and
five measured 128-token generations each, one warmup and one measured
concurrent pair, and a 32-token profiling request. Sampling used temperature
zero and ignored EOS. Outputs repeated within each measured prompt setting. All ten measured
response texts also matched the preceding tuned TP2 branch.

| Prompt | Input tokens | Median decode tokens/s | Median TTFT |
| --- | ---: | ---: | ---: |
| Short | 5 | 50.96 | 58.35 ms |
| Long | 199 | 49.05 | 78.41 ms |

This is a functional and performance smoke test, not a model-quality
evaluation. The retained release wheel SHA-256 is
`e23d7eb000f5f6d03abdc63a1f2c6e758b6409dd6cd921bdeb9f79c6c26a002b`.

Detailed local test and benchmark artifacts are in `/tmp/libtt-quietbox/` on
the validation machine; generated traces and prior experiment archives are
not part of this branch.
