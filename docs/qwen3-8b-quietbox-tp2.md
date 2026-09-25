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

Validated with `pcmoritz/sglang-jax` commit
`6ca38400e6130bc242b215af52b8984bc859036e` (single-process TP path), JAX/jaxlib
0.11.1, Flax 0.12.9 and Transformers 4.57.6. Other SGLang-JAX revisions have
not been validated.

The TT backend uses `optimization_level="O1"`, BF8 weights, BF16 activations,
and traced decode. An older backend spelling of `optimization_level="1"`
needs updating in SGLang-JAX.

The same recipe runs four-chip tensor parallelism: list all four chips in
`TT_VISIBLE_DEVICES` and pass `--tp-size 4`. A two-chip mesh must be run with
exactly those two chips visible; a two-chip sub-mesh of a larger visible
system fails during fabric initialization.

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
and unfused projection widths. The local-mesh tests skip unless exactly two
chips are visible.

## Measured performance

Qwen3-8B, median of five 128-token generations after three warmups,
temperature zero, EOS ignored:

| Prompt | Input tokens | Median decode tokens/s | Median TTFT |
| --- | ---: | ---: | ---: |
| Short | 5 | 50.96 | 58.35 ms |
| Long | 199 | 49.05 | 78.41 ms |
