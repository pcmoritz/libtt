# libtt

`libtt.so` is a Bazel-built PJRT plugin for Tenstorrent devices. The PJRT
implementation comes from the pinned `tt-xla` repository, with `tt-mlir` and
`tt-metal` built through Bazel overlays in this repository. Everything needed
to run Jax code (including the tt-metal runtime and compiler) is bundled into
the `libtt.so` file. We also apply patches so sglang-jax works out of the box.

The local code in this repository is intentionally small:

- it materializes the embedded TT-Metal runtime archive before the plugin starts;
- it links the upstream `tt-xla` PJRT plugin into the final shared library;
- it hides internal symbols so the shared object only exports the PJRT entrypoints.

## Design documents

- [General graph fusion for TT-MLIR and TTNN](docs/design/general_fusion.md)

## Build

```bash
bazel build //:tt
```

The output is `bazel-bin/libtt.so`.

## Qwen3 With SGLang-JAX

Build `libtt.so` first:

```bash
cd /path/to/libtt
bazel build //:tt
export LIBTT_DIR="$PWD"
```

Then check out the SGLang-JAX TT backend branch from
[pcmoritz/sglang-jax#1](https://github.com/pcmoritz/sglang-jax/pull/1):

```bash
export SGLANG_JAX_DIR="$HOME/sglang-jax"
git clone git@github.com:pcmoritz/sglang-jax.git "$SGLANG_JAX_DIR"
cd "$SGLANG_JAX_DIR"
git fetch origin pull/1/head:codex/qwen3-tt-sglang
git switch codex/qwen3-tt-sglang
```

Install or activate the Python environment for that checkout, then start a
Qwen3-8B server with the TT backend:

```bash
cd "$SGLANG_JAX_DIR"

env -u TT_METAL_RUNTIME_ROOT \
  PYTHONPATH="$SGLANG_JAX_DIR/python" \
  PJRT_NAMES_AND_LIBRARY_PATHS="tt:$LIBTT_DIR/bazel-bin/libtt.so" \
  JAX_PLATFORMS=tt \
  JAX_USE_SHARDY_PARTITIONER=false \
  JAX_COMPILATION_CACHE_DIR=/tmp/sglang-jax-qwen3-8b-jax-cache \
  SGLANG_TT_HOST_WEIGHT_LOAD=1 \
  SGLANG_TT_OPTIMIZATION_LEVEL=1 \
  SGLANG_TT_EXPERIMENTAL_WEIGHT_DTYPE=bfp_bf8 \
  SGLANG_TT_TRACE_DECODE_ONLY=false \
  .venv/bin/python -m sgl_jax.launch_server \
    --model-path Qwen/Qwen3-8B \
    --host 127.0.0.1 \
    --port 31000 \
    --device tt \
    --dtype bfloat16 \
    --attention-backend tt \
    --max-running-requests 2 \
    --max-total-tokens 1024 \
    --max-prefill-tokens 256 \
    --chunked-prefill-size 256 \
    --page-size 32 \
    --watchdog-timeout 1200 \
    --disable-precompile \
    --skip-server-warmup \
    --disable-overlap-schedule \
    --disable-radix-cache
```

`SGLANG_TT_TRACE_DECODE_ONLY=false` records and replays the fixed-shape
prefill graph as well as the decode graph. This avoids dispatching the model's
prefill operations individually from the host.

Because the example disables precompilation and server warmup, the first two
requests can spend substantial time compiling programs and capturing traces.
Warm each input bucket before measuring it. With the five-token prompt below,
SGLang-JAX pads prefill to one 32-token tile. In a 32-sample streaming run on a
P150, after two warmups, time to first token averaged 58.52 ms (95% CI
58.38--58.66 ms) and decode averaged 25.96 tokens/s (95% CI 25.80--26.13
tokens/s). The same libtt build with decode-only tracing averaged 610.50 ms to
first token, so tracing prefill reduced that latency by 90.42% without changing
the deterministic completion.

In another terminal, generate 128 tokens:

```bash
curl -sS http://127.0.0.1:31000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"The capital of France is","sampling_params":{"temperature":0,"max_new_tokens":128}}'
```

On a P150, generation should be about 26 tokens per second after the compile
and trace-capture warmups.
