# libtt

`libtt.so` is a Bazel-built PJRT plugin for Tenstorrent devices. The PJRT
implementation comes from the pinned `tt-xla` repository, with `tt-mlir` and
`tt-metal` built through Bazel overlays in this repository.

The local code in this repository is intentionally small:

- it materializes the embedded TT-Metal runtime archive before the plugin starts;
- it links the upstream `tt-xla` PJRT plugin into the final shared library;
- it hides internal symbols so the shared object only exports the PJRT entrypoints.

## Build

```bash
bazel build //:tt
```

The output is `bazel-bin/libtt.so` on Linux and `bazel-bin/libtt.dylib` on
macOS.

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
export SGLANG_JAX_DIR=/tmp/sglang-jax
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

Point `PJRT_NAMES_AND_LIBRARY_PATHS` at the `libtt.so` built from this checkout
if you build in a different directory. Keep `JAX_COMPILATION_CACHE_DIR` stable
between runs to avoid recompiling the same Qwen3-8B shapes.
