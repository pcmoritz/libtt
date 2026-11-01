# SGLang-TorchTPU

[SGLang-TorchTPU](https://github.com/sgl-project/sglang-torchtpu) serves models
through [torch_tpu](torch-tpu.md). With `LIBTT_TPU_COMPAT=1` it runs on one
Tenstorrent chip with no changes to SGLang or torch_tpu. Its attention kernel
is a Pallas TPU kernel, which libtt replaces with JAX and TTNN's paged
attention operations (`python/jax_plugins/libtt/sglang_tpu_kernels.py`).

## Setup

Install SGLang-TorchTPU from source as its README describes, then the libtt
wheel without its dependencies, so that SGLang's pinned JAX stays:

```bash
git clone https://github.com/sgl-project/sglang-torchtpu.git
cd sglang-torchtpu
uv venv -p 3.12 .venv
cp python/pyproject_tpu.toml python/pyproject.toml
VIRTUAL_ENV=.venv uv pip install --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cpu -e python/
VIRTUAL_ENV=.venv uv pip install --no-deps -e .
VIRTUAL_ENV=.venv uv pip install --no-deps \
  /path/to/libtt/bazel-bin/jax_tt_plugin-0.1.5-py3-none-linux_x86_64.whl
```

## Launch a server

```bash
LIBTT_TPU_COMPAT=1 SGLANG_PLATFORM=tpu \
TORCH_TPU_INTERNAL_XLA_OPTIONS="enable_trace=true" \
.venv/bin/python -m sglang.launch_server \
  --model-path Qwen/Qwen3-8B --port 31000 \
  --device tpu --dtype bfloat16 --kv-cache-dtype bfloat16 \
  --mem-fraction-static 0.75 --max-total-tokens 16384 \
  --enable-torch-compile --tp-size 1 --page-size 64 \
  --context-length 2048 --chunked-prefill-size 512 \
  --torch-compile-max-bs 4 --tpu-compile-bs 1 4 --tpu-compile-tokens 64 \
  --grammar-backend none
```

Then:

```bash
curl -sS http://127.0.0.1:31000/generate -H 'Content-Type: application/json' \
  -d '{"text": "The capital of the Netherlands is",
       "sampling_params": {"temperature": 0, "max_new_tokens": 64}}'
```

The settings that matter:

- `--kv-cache-dtype bfloat16`. SGLang-TorchTPU defaults to an FP8 KV cache
  for its attention kernel. TTNN's paged decode attention reads bf16, so with
  FP8 every step gathers each sequence's pages and converts them on the host,
  which is about 50 times slower.
- `TORCH_TPU_INTERNAL_XLA_OPTIONS="enable_trace=true"` records and replays
  the compiled prefill and decode programs instead of dispatching their
  operations from the host. It doubles decode speed with Qwen3-0.6B. torch_tpu
  takes several options separated by spaces.
- `--mem-fraction-static`: SGLang-TorchTPU's default of 0.4 leaves no memory
  for the KV cache next to Qwen3-8B's 16 GB of bf16 weights.
- `--max-total-tokens` keeps the KV pool small. Traced programs keep their
  buffers, and a pool sized by the memory fraction alone has not been tried.
- `--tpu-compile-bs` and `--tpu-compile-tokens` choose the shapes compiled at
  startup. Every other shape fails once the server is up.

## Speed

Greedy decoding of 256 tokens (128 for Qwen3-8B) on one Blackhole P150,
batch 4 as four concurrent requests:

| Model | Batch 1 | Batch 4 |
|---|---|---|
| Qwen3-0.6B | 89 tokens/s | 291 tokens/s (76 per request) |
| Qwen3-8B, bf16 weights | 19 tokens/s | 72 tokens/s (18 per request) |

Qwen3-8B reads its 16 GB of weights on every step, so it decodes at about the
speed the chip's memory allows for bf16 weights.
