# PyTorch with torch_tpu

[torch_tpu](https://github.com/google-pytorch/torch_tpu) is a PyTorch backend
for TPUs that drives its device through the PJRT C API. It loads the plugin
named by `TPU_LIBRARY_PATH`, so it runs on Tenstorrent devices with `libtt.so`.
Validated with torch_tpu 0.2.0 and torch 2.14.1 on one Blackhole and on the
four chips of a QuietBox 2.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch_tpu==0.2.0 --extra-index-url https://download.pytorch.org/whl/cpu
pip install bazel-bin/jax_tt_plugin-0.1.2-py3-none-linux_x86_64.whl
```

Set `LIBTT_TPU_COMPAT=1` and use torch_tpu as usual:

```bash
LIBTT_TPU_COMPAT=1 python -c '
import torch
x = torch.randn(4, 4, device="tpu")
print((x @ x).cpu())
'
```

The wheel's `libtt_tpu_compat.pth` checks the variable at interpreter startup
and then loads `jax_plugins/libtt/tpu_compat.py`, which runs before torch_tpu:

- `TPU_LIBRARY_PATH` defaults to the wheel's `libtt.so`, which torch_tpu loads
  as its PJRT plugin.
- torch_tpu only opens a device when its PCI scan finds a Google TPU; its
  device count reports the Tenstorrent chips instead.
- torch_tpu's flash attention is a Pallas kernel that only TPUs run, so it is
  turned off and `scaled_dot_product_attention` runs torch_tpu's StableHLO
  implementation.

Without the variable, nothing changes.

## Several chips

torch_tpu runs one process per chip and launches them as on a TPU VM.
`python -m torch.tpu.distributed` prints the environment that `torchrun`'s
processes need:

```bash
export LIBTT_TPU_COMPAT=1
eval $(python -m torch.tpu.distributed | sed 's/^/export /')
torchrun --nproc_per_node=4 train.py
```

torch_tpu never destroys its client, so libtt closes the chips when the
process exits. A process that crashes or is killed leaves their ethernet cores
down, and the next run stops with "Timed out while waiting for active ethernet
core" until the chips are reset with `tt-smi -r`.

libtt reads the environment that torch_tpu gives each process: its chip
(`TPU_VISIBLE_CHIPS`), its index (`CLOUD_TPU_TASK_ID`) and the addresses of all
processes (`TPU_PROCESS_ADDRESSES`). Each process opens its chip as part of a
1xN mesh, and process 0 serves the processes' key-value store at the first
address. Collectives run over TT-Fabric. TT-NN's collectives expect their
buffers at the same address on every chip, so each process reserves
`LIBTT_SYMMETRIC_DRAM_MB` (2048 by default) of DRAM for them when it opens its
chip.

### TorchTitan

[TorchTitan's TPU trainer](https://github.com/google-pytorch/torchtitan/tree/61bdf4c2/torchtitan/experiments/tpu)
runs unmodified. Qwen3-1.7B with FSDP on four chips, 2,048-token sequences,
full activation checkpointing and AdamW:

```bash
torchrun --nproc_per_node=4 -m torchtitan.experiments.tpu.train \
  --module=torchtitan.experiments.tpu.qwen3 --config=qwen3_1_7b \
  --splash_attention_kernel.no_use_splash_attention_kernel \
  --hf_assets_path=<Qwen3 tokenizer directory> --dataloader.dataset=c4
```

It trains at 2.55 s per step, 804 tokens/s per chip, after a first step that
compiles each process's programs. Splash attention is a Pallas kernel, so
attention runs through `scaled_dot_product_attention`.

Checkpoints save and load with `--checkpoint.enable` (PyTorch's distributed
checkpointing): a Qwen3-1.7B checkpoint takes 23 GB and 13 s, and a run
resumed from step 3 reproduces the following steps and the step-6 checkpoint
bit for bit. A streamed dataset such as C4 needs
`--checkpoint.exclude_from_loading dataloader` to resume, since TorchTitan
can't load its data loader's position (also on the CPU). From a seed
checkpoint (`--checkpoint.create_seed_checkpoint`, then
`--checkpoint.load_only`), losses and gradient norms match a 4-process CPU run
to 3-4 digits.

## What runs

- Eager mode and `torch.compile` (torch_tpu's default backend). HF
  Qwen3-0.6B greedy generation in eager mode matches CPU token for token.
  With `cache_implementation="static"` it decodes at about 31 tokens/s once
  compiled. The first call of each shape compiles.
- Training steps with SGD and AdamW, eager and compiled.
- torch_tpu's op tests (`tests/ops_test.py`, CPU goldens, float32, bfloat16,
  float16 and integer dtypes): 11,963 of 12,493 sampled cases pass.

float32 matmuls use the matrix unit's TF32 precision by default, like TPUs.
`torch.set_float32_matmul_precision("highest")` makes them exact (about 5e-7
relative error) on the vector unit, at a large cost in speed. float64 runs as
float32.

## Not supported

- Dynamic shapes (`torch_tpu.dynamism.mark_dynamic`, ops with data-dependent
  output shapes such as `unique_consecutive`). Compilation fails with
  `dim != mlir::ShapedType::kDynamic`.
- Complex numbers and FFTs.
- Max and average pooling (`stablehlo.reduce_window` beyond cumulative sums).
- Reproducible random numbers: `torch.manual_seed` does not make draws
  repeat.
- Width-changing bitcasts (`tensor.view(torch.uint8)` of a float32 tensor).
- Host offloading, executable serialization (torch_tpu's persistent
  compilation cache), and TPU-only features such as SparseCore ops and Pallas
  kernels.
