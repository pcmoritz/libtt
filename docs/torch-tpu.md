# PyTorch with torch_tpu

[torch_tpu](https://github.com/google-pytorch/torch_tpu) is a PyTorch backend
for TPUs that drives its device through the PJRT C API. It loads the plugin
named by `TPU_LIBRARY_PATH`, so it runs on Tenstorrent devices with `libtt.so`.
Validated with torch_tpu 0.2.0 and torch 2.14.1 on one Blackhole.

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
