# PyTorch with torch_tpu

[torch_tpu](https://github.com/google-pytorch/torch_tpu) is a PyTorch backend
for TPUs that drives its device through the PJRT C API. It loads the plugin
named by `TPU_LIBRARY_PATH`, so it runs on Tenstorrent devices with `libtt.so`.
Validated with torch_tpu 0.2.0 and torch 2.14.1 on one Blackhole.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch_tpu==0.2.0 --extra-index-url https://download.pytorch.org/whl/cpu
pip install bazel-bin/jax_tt_plugin-0.1.4-py3-none-linux_x86_64.whl
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
  implementation. `sdpa_kernel(SDPBackend.FLASH_ATTENTION)` selects that
  implementation and MATH (for dropout and masks that need gradients), not a
  native flash attention.

Without the variable, nothing changes.

## What runs

- Eager mode and `torch.compile` (torch_tpu's default backend). HF
  Qwen3-0.6B greedy generation in eager mode matches CPU token for token.
  With `cache_implementation="static"` it decodes at about 31 tokens/s once
  compiled. The first call of each shape compiles.
- Training steps with SGD and AdamW, eager and compiled.
- torch_tpu's op tests (`tests/ops_test.py`, CPU goldens, float32, bfloat16,
  float16 and integer dtypes): 11,963 of 12,493 sampled cases pass. To run
  torch_tpu's tests, see [torch_tpu tests](../README.md#torch_tpu-tests).

float32 matmuls use the matrix unit's TF32 precision by default, like TPUs.
`torch.set_float32_matmul_precision("highest")` makes them exact (about 5e-7
relative error) on the vector unit, at a large cost in speed. float64 runs as
float32.

`torch.manual_seed` and generators make random draws repeat, eager and
compiled. The device draws with Threefry-2x32 (Philox-4x32 for
`torch.randperm`), so the numbers differ from CPU, CUDA and TPU ones. Seeds
and RNG counters preserve all 64 bits; 64-bit random integers have 32 random
bits. This is not general uint64 support: high words are lost across
unconverted operations, including `while`, `case`, `optimization_barrier`
and `custom_call`. Do not carry uint64 RNG state through those operations.

## Not supported

- Dynamic shapes (`torch_tpu.dynamism.mark_dynamic`, ops with data-dependent
  output shapes such as `unique_consecutive`). Compilation fails with
  `dim != mlir::ShapedType::kDynamic`.
- Complex numbers and FFTs.
- Max and average pooling (`stablehlo.reduce_window` beyond cumulative sums).
- Width-changing bitcasts (`tensor.view(torch.uint8)` of a float32 tensor).
- Host offloading, executable serialization (torch_tpu's persistent
  compilation cache), and TPU-only features such as SparseCore ops and Pallas
  kernels.
