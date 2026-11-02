# PyTorch with torch_tpu

[torch_tpu](https://github.com/google-pytorch/torch_tpu) is a PyTorch backend
for TPUs that drives its device through the PJRT C API. It loads the plugin
named by `TPU_LIBRARY_PATH`, so it runs on Tenstorrent devices with `libtt.so`.
Validated with torch_tpu 0.2.0 and torch 2.14.1 on one Blackhole, and with
one process per chip on a 4-chip QuietBox.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch_tpu==0.2.0 --extra-index-url https://download.pytorch.org/whl/cpu
pip install bazel-bin/jax_tt_plugin-0.1.5-py3-none-linux_x86_64.whl
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
- torch_tpu only opens a device when its PCI scan finds a Google TPU; when
  the scan finds none, its device count, topology, name and memory describe
  the Tenstorrent chips instead.
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

float8_e4m3fn and float8_e5m2 tensors take one byte per element, and
conversions to and from them round like PyTorch's. A matmul of two float8
matrices of either format, as in `torch._scaled_mm`, reads them natively on
the matrix unit, which treats subnormal operands as zero. A matmul of a float8
matrix and a bfloat16 one computes on bfloat16 values.

## Multi-process tensor parallelism

torch_tpu runs one process per chip, as on a TPU host. Each process opens only
its own chip, and the processes share one 1xN mesh. Collectives run between
them over TT-Fabric, like the single-process mesh in
[Qwen3-8B on two QuietBox chips](qwen3-8b-quietbox-tp2.md).

torch_tpu describes the processes to its PJRT plugin through libtpu's
environment variables:
- `TPU_VISIBLE_CHIPS` gives each process its chip, which libtt passes on as
  `TT_VISIBLE_DEVICES`;
- `TPU_PROCESS_ADDRESSES` and `CLOUD_TPU_TASK_ID` give the processes'
  addresses and this process's index. Process 0 serves a small key-value store
  at its address, which the processes coordinate through.

libtt writes the 1xN mesh descriptor itself. Leave `TT_VISIBLE_DEVICES` and
`TT_MESH_GRAPH_DESC_PATH` unset for multi-process runs.

### torchrun

Let torch_tpu's launch helper describe the chips, then start one process per
chip:

```bash
export LIBTT_TPU_COMPAT=1
eval $(python -m torch.tpu.distributed | sed 's/^/export /')
torchrun --nproc_per_node=4 train.py
```

The helper's `WORLD_SIZE` counts every chip, so `--nproc_per_node` must match
it; with fewer processes the run waits forever. In the script, use the `tpu`
backend:

```python
import torch
import torch.distributed as dist

dist.init_process_group(backend="tpu")
x = torch.ones(8, device="tpu") * dist.get_rank()
dist.all_reduce(x)
```

Collectives run eagerly and in `torch.compile`d programs, as do
`all_gather_object` and `broadcast_object_list`.

### SGLang-TorchTPU

Pass `--tp-size N` to `sglang.launch_server`: SGLang-TorchTPU starts one
process per chip itself. On a 4-chip QuietBox, Qwen3-0.6B decodes 119 tokens/s
at TP4 (89 on one chip) and Qwen3-32B 16.3 tokens/s, with BF16 weights.

### Collective buffers

TT-NN's collectives write to their peers at the addresses of their own
buffers, but separate processes don't allocate in the same order. With more
than one process, each therefore reserves the same DRAM region when the mesh
opens, and collectives allocate their buffers there. The region is
`LIBTT_SYMMETRIC_DRAM_MB` megabytes per chip, 2048 by default. Raise it if a
collective fails with "Symmetric DRAM region of ... is too small for this
collective".

### Recovery

A multi-chip process that is killed can leave the chips' Ethernet links down,
and the next run then hangs or fails while starting the fabric. Reset the
chips before starting again:

```bash
tt-smi -r
```

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
