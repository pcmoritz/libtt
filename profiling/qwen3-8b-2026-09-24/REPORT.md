# Qwen3-8B TP2 tuning, September 24

The latest retained change optimizes wide sampler reductions in libtt. Fresh paired benchmarks measured **35.03 / 51.91 tokens/s TP1 / TP2** on the short prompt, with approximately **1.8% higher TP2 throughput** on both tested prompt lengths. Scaling is 1.48x short and 1.46x long; the 1.8x target remains unmet. See [the implementation, numerical checks and device profile](wide-reduction/REPORT.md).

## Earlier kernel-tuning round

No performance change was retained from the earlier kernel-tuning round. Seven model-level candidates were tested; none produced a precision-preserving improvement over the September 23 build. The 1.8x target was not reached. The paired confirmation measured 34.49 tokens/s for TP1 and 50.87 for TP2, or **1.47x**. This is the restored baseline, not a new speedup; the slightly different ratio reflects run-to-run variation.

## Protocol

Unmodified sglang-jax commit `6ca38400e6130bc242b215af52b8984bc859036e`, Qwen3-8B BF16 activations / BF8 weights, one active Blackhole chip per P300C board. TP1 uses PCI `0000:01:00.0`; TP2 also uses `0000:04:00.0`. JAX 0.11.1. Settings match [the prior report](../qwen3-8b-2026-09-23/optimization/REPORT.md).

Candidate measurements use a five-token prompt, 128 generated tokens, temperature zero, ignored EOS, three warmups and five measured requests. Values are median streaming decode throughput; compilation is excluded. The runner checks server TP size, and servers are stopped completely between configurations.

## Experiments

| Candidate | TP2 tokens/s | Decision |
| --- | ---: | --- |
| Previous retained build | 50.88 | Reference |
| SwiGLU K block 8 | 50.32 | Slower |
| SwiGLU K block 8, multicast 32 | 49.89 | Slower |
| SwiGLU full-width compute subblock | 50.96 | No meaningful gain |
| SwiGLU K block 4 plus full-width compute subblock | 50.54 | Slower |
| DRAM-sharded projections, implicit LoFi | 52.60 | Rejected: lower default math fidelity |
| DRAM-sharded projections, explicit HiFi2 | 46.96 | Slower |
| DRAM-sharded projections plus unfused up/gate, HiFi2 | 42.64 | Slower |

The DRAM experiment used a new compiler pass inside libtt's tt-mlir dependency. It converted eligible decode activations to width-sharded L1, weights to width-sharded DRAM, selected the DRAM matmul program, and converted outputs back. Weight conversion could be hoisted by the existing const-evaluation/cache machinery. FP32 accumulation passed eight projection tests covering changing weights, activations, and trace replay. The restricted unfused-MLP variant passed these plus ten SwiGLU tests. An earlier overly broad unfusing experiment failed two unsupported-shape cases and was corrected before model benchmarking.

The 52.60 result is **not** a retained improvement. Adding an explicit matmul program and compute configuration changed the default fidelity to LoFi. Restoring HiFi2 removed the apparent gain. The short-prompt token sequence matched the earlier output even in that experiment, illustrating why a generation smoke check alone is insufficient.

## Kernel probe

`kernel-tuning/matmul_bench.cc` is a standalone diagnostic, not production code. For a [1,6144] x [6144,4096] projection it sweeps program configurations and validates exact constant inputs before timing captured traces. The first apparently fast DRAM result, 76 microseconds, did not pass the stricter exact-value check and is excluded. FP32 HiFi2 with packer accumulation passed and reached about 91 microseconds at K block 8, but isolated kernel speed did not translate into a full-model gain. Layout conversions and the complete model workload are included only in the serving measurements.

Larger block 24 exceeded available L1 in the standalone sweep. These logs record exploratory failures, not passing benchmark results. The diagnostic binary was built with `--force_pic` to reuse cached C++ objects and used `LIBTT_BENCH_RUNTIME_ROOT` pointing at the regular wheel's extracted runtime assets, because PIC firmware object names differ. `BENCH_COMPUTE=fp32` selects FP32 without packer accumulation; `packer` selects FP32 with it. Unset uses Metal defaults. Do not compare timings without the associated precision and correctness results.

## Remaining gap

At the prior TP1 rate of 34.83 tokens/s, 1.8x requires 62.70 tokens/s, or about 15.95 ms/token. The retained TP2 baseline takes about 19.65 ms/token: approximately 3.70 ms/token still needs to be removed. The earlier device profile attributed about 1.6 ms/token to collective kernels, so eliminating collective time alone would not suffice. This round targeted the larger matmul cost, but the apparent DRAM-layout win did not survive precision-matched end-to-end validation.

Raw candidate requests, server information and summaries are under `results/`. Rejected patches and validation logs are under `experiments/`; they are not registered in MODULE.bazel. These experiments do not establish a 1.8x speedup.

## Restored build

The rebuilt and reinstalled wheel has SHA256 `a9426eb5435aad09fb355020a44801031437de6f74d95275db0f6d78de2194e1`, identical to the prior validated wheel. All production source changes made for this round were reverted; the previous mesh-shard, replicated-scalar, and physical-link discovery fixes remain intact. SGLang was not modified. The only new workspace artifacts from this round are the report, benchmark sources/scripts, rejected patches, tests used for those experiments, and raw results.

## Final paired confirmation

| Configuration | Median decode tokens/s | Median TTFT |
| --- | ---: | ---: |
| TP1 | 34.49 | 79.05 ms |
| TP2 | 50.87 | 58.44 ms |

Scaling is 1.4747x on the short-prompt protocol, with identical final wheels. All five measured output-token sequences match the prior retained build for TP1=True and TP2=True. Long-prompt and device-profile measurements remain those of the byte-identical prior build; they were not repeated in this confirmation. All benchmark servers were stopped after the run.

## Matched projection profiles and trace-input experiments

See [projection-tuning/REPORT.md](projection-tuning/REPORT.md). The actual projection kernels already scale approximately 2×. Stable device trace-input slots gave only a small throughput change and no scaling improvement; AOT dispatch added no gain. Experimental patches are archived, and the proven baseline was restored. The 1.8× target remains unmet.
