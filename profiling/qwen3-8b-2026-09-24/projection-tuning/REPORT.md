# Projection scaling and trace-input experiments

Qwen3-8B TP2 still runs about **1.47× TP1**, not 1.8×. This round mapped the actual decode projection kernels and tested two libtt trace-input implementations plus sglang-jax's existing AOT dispatch option. None provided a material scaling improvement. Experimental changes were archived and removed from the production build; the previous local-mesh correctness and link-discovery fixes remain.

## Matmuls already scale close to 2×

These are matched warmed device-program measurements, not standalone matmul estimates. Configuration matches the existing benchmark: BF16 activations, BF8 weights, one active Blackhole chip on each physical board, PCI 01:00.0 / 04:00.0, 12 KiB fabric packets, greedy batch-one decoding. sglang-jax is the unmodified pcmoritz blackhole-multinode checkout at `6ca38400e6130bc242b215af52b8984bc859036e`.

| Projection | TP1 µs/call | TP2 µs/call, chip 0 | TP1 / TP2 |
|---|---:|---:|---:|
| QKV | 72.81 | 37.84 | 1.92× |
| Attention output | 48.27 | 25.76 | 1.87× |
| Fused SwiGLU | 285.60 | 140.15 | 2.04× |
| MLP down | 135.80 | 70.02 | 1.94× |
| LM head | 2293.75 | 1078.31 | 2.13× |

The projection totals are approximately **21.84 ms/token TP1 and 10.94 ms/token TP2**. In particular, the actual TP2 down projection is already 70 µs. The earlier standalone DRAM-sharded candidate at about 91 µs was therefore slower than the actual production kernel; its comparison with an explicitly configured 151 µs microbenchmark was misleading.

The analyzer identifies `[QKV, output, SwiGLU, down] × 36 + LM head` using the executable order verified in the fusion audit. The distinct SwiGLU kernel anchors each group. It excludes incomplete or unexpected cycles, deduplicates profiler callback records by chip/start/end, and uses the last 500 ms. There are 16 complete TP1 cycles and 24 per TP2 chip, with zero reported dropped records. Chip 1 has similar or slightly lower projection times. See `tp1-projections.json` and `tp2-projections.json`.

## What remains

| Profile component, chip 0 | TP1 ms/token | TP2 ms/token |
|---|---:|---:|
| Matmul programs | 21.84 | 10.94 |
| QKV-head creation | 1.21 | 0.61 |
| Normalization | 0.77 | 0.77 |
| KV updates | 0.61 | 0.38 |
| RoPE | 0.33 | 0.33 |
| SDPA | 0.27 | 0.25 |
| Generic reductions | 0.43 | 0.43 |
| Reduce-scatter | 0 | 0.76 |
| All-gather, including sampler | 0 | 0.83 |
| Binary elementwise, including residual adds | 0.12 | 0.34 |
| Median complete device-program coverage | 26.82 | 16.69 |
| Median LM-head-end to next LM-head-end | 28.43 | 19.39 |

Program coverage is the union of program intervals. The approximately 1.61 / 2.69 ms outside recorded programs includes dispatch gaps and unrecorded transfers or synchronization; it is **not a measured Python-only cost**. Category medians need not sum to the median coverage. These short profiler runs are separate from uninstrumented 128-token throughput runs.

The existing TP1/TP2 fusion audit still applies. TP2's 72 collective-then-residual-add chains cannot safely become a pre-reduction replicated bias: that would count the residual twice. A collective-aware residual/RMSNorm implementation would require new work. The added residual kernels account for only about 0.22 ms/token; removing them alone cannot deliver the target.

Against the previous paired baseline, 1.8× requires TP2 to reach 62.09 tok/s, or 16.11 ms/token—about 3.55 ms below its 19.66 ms baseline. Even eliminating all recorded collective time and the extra residual adds is insufficient. Any optimization shared with TP1 must also be evaluated against its improved TP1 baseline.

## Implemented and measured experiments

Three warmups and five measured requests, 5-token prompt, 128 generated tokens, temperature 0, ignore EOS, streaming interval 1. Medians:

| Configuration | TP1 tok/s | TP2 tok/s | Paired ratio |
|---|---:|---:|---:|
| Previous restored baseline | 34.4926 | 50.8677 | 1.4747× |
| Device activation slots | 35.0112 | 51.2923 | 1.4650× |
| Device slots + existing AOT dispatch | — | 51.1536 | — |

1. **Direct device activation aliasing:** taught trace hoisting to retain large BF16/F32 activations on device and skip same-buffer copies. This regressed badly: changing input addresses caused trace recapture. Warmup requests reached only 8.20 and 11.84 tok/s. Stopped early; no valid warmed median. Saved as `experiments/activation-alias.patch`.
2. **Stable device activation slots:** allocate private trace slots for large ordinary floating-point function arguments, preserve aliasing for weights/KV caches, copy device-to-device, and distinguish copied slots from aliased inputs in trace reuse and ownership. Passed the targeted tests and gained roughly 0.8% TP2 throughput relative to the previous baseline, but TP1 also improved and the scaling ratio did not. No claim of a statistically established production gain: this was one paired quick benchmark, not repeated interleaved A/B trials. Saved as `experiments/trace-device-slots.patch`; opt-in compiler switch was `LIBTT_KEEP_TRACE_ACTIVATIONS=1`.
3. **Cached AOT dispatch:** existing `SGLANG_JAX_AOT_DISPATCH=1`, with the device-slot prototype. Verified logs show the AOT path used 399 stable and 76 dynamic model arguments. No incremental improvement. No sglang-jax source changes.

All five measured token sequences for each device-slot TP configuration exactly match its corresponding baseline. Five targeted activation/parameter-layout tests and ten local-mesh tests passed on the prototype. The activation tests cover changing values, multiple producers, retained outputs, 24 iterations, and alternating conditional branches. They do not by themselves prove internal trace cache eviction occurred.

Given the small observed gains, the prototype's added compiler/runtime ownership complexity is not being made the default. The prototype, test source, request data and summaries are retained under `experiments/`. No reduced math fidelity was used.

## Reproduction and next implementation target

Compressed raw callback data and the diagnostic profiler patch are included. Analyze each TP directory separately:

```sh
python profiling/qwen3-8b-2026-09-24/projection-tuning/analyze_projections.py \
  profiling/qwen3-8b-2026-09-24/projection-tuning/device-profile/tp1
python profiling/qwen3-8b-2026-09-24/projection-tuning/analyze_projections.py \
  profiling/qwen3-8b-2026-09-24/projection-tuning/device-profile/tp2
```

The next substantial libtt work should target collective/normalization boundaries and the TP2-specific gaps between programs, with measurements proving each contribution. The existing minimal all-reduce implementation requires L1 width-sharded inputs, scratch buffers and semaphore management; it is not a safe drop-in replacement for the present DRAM-interleaved path. An effective implementation should keep output projections, collective results, residual addition and normalization in compatible layouts, rather than inserting conversions around each operation. This round did not implement that kernel or reach 1.8×.
