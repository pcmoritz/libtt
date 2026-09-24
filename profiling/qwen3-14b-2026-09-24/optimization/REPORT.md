# Qwen3-14B bottlenecks and SwiGLU optimization

Subsequent collective optimization and measurements: [fused TP2 all-reduce](../collectives/REPORT.md).

The retained libtt change improves TP2 decode throughput by **about 1.7–1.9%**, confirmed by a fresh baseline/optimized recheck. The initial full benchmark gains are 1.9% on the short prompt and 1.8% on the long prompt. Paired TP2/TP1 scaling is now about **1.65×**, still below 1.8×. No sglang-jax source or serving-setting changes were needed.

## Uninstrumented benchmark

Same workload as the initial 14B benchmark: three warmups and five measured requests per prompt, 128 generated tokens, temperature zero, EOS ignored. Values below are medians; decode timing excludes compilation and time to first token.

| Configuration | Short tokens/s | Long tokens/s | Short ms/token | Long ms/token |
|---|---:|---:|---:|---:|
| Baseline TP1 | 20.46 | 19.86 | 48.868 | 50.360 |
| Optimized TP1 | 20.36 | 19.85 | 49.109 | 50.377 |
| Baseline TP2 | 33.00 | 32.10 | 30.304 | 31.153 |
| Optimized TP2 | **33.62** | **32.69** | **29.741** | **30.592** |

TP1 is essentially unchanged (short −0.49%, long −0.03%); its 17408-wide MLP remains outside the fused kernel's register-capacity limit. The short/long scaling ratios change from 1.613×/1.617× to 1.651×/1.647×. Do not attribute the small TP1 slowdown to the optimization without further evidence.

Optimized TP2 median time to first token is 76.04 ms / 105.95 ms (short/long), versus 76.11 ms / 105.99 ms before. Two-request aggregate end-to-end throughput changes from 56.25 to 57.19 tokens/s on TP2, and from 36.59 to 36.48 on TP1. Concurrency uses only one measured pair after a warmup pair, so it is weaker evidence than the single-request medians.

See `comparison.json`, `final-tp*/requests.json`, summaries, host traces, and logs. The original baseline artifacts are one directory above.

## Fresh baseline/optimized recheck

After the full paired run, the original wheel was reinstalled and the short-prompt protocol repeated on TP1 and TP2, followed by the optimized wheel on TP2. Each uses three warmups and five measured 128-token requests.

| Recheck | Short tokens/s | Decode ms/token |
|---|---:|---:|
| Original wheel, TP1 | 20.3610 | 49.1135 |
| Original wheel, TP2 | 33.0539 | 30.2536 |
| Optimized wheel, TP2 | 33.6224 | 29.7421 |

This confirms a **1.72% TP2 improvement** and **1.651× scaling**. TP1's fresh original-wheel result matches the optimized full run (20.3628 tokens/s), supporting the conclusion that its small difference from the initial historical baseline was run-to-run variation. Both optimized TP2 measurements agree within 0.01%. See `recheck.json` and `recheck-*/requests.json`. The optimized production wheel was restored afterward and all servers were stopped.

## Where time is lost

Device-program profiles use separate 32-token requests. The analyzer identifies complete 40-layer cycles through QKV-head creation, verifies the four projections per layer plus LM head, and deduplicates callback records by chip/start/end. The final 500 ms contains nine complete TP1 cycles, 15 baseline TP2 cycles per chip, and 16 optimized TP2 cycles per chip. All report zero dropped records. The following numbers are from chip 0; do not add times from both chips.

| Profile component, ms/token | Baseline TP1 | Baseline TP2 | Optimized TP2 |
|---|---:|---:|---:|
| Matmul programs | 39.90 | 20.10 | 20.25 |
| All-gather + reduce-scatter | 0 | 1.96 | 1.96 |
| Normalization | 0.97 | 0.90 | 0.89 |
| QKV-head creation | 1.58 | 0.80 | 0.80 |
| Binary elementwise | 0.49 | 0.69 | 0.43 |
| Slice | 0.52 | 0.29 | 0.01 |
| Unary elementwise | 0.37 | 0.29 | 0.08 |
| Union of recorded program intervals | 46.58 | 27.16 | 26.54 |
| LM-head-end to next LM-head-end | 48.43 | 29.72 | 29.43 |
| Difference: time outside recorded programs | 1.85 | 2.56 | 2.88 |

Other categories, including KV updates, RoPE, attention, and layout conversions, are in the analysis JSON files. Category medians need not sum to median union coverage. Time outside programs includes dispatch gaps and unrecorded transfers/synchronization; it is **not a measured CPU-only cost**. These shorter profiled runs are not interchangeable with the uninstrumented throughput measurements.

Matmuls already scale almost 2×. TP2 adds 80 all-reduces per token, and normalization, sampling, and other small operations do not halve in cost. Together with the gaps between programs, these explain why end-to-end scaling trails matrix-multiplication scaling.

## Retained change

The compiler previously allowed fused matmul–SwiGLU only when the output width divided evenly across 96 workers. Qwen3-14B TP2 produces 8704 output elements, or 272 tiles, so all 40 decode MLP layers missed that fusion.

The compiler now accepts whole-tile widths within the existing 96–384-tile capacity. The TT-Metal factory rounds up tiles per worker. The reader zero-fills unused tiles locally and suppresses writes outside the logical output. This avoids both out-of-bounds access and fetching unused weights from DRAM. Existing compute fidelity, FP32 accumulation, K-block size, and multicast geometry are retained.

Compiler IR confirms **40 fused SwiGLU operations** in the 14B TP2 batch-one decode graph. The fused up/gate+activation program takes about 253.39 µs/layer, compared with 249.65 µs for the old projection alone. Removing the separate slices, SiLU, and multiply reduces total recorded program coverage by about 0.62 ms/token. The profiled iteration saves about 0.29 ms, while the uninstrumented 128-token benchmark saves about 0.56 ms/token.

Production patches:

- `third_party/tt_metal/matmul_swiglu_tail_tiles.patch`
- `third_party/tt_xla/tt_mlir_swiglu_tail_tiles.patch`

They are registered in `MODULE.bazel`. Diagnostic profiler hooks and rejected tuning changes are not registered in the production build.

## Experiments not retained

| Variant | Fused up/gate µs/layer | Profiled decode interval, ms | Outcome |
|---|---:|---:|---|
| Pad by reading the last valid weight tile | — | — | 31.29 tokens/s uninstrumented; regression |
| Distribute padded reads across valid tiles | 265.93 | 30.03 | Extra DRAM traffic erases fusion savings |
| Eight-tile K/multicast blocks | 266.24 | 30.05 | No improvement |
| Eight-tile K blocks, six-tile output block | 266.03 | 29.93 | No material improvement |
| Local zero-fill, original blocks | **253.39** | **29.43** | Retained |
| Exact 68-core partition, four tiles/core | 257.02 | 29.65 | Slower than retained version |
| Local zero-fill + larger general matmul K blocks | — | — | 33.47 tokens/s; slower than retained version |

The padded 96-core layout has 288 slots for 272 real tiles, adding about 5.9% weight reads if padding fetches real weights. The measured extra kernel time was consistent with this cost. This motivated local zero-fill rather than further compute-block tuning. Analysis JSON, experiment patches, and available benchmark request data are archived.

## Validation and limitations

- All **17 SwiGLU regression cases passed**, with three executions per case. Coverage includes partial worker groups, one/two/three/four tiles per worker, the actual 5120×17408 weight matrix, batch two, batch 32, and unsupported-shape fallbacks. Exactly representable BF8 weights are checked against a NumPy FP32 reference using the existing atol 0.01 / rtol 0.04 criterion.
- Both final servers completed the full protocol: 21 requests each, including warmups, ten measured single requests, concurrent pairs, and host profiling. All requested output lengths were returned.
- All five measured generations per prompt are identical within each final TP setting. TP1 outputs exactly match baseline. TP2 differs from baseline starting at token 38 (short) and 53 (long), zero-based. The fused kernel retains intermediate values in FP32 instead of materializing the separate BF16 intermediates, so bitwise equivalence is not promised. These are functional/numerical checks, not a model-quality evaluation.
- The retained production wheel SHA256 is `e03a05ace54a038352a7d092fb9c64562657d884e707302f35a4a5dbe75d29d3`. Baseline SHA256 is `4b5f5477f0598192b99aa88a58271b40d19dda7f8331ffbb61f1df920a811c66`.
- Model revision, sglang-jax commit, hardware, precision and launch flags match `../environment.json` and the original benchmark. The source changes are based on libtt `3f9d137`.

## Remaining target

At the optimized TP1 rate, 1.8× would require about 36.65 short-prompt tokens/s, or 27.28 ms/token. The current 29.74 ms/token still needs about **2.46 ms/token** removed. The next substantial target is the collective/residual/normalization boundary and the associated gaps, rather than expecting another 2× from matmuls. The graph has 80 projection → all-reduce → residual-add chains per token. Improving these requires a collective-aware implementation with compatible layouts; adding a replicated residual before all-reduce would incorrectly count it twice.
