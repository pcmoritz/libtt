# Wide sampler reductions: retained libtt optimization

Implemented and retained `third_party/tt_xla/tt_mlir_wide_row_reduction.patch`, registered in `MODULE.bazel`. Qwen3-8B TP2 throughput improves approximately **1.8%** on both tested prompt lengths. TP2/TP1 scaling is **1.482× for the short prompt and 1.459× for the long prompt**, still below 1.8×. No sglang-jax source changes or lower-fidelity math were used.

## Why this target

Analysis of the previous device-program records located most inter-program time at iteration and model/sampler boundaries. On TP2, median positive gaps were about 1,502 µs from the final copy to the next unary program, and 615 µs from untilize to the next unary program. Transformer-layer gaps were generally much smaller. These are interval measurements, not a breakdown of CPU-only work. `analyze_gaps.py` and `tp{1,2}-gaps.json` reproduce the adjacent-program attribution.

The sampler also runs reductions over a 151,936-element vocabulary row. The existing row reduction exposes little parallelism and uses heavily padded tiles. This was an isolated compiler optimization with a directly measurable cost, while the earlier AOT dispatch and device trace-slot experiments had not materially improved the boundary gaps.

## Change

After the existing TTIR fusion rewrites, split supported wide rows into 128-element chunks. Reduce each chunk, then reduce the partial results and restore the original output shape. For Qwen's vocabulary this is `[batch, 151936] → [batch, 1187, 128] → [batch, 1187, 1] → [batch, 1, 1]`.

The rewrite applies to FP32 max/sum and BF16 max, rank-two inputs with batch 1 or 2, width at least 32768 and divisible by 128, and reduction over the last axis. Other shapes/types keep the existing lowering. BF16 sums are excluded to avoid BF16 rounding of intermediate partial sums. FP32 sum reassociation can change low bits; numerical accuracy is tested separately from token equivalence. Running after fusion preserves opportunities for softmax and normalization fusion.

The final exported sampler graph (`sampler-tp2.mlir.gz`) confirms both BF16 maximum and FP32 sum use the chunked lowering. Existing model projection fusions are unaffected by this shape guard.

## Fresh end-to-end comparison

Same hardware and configuration as the preceding reports: one active chip per Blackhole board, TP1 PCI `0000:01:00.0`, TP2 additionally `0000:04:00.0`, BF16 activations/BF8 weights, Qwen/Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`, unmodified sglang-jax commit `6ca38400e6130bc242b215af52b8984bc859036e`, JAX 0.11.1, no overlap scheduling. Short/long prompts contain 5/199 tokens; each request generates 128 tokens with temperature 0 and ignore-EOS. Each prompt has three warmups and five measured requests; table values are medians. Separate concurrent-request checks and host profiles follow the timed single-request tests.

| Configuration | Short tok/s | Long tok/s |
|---|---:|---:|
| Fresh baseline TP1 | 34.6705 | 33.6165 |
| Optimized TP1 | 35.0280 | 34.2255 |
| Fresh baseline TP2 | 50.9828 | 49.0478 |
| Optimized TP2 | **51.9146** | **49.9384** |
| TP2 throughput gain | **1.828%** | **1.816%** |

TP2 saves **0.352 ms/token short** and **0.364 ms/token long**. TP1 also benefits, by 1.03%/1.81%; the gain should not be presented as a TP2-only optimization. The short-prompt scaling ratio changes from 1.4705× to 1.4821×, and the long-prompt ratio remains essentially unchanged at 1.4591×.

The baseline wheel was rebuilt and its SHA verified against the previously restored baseline. Optimized measurements preceded the fresh baseline measurements. Earlier candidate measurements also showed the same direction: FP32-only rewrite 51.48 tok/s, FP32 plus BF16-max rewrite 51.87 tok/s, final rewrite 51.91 tok/s. These are small gains from one host, not a broad hardware study.

Two concurrent requests also completed correctly: batch end-to-end throughput changed from 59.64 to 60.42 tok/s TP1 and from 83.00 to 84.15 tok/s TP2. These include prefill and are not the single-request decode rates in the table.

## Device-level evidence

Matched 500 ms warmed windows, chip 0; baseline is the preceding projection-profile run and optimized data is a separate diagnostic run. The optimized analyzer found 25 complete decode cycles on each chip with zero reported dropped records.

| Device program category | Baseline ms/token | Optimized ms/token |
|---|---:|---:|
| Generic reductions | 0.43069 | **0.04338** |
| Fill/padding | 0.15243 | **0.03165** |
| Reshape | 0.08107 | 0.20439 |
| Matmul | 10.93655 | 10.93202 |
| Union of all program intervals | 16.69463 | 16.30325 |
| LM-head-end to next LM-head-end | 19.38766 | 19.06370 |

Reduction programs are **9.93× faster**. Added reshapes cost approximately 123 µs/token, nearly offset by 121 µs less fill/padding. The reduced device-program coverage is consistent with the uninstrumented end-to-end savings. Time outside recorded programs remains around 2.7 ms/token; this change does not solve the remaining synchronization/dispatch bottleneck.

Raw compressed data is under `device-profile/`. Reproduce the analysis with:

```sh
python profiling/qwen3-8b-2026-09-24/projection-tuning/analyze_projections.py \
  profiling/qwen3-8b-2026-09-24/wide-reduction/device-profile
```

## Validation and limits

- 16 new wide-reduction cases passed: exact finite max/sum, negative-value padding, supported and fallback widths, batch 1/2, keepdims on/off, BF16 maximum, FP32 cancellation, log-softmax accuracy and repeated trace replay.
- 10 existing local-mesh regression cases passed on two devices.
- All 20 measured single-request token sequences and all four concurrent-request token sequences match their corresponding fresh baseline outputs.
- A final TP1 production-wheel repeat measured 35.0098 tok/s short. All eight returned token log probabilities and top-three log-probability entries exactly match the baseline; see `tp1-logprob-check.json`.
- Full paired benchmark requests, summaries, server configuration and comparison data are saved in `results/`, `comparison.json` and `token-checks.json`.
- Existing overlap scheduling was tested separately with this compiler change. It measured 51.65 tok/s TP2 versus 51.91 without overlap, with matching tokens; it was not selected.
- Additional TP2 service validation with requested log probabilities failed **on the unchanged baseline** during model compilation: a reshape expected 311164928 elements but received 622329856. The normal benchmark completed before this extra request. This does not invalidate the timed runs, but an end-to-end TP2 log-probability comparison is unavailable. See `baseline-tp2-logprob-error.txt`. Direct numerical log-softmax tests passed.

The diagnostic profiler hook was removed. The final production wheel SHA256 is `4b5f5477f0598192b99aa88a58271b40d19dda7f8331ffbb61f1df920a811c66`, identical to the wheel used for the final paired optimized benchmarks. Baseline SHA256: `a9426eb5435aad09fb355020a44801031437de6f74d95275db0f6d78de2194e1`.
