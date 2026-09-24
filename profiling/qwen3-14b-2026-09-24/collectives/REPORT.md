# Qwen3-14B: reuse TT-Metal's fused two-rank all-reduce

This change is implemented in libtt's TT-Metal patch layer. SGLang-JAX and the model are unchanged. It replaces decode-sized Blackhole TP2 reduce-scatter + all-gather with the existing experimental exchange/reduce kernel, extended to read and write DRAM directly.

## Existing implementations

Source audit is against pinned TT-Metal commit `5beed318d0f0d1c6212e605947fb0be80c9e0a1d`, not an assertion about newer upstream versions.

- `ttnn/cpp/ttnn/operations/ccl/all_reduce/all_reduce.cpp` delegates to the generic experimental composite implementation. This was the path libtt used.
- `ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/all_reduce_async.cpp` also exposes a different overload taking an intermediate buffer and global semaphore. It delegates to `ttnn::prim::all_reduce_async`, which exchanges inputs into L1 scratch and reduces them in one mesh workload.
- `ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/` contains that fused implementation. It originally required width-sharded input/output and explicitly rejected Blackhole DRAM input. Its raw NOC addressing was not suitable for Blackhole DRAM.
- `tests/ttnn/unit_tests/operations/ccl/test_new_all_reduce.py` exercises the sharded implementation.

The retained adapter reuses its fabric sender, reduction receiver, and reduction compute kernel. Two small accessor-based dataflow kernels read interleaved DRAM input and write DRAM output; remote fabric writes still target L1 scratch. There is no intervening all-gather or separate input/output layout conversion.

## Compact transport (option 2)

The logical single-request BF16 vector is 5,120 elements = 10 KiB. Its tiled storage is 32 × 5,120 elements = 320 KiB. This implementation still transports full tiles.

TT-Metal has aligned row-major all-gather transport, and the generic all-reduce's `local_sum` / `local_sum_float32` helpers accept row-major tensors by converting them to tiles for reduction and converting back afterward. These are reusable pieces, but the audited path does not provide a drop-in all-reduce that skips the unused rows of this tiled decode tensor. A compact extension would gather the valid face rows while reading tiles, send packed data through the existing fabric protocol, then restore tile positions in L1 scratch before reduction. It needs explicit handling for both batch 1 and 2, face offsets, scratch initialization and synchronization. A standalone untilize/collective/tilize chain adds launches and is not assumed to be faster.

## Scope and correctness

The default specialization is limited to exactly two Blackhole devices, BF16 tiled DRAM input/output, one or two logical rows within one physical tile row, widths 1,024–8,192 divisible by 1,024, an explicit two-device cluster axis, and no subdevice restriction. Other tensors use the existing generic implementation. Set `LIBTT_TP2_MINIMAL_ALL_REDUCE=0` before starting a fresh process to disable it.

There are 32 reduction workers, with a width-sharded L1 intermediate holding both ranks' tiles. Fabric link selection uses TT-Metal's existing helper. The program cache owns the semaphore when the caller does not supply one; allocation and initialization happen on cache miss, outside trace replay. FP32 accumulation is enabled and included in the program-cache key. Weight format and matmul fidelity are unchanged.

`test-final.log`: 18 tests passed with the default specialization. Coverage includes distinct values per rank, two sums in one executable, repeated changing inputs, trace on/off, BF16 rounding, supported widths, batch 2, float32/large-width/batch-32 fallbacks, and a leading-batch shape whose padding occupies two tile rows. Both prototype paths reproduced the previous TP2 short-prompt text exactly across all eight requests. Model text comparisons are not a formal model-quality evaluation.

## Device profile

The diagnostic callback was enabled only in a separate profiling wheel and removed from the final wheel. Analysis uses the last 500 ms, deduplicates program records and reports medians over 16 complete decode iterations per chip; there were no reported dropped records. Timings below are per chip, not summed across devices.

| Chip 0 category | Previous TP2 | Direct exchange/reduce |
|---|---:|---:|
| Collective programs, including remaining all-gather | 1.955 ms/token | 1.188 ms/token |
| Matmul programs | 20.254 ms/token | 20.260 ms/token |
| Union of program intervals | 26.544 ms/token | 25.812 ms/token |
| LM-head-to-LM-head interval | 29.428 ms | 28.636 ms |

The fused all-reduce itself totals 1.095 ms/token on chip 0 and 1.510 ms/token on chip 1. Chip 1's matmuls are correspondingly faster: collective duration includes waiting for the peer, so these times must not be added across chips or interpreted as pure link bandwidth. Both devices complete a decode iteration in about 28.65 ms in this profile.

The profile supports a saving of about 0.77 ms/token from this change. Roughly 1.6 ms/token more must be removed from the end-to-end result to reach 1.8× the recorded TP1 rate. Compact transport may help further, but eliminating all remaining chip-0 collective time alone would not cover that gap; the large projection matmuls remain the largest target.

## Benchmark protocol and preliminary trials

Same protocol and model as [the preceding optimization report](../optimization/REPORT.md): Qwen/Qwen3-14B, BF8 weights, BF16 activations, one/two Blackhole cards, 128 generated tokens, temperature zero, ignore EOS, overlap scheduling and radix cache disabled. Short prompt has 5 tokens; long prompt has 199. Each prompt has three warmups and five measured requests; reported rates are medians. Concurrent throughput is one measured pair after a warmup pair.

Quick short-prompt trials:

| Variant | Decode tokens/s | ms/token |
|---|---:|---:|
| Previous optimized TP2 | 33.622 | 29.742 |
| Existing fused kernel via separate L1 conversions | 34.515 | 28.973 |
| Direct DRAM reader/writer prototype | 34.607 | 28.896 |

The L1-conversion prototype is archived for comparison, not registered in the build. The direct path is the retained implementation. Raw request data, profile CSVs and analysis accompany this report. Run scripts preserve the original machine-specific paths; the benchmark and launcher are also archived in the parent reports.

## Reproduction and artifacts

Build with `/tmp/libtt-profile/bin/bazel build -c opt //:jax_tt_plugin_wheel`. The retained wheel SHA-256 is `5caa0d43fbe10d8a58a18fb8350718b41f454d9f82d2c3e28b306396cc957ecd`; it has no profiler hook. The preceding baseline wheel was `e03a05ace54a038352a7d092fb9c64562657d884e707302f35a4a5dbe75d29d3`.

Regression command:

```sh
/tmp/libtt-profile/bin/bazel test -c opt //tests:libtt_test_suite \
  --test_arg=tp2_all_reduce_regression_test.py --test_arg=-xq \
  --test_env=TT_VISIBLE_DEVICES=0000:01:00.0,0000:04:00.0 \
  --test_env=TT_METAL_OPERATION_TIMEOUT_SECONDS=30 \
  --test_timeout=180 --test_output=errors
```

`build-overlap-run/` preserves a completed full run whose short-prompt measurement overlapped the final library build. It is excluded from performance conclusions. All 21 response texts matched the preceding TP2 benchmark, including concurrent requests and the profiling request. The later `confirmed/` run uses the final wheel, with no build or other hardware test running concurrently.

## Confirmed final performance

| Prompt | TP1 tokens/s | Previous TP2 | Fused TP2 | TP2 improvement | TP2 / TP1 |
|---|---:|---:|---:|---:|---:|
| short | 20.363 | 33.624 | 34.612 | 2.94% | 1.700× |
| long | 19.850 | 32.688 | 33.574 | 2.71% | 1.691× |

The clean final run completed all 21 requests, and all 21 response texts matched the preceding TP2 run. Concurrent aggregate end-to-end throughput was 58.58 tokens/s (one measured pair). Short/long median TTFT was 76.00/106.03 ms. TP1 values are the preceding confirmed full benchmark; TP1 was not rerun because this change requires two devices.

The final wheel is installed in the serving environment, the specialization is enabled by default for the guarded shapes, and benchmark servers were stopped. The collective implementation is a performance-oriented prototype and probably needs cleanup before upstreaming.
