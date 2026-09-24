# TP2 optimization experiments

All trials use Qwen3-8B, the supplied sglang-jax branch, BF16 activations, BF8 weights, the same two Blackhole chips on separate boards, and a fresh JAX executable cache per compiled candidate. Short-prompt rates below are medians of five requests after three warmups, each generating 128 tokens. No temporary SGLang instrumentation remains.

| Candidate | Decode tokens/s | Decision |
| --- | ---: | --- |
| Original baseline | 49.06 | Reference |
| Stage-timing diagnostic | 49.08 | Diagnostic only; JAX readiness does not isolate asynchronous device execution |
| Small TP2 all-gather + local sum | 48.44 | Removed: slower |
| Decode matmul K block 32 instead of 8 | 48.10 | Removed: slower |
| 8 KiB fabric packets instead of 12 KiB | 49.12 | Removed: no convincing improvement |
| Route-aware link discovery + 8 KiB packets | 50.88 | Improvement from link discovery |
| Route-aware link discovery, original 12 KiB packets | 50.88 | Final retained change |

The rejected matmul candidate changes `in0_block_w = decode_matmul ? 8 : 2` to `32 : 2` in `third_party/tt_metal/matmul_balanced_1d_planner.patch`. Other experimental patches and all trial request data are saved in `experiments/`. Different reduction/matmul decompositions changed some generated tokens; neither candidate is retained. Both packet and link-discovery candidates preserved the short-prompt baseline output.

## Device profiling

The temporary patch `experiments/realtime-csv-diagnostic.patch` registers the Metal real-time program profiler callback after mesh creation. Registration must happen again after MetalContext reinitialization, which clears the registry. Multiple callbacks can observe the same record; analysis deduplicates by chip and start/end timestamps. The patch is archived, not enabled in MODULE.bazel or the final wheel.

The successful diagnostic run used route-aware links and 8 KiB packets. It generated three 32-token short-prompt requests. `device-profile/` contains compressed raw CSVs and a summary of the final 500 ms on chip 0, excluding startup and prefill. No dropped records were reported in this window. Run `python3 analyze_device.py` to reproduce it.

Matmuls occupy approximately 281 ms of that 500 ms window (56%). All-gather and reduce-scatter occupy approximately 41 ms together (8%). Collective time is roughly 1.6 ms per decode token, estimated from the 72 per-layer all-reduces per token. The rest includes normalization, cache updates, attention, reshapes, sampling, and gaps between programs. These are observed device-program intervals; percentages use elapsed clock time and are not a CPU/device critical-path decomposition. The device profile is diagnostic and uses a different packet size from the final benchmark.

At the original TP1 rate, the 2x target is approximately 69 tokens/s (14.5 ms/token), versus 19.65 ms/token achieved by final TP2. Even eliminating the observed collective kernel time would not close that gap. Further work should focus on decode matmul efficiency and reducing the repeated non-matmul kernels and host synchronization, with another matched TP1 measurement for any general optimization.
