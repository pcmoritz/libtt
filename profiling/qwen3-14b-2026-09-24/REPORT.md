# Qwen3-14B: Blackhole TP1 versus TP2

Subsequent libtt optimization: [bottleneck analysis and updated results](optimization/REPORT.md), including a confirmed ~1.7% TP2 improvement and ~1.65× scaling. The tables below preserve the original baseline.

Both configurations completed the full benchmark on the retained production build. TP2 delivers 1.61–1.62× single-request decode speedup, below the 1.8× target. No additional implementation changes were needed for 14B.

| Workload | TP1 tokens/s | TP2 tokens/s | Speedup |
|---|---:|---:|---:|
| Short prompt (5 tokens) | 20.46 | 33.00 | 1.613× |
| Long prompt (199 tokens) | 19.86 | 32.10 | 1.617× |
| Two concurrent requests, aggregate end-to-end | 36.59 | 56.25 | 1.537× |

| Workload | TP1 TTFT | TP2 TTFT | TP1 decode ms/token | TP2 decode ms/token |
|---|---:|---:|---:|---:|
| Short | 115.91 ms | 76.11 ms | 48.868 | 30.304 |
| Long | 161.42 ms | 105.99 ms | 50.360 | 31.153 |

## Protocol and environment

- Same benchmark script, prompts, and serving flags as the final Qwen3-8B runs. Three warmups then five measured requests per prompt; table shows medians. Each request generates 128 tokens with temperature zero and EOS ignored. Decode timing excludes time to first token and compilation.
- Concurrency measurement: one warmup pair, then one measured pair, both short prompts; aggregate throughput includes prefill and scheduling. It is not a five-run median.
- TP1: PCI 0000:01:00.0. TP2 adds PCI 0000:04:00.0. One active Blackhole chip per P300C board; two boards for TP2. Runs execute sequentially.
- BF16 activations, BF8 weights. Same production wheel as final 8B tests; SHA256 `4b5f5477f0598192b99aa88a58271b40d19dda7f8331ffbb61f1df920a811c66`.
- libtt commit `3f9d1371cf6c67a3c5f7a67c9481a647ac0c8563`, containing implementation commit `bf66648`.
- Unmodified sglang-jax blackhole-multinode commit `6ca38400e6130bc242b215af52b8984bc859036e`.
- Model `Qwen/Qwen3-14B`, revision `40c069824f4251a91eefaf281ebe4c544efd3e18`. Forty layers, hidden size 5120, intermediate size 17408.
- max-running-requests 2, max-total-tokens 1024, max-prefill-tokens/chunked-prefill-size 256, page-size 32. Overlap scheduling and radix cache disabled; stream interval 1.

## Comparison with 8B

| Model | Short TP1 | Short TP2 | Short speedup | Long speedup |
|---|---:|---:|---:|---:|
| Qwen3-8B, final production build | 35.03 | 51.91 | 1.482× | 1.459× |
| Qwen3-14B, same build | 20.46 | 33.00 | 1.613× | 1.617× |

14B scales better, consistent with a larger share of work benefiting from sharding. This benchmark alone does not identify the remaining bottleneck; no new device-kernel profile was collected for 14B.

## Validation and artifacts

All 20 measured single-request generations and four measured concurrent generations completed with 128 output tokens. Each prompt produced identical tokens across its five repeated single-request measurements within each TP setting. TP1 and TP2 outputs are not identical: the short prompt first differs at token index 25 and the long prompt at index 19 (zero-based). Both begin with sensible answers. These checks establish successful generation and repeatability, not numerical equivalence or a quality evaluation.

Each configuration also completed a separate 32-token request with host/Python trace capture. Profiled throughput is excluded from the benchmark medians. Host trace durations overlap and must not be summed as device execution time.

See `comparison.json`, `output-checks.json`, `environment.json`, `model-config.json`, `tp*/requests.json`, `tp*/summary.json`, `tp*/trace-summary.json`, compressed traces, and server/benchmark logs. `launch.sh`, `run_pair.py`, and `benchmark.py` preserve the executed setup; scripts use the original `/tmp/libtt-profile` paths. Both servers were stopped after completion.
