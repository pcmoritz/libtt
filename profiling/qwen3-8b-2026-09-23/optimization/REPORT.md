# Qwen3-8B TP2 optimization in libtt

TP2 now reaches **50.88 tokens/s**, up from **49.06**. The retained change improves short-prompt decode by **3.7%** and achieves **1.46× TP1**. **The 2× target was not reached.**

## Final matched results

Both configurations use the same final wheel and the unmodified `pcmoritz/sglang-jax` branch at `6ca38400e6130bc242b215af52b8984bc859036e`. These are medians of five sequential requests after three warmups, with 128 generated tokens, temperature zero, ignored EOS, and disabled prefix caching. Compilation and profiling are excluded.

| Prompt tokens | TP1 decode tok/s | TP2 decode tok/s | Scaling | TP2 gain over prior build | TP1 TTFT | TP2 TTFT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 34.83 | 50.88 | 1.46× | 3.7% | 78.92 ms | 58.41 ms |
| 199 | 33.77 | 49.03 | 1.45× | 3.4% | 108.18 ms | 78.50 ms |

The exploratory concurrent pair yields 59.73 aggregate output tokens/s for TP1 and 83.12 for TP2 (1.39×). This has only one measured pair per configuration, after one warmup pair.

## Retained change

`third_party/tt_metal/reshaped_mesh_link_discovery.patch`, registered in `MODULE.bazel`, fixes collective link discovery when a logical MeshDevice is reshaped relative to the physical fabric. Previously the helper assumed logical axis 1 was always physical east/west and fell back to one link. It now queries the forwarding direction to the actual logical neighbor before asking for usable routing planes. The final TP2 logs no longer contain the link-discovery fallback.

All performance changes are in libtt’s bundled TT-Metal layer. SGLang is unchanged. The previous libtt mesh-shard and replicated-scalar correctness fixes remain. The packet size and matmul planner retain their original settings. Temporary profiling instrumentation is excluded from the final wheel. `libtt-changes.patch` captures the complete uncommitted code/test diff, including the earlier correctness fixes.

## Validation and profiling

All **10 local-mesh regression tests passed** under the repository’s pinned JAX 0.11.2, including differing per-chip inputs, collective sums, replicated predicates, and trace replay. All five measured continuations for both prompts match each configuration’s original baseline exactly: TP1=True, TP2=True. Generation smoke checks are saved alongside request data; these checks are not a full accuracy evaluation.

A separate diagnostic device profile shows matmuls occupying about 56% of a warmed 500 ms window, with all-gather and reduce-scatter together about 8%. Roughly 1.6 ms/token was spent in collective kernels, while reaching 2× would require saving approximately 5 ms/token overall. Collective tuning alone is therefore insufficient. See [EXPERIMENTS.md](EXPERIMENTS.md) for candidate results, profiling caveats, and the next optimization targets. Raw compressed device-program records and a reproducible analyzer are included. Host/Python traces are also saved for both final configurations; their inclusive times overlap and must not be summed.

## Reproduction

Hardware and dependency details match [the baseline report](../multinode/REPORT.md): one active Blackhole chip per P300C board; TP1 uses `0000:01:00.0`, TP2 additionally uses `0000:04:00.0` on the second board. JAX 0.11.1, Flax 0.12.9, Qwen/Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`, BF16 activations and BF8 weights. Max running requests 2, token pool 1024, prefill/chunk size 256, page size 32, no overlap scheduling.

Build with `/tmp/libtt-profile/bin/bazel build -c opt //:jax_tt_plugin_wheel`, install the wheel into the profiling venv, and run `PROFILE_VARIANT=final ./launch-optimized.sh 2` (or `1`). Then use `python benchmark.py --tp 2 --output results/tp2`. The launcher records the original environment paths; adjust them for another machine. Use a fresh JAX executable cache after compiler changes. The final wheel SHA256 is saved in `results/wheel-sha256.txt`.

Stop the old server completely before switching configurations. One attempted TP1 run overlapped a draining TP2 server and failed initialization; its requests were excluded entirely. The valid rerun checked the server’s `tp_size` before benchmarking. The saved benchmark now enforces that check.
