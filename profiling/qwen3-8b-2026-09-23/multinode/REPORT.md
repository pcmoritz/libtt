# Qwen3-8B Blackhole TP1 / TP2 comparison

Latest follow-up: [libtt TP2 optimization results](../optimization/REPORT.md).

**TP2 works with `pcmoritz/sglang-jax:codex/blackhole-multinode` plus local libtt fixes. It delivers 1.41–1.42× the single-request decode throughput of TP1.** This follow-up supersedes the failed TP2 attempt on SGLang main.

## Matched measurements

Both configurations use the same optimized plugin, SGLang commit, JAX/Flax versions, Shardy partitioner, model weights, and serving settings. Each row is the median of five sequential requests after three warmups. Every request generates 128 tokens at temperature zero, with EOS ignored and prefix caching disabled.

| Prompt tokens | TP1 decode tok/s | TP2 decode tok/s | Speedup | TP1 first token | TP2 first token | TP1 total latency | TP2 total latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 34.50 | 49.06 | **1.42×** | 78.81 ms | 59.05 ms | 3.760 s | 2.648 s |
| 199 | 33.64 | 47.43 | **1.41×** | 107.82 ms | 81.71 ms | 3.883 s | 2.760 s |

Decode time per output token falls from 28.99 to 20.38 ms for the short prompt, and from 29.72 to 21.08 ms for the longer prompt. First-token latency includes HTTP handling, prefill, and first-token sampling; it is not isolated kernel time.

An exploratory concurrent pair, after one concurrent warmup pair, yields **59.93 aggregate output tok/s for TP1 versus 80.81 for TP2 (1.35×)**. There is only one measured concurrent pair per configuration; the sequential results above have five repetitions.

Raw requests, streaming timestamps, output token IDs, and summaries are in `results/tp{1,2}/`. `results/comparison.json` contains exact values.

## Hardware and software

- Two P300C boards, each containing two Blackhole chips; firmware 19.13.1 and KMD 2.10.0.
- TP1: one chip at `0000:01:00.0`, board `000004613193403b`.
- TP2: that chip plus `0000:04:00.0`, board `0000046131935046`, connected by two Ethernet channels. **One active chip per board**, not both chips of each P300C.
- libtt base: `6b74ca2ab2b86abc85a454460c451995f1b52f62`, with the uncommitted fixes described below.
- SGLang: `6ca38400e6130bc242b215af52b8984bc859036e`, unmodified checkout of the user-specified branch.
- Python 3.12.3; JAX/jaxlib 0.11.1; Flax 0.12.9; Transformers 4.57.6. Full package freeze and plugin SHA256 are saved under `results/`.
- Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`; BF16 activations and TT backend BF8 weight annotation.
- Shardy enabled for both runs; host performance power profile. TP1 measured after TP2, with separate fresh JAX compilation caches for the repaired build.
- Max running requests 2, token pool 1024, max/chunked prefill 256, page size 32, stream interval 1; overlap scheduling and radix caching disabled. Prefill buckets are 32 and 224 tokens.

JAX 0.11.2 was initially attempted for serving, but Flax 0.12.9/0.12.8 expect its removed `HiPrimitive` name; Flax 0.12.5 expects the removed `jax.core.Effect`. Serving therefore uses the branch's declared JAX 0.11.1. The final libtt regression tests also pass independently under the repository's pinned JAX 0.11.2 (without Flax).

## Changes needed in libtt

The supplied SGLang branch resolves the previous FFI sharding and sampling compilation failures. The first run then failed while reading a replicated trace predicate: `Can't get a single buffer from host storage distributed over mesh shape MeshShape([1, 2])`.

The local libtt fixes are:

1. Extend existing shard-boundary handling from multi-process execution to computations targeting multiple devices in one process. This preserves local input annotations and prevents returning a global tensor as each PJRT shard.
2. Read replicated scalar host buffers locally and check that their values agree, rather than assuming a single local buffer.
3. Apply the same replica agreement check to control-flow scalar reads, removing their one-shard restriction.

`MODULE.bazel` registers the two new patches under `third_party/tt_xla/`; control-flow handling is changed in `tt_mlir_control_flow_runtime.cpp`. `libtt-fixes.patch` records all code and test changes. No commits were created.

## Validation

- Optimized wheel build passes.
- Four new numerical regressions pass: inferred and explicit shard-local outputs; conditional execution with tracing off/on, alternating predicates through warmup, capture, and replay.
- The same four tests pass through `//tests:libtt_test_suite` using Python 3.13.9 and JAX 0.11.2. The fixture scopes Shardy to these tests; the existing harness's default remains unchanged. Final log: `results/bazel-test-details.log`.
- Nine SGLang TT attention metadata tests pass.
- TP1 and TP2 each complete all 21 benchmark/profile requests, plus three basic generation checks. Both return `Paris`, `4`, and `H₂O`; Unicode normalization is used for the latter.
- All five sequential measured continuations are identical within each configuration. Continuations differ between TP1 and TP2. These checks establish basic functionality and repeatability, **not full model accuracy or numerical equivalence**; no MMLU evaluation was run.
- `git diff --check` passes.

The first Bazel attempt forced legacy GSPMD through the existing runner and crashed; the final regression fixture explicitly selects Shardy and restores the prior setting afterward. The failed diagnostic log is retained.

## Profile interpretation

Each configuration has a separate warmed 32-token request captured through SGLang's `/start_profile` and `/stop_profile`, with host level 2 and Python level 1. Measurement requests were not profiled.

| Inclusive host event, 32 calls | TP1 total | TP2 total |
| --- | ---: | ---: |
| Scheduler `run_batch` | 1010.2 ms | 757.2 ms |
| `sample` | 882.4 ms | 591.6 ms |
| Model dispatch wrapper | 59.8 ms | 93.6 ms |

These timings overlap and include asynchronous synchronization. In particular, time spent inside sampling can include waiting for the model; it does not establish that the sampling kernel dominates. TP2 has more host dispatch time but lower total batch time in this sample.

The plugin provides a no-op PJRT device profiler, and this build does not enable Tracy. **Profiles contain host/Python events, not device kernel timings.** Chrome trace (`*.trace.json.gz`) and XPlane (`*.xplane.pb`) artifacts are under `results/tp1/trace/` and `results/tp2/trace/`, with event summaries alongside them. TT runtime also logged a one-link fallback for some all-gathers despite two physical channels; that message alone does not identify the scaling bottleneck.

## Reproduce

The environment remains at `/tmp/libtt-profile/multinode-venv`, the SGLang checkout at `/tmp/libtt-profile/sglang-multinode`, and the model cache at `/tmp/libtt-profile/hf`. No benchmark servers remain running. Scripts beside this report preserve these paths.

```bash
# Run each configuration separately; stop the server before switching.
/tmp/libtt-profile/launch-multinode.sh 1
/tmp/libtt-profile/launch-multinode.sh 2

# Against the running server, using its corresponding TP value:
/tmp/libtt-profile/multinode-venv/bin/python /tmp/libtt-profile/benchmark.py \
  --tp 2 --output /tmp/libtt-profile/multinode-results/tp2-rerun

# Focused local two-chip regression tests:
/tmp/libtt-profile/bin/bazel test -c opt //tests:libtt_test_suite \
  --test_arg=-k --test_arg=local_mesh \
  --test_env=TT_VISIBLE_DEVICES=0000:01:00.0,0000:04:00.0 \
  --test_env=TT_METAL_OPERATION_TIMEOUT_SECONDS=120 \
  --test_output=errors --nocache_test_results
```

TP1's launch script supplies `tp1.textproto`, which is required when exposing only one chip of a P300 board. SGLang's compiler IR exports remain at `/tmp/qwen-full-ir` and `/tmp/qwen-sampler-ir`; the saved request traces and logs are the durable profiling artifacts in this report directory.
