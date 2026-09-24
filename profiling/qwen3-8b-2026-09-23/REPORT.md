Follow-up: TP2 now works with the supplied SGLang branch and local libtt fixes. See [the matched TP1/TP2 report](multinode/REPORT.md). The original main-branch results below are retained for reference.

Latest follow-up: [libtt TP2 optimization results](optimization/REPORT.md).

# Qwen3-8B: TP1 versus TP2 on Blackhole

Tested 2026-09-23. **TP1 works with a one-line SGLang-JAX compatibility patch. TP2 fails before producing tokens; no TP2 throughput or speedup can be reported.** The libtt source was not modified.

## Versions and hardware

- libtt branch `codex/blackhole-multinode`, commit `6b74ca2ab2b86abc85a454460c451995f1b52f62`.
- SGLang-JAX `main`, commit `eb061d8b154056e0f07eef07cd6f98047dfcf4a7`.
- Python 3.12.3, JAX/jaxlib 0.11.1 (SGLang's CPU extra), Flax 0.12.9, Transformers 5.12.1.
- Model `Qwen/Qwen3-8B`, revision `b968826d9c46dd6066d109eabc6255188de91218`.
- Optimized plugin built successfully with `bazel build -c opt //:jax_tt_plugin_wheel` (30.4 minutes).
- Two P300C physical boards, each containing two Blackhole chips. Firmware 19.13.1, KMD 2.10.0. Host CPU power profile: performance.
- TP1 uses one chip, PCI `0000:01:00.0`, on board `000004613193403b`.
- TP2 uses that chip plus PCI `0000:04:00.0` on board `0000046131935046`. These chips have two direct Ethernet channels. This tests one active chip versus two active chips across two boards; it does not use every chip on either P300C.

Full environment and hardware snapshots are in `results/environment.json` and `results/hardware-{before,after}.json`. Ethernet connectivity is in `results/topology.yaml`.

## TP1 performance

Median of five sequential measured requests per prompt, after three warmups per prompt. Each request generates exactly 128 tokens, temperature 0, ignoring EOS. Streaming interval is one token; timing is measured at a localhost HTTP client. Decode rate excludes time to first token. Prefix caching and overlap scheduling are disabled.

| Prompt tokens | First token | Decode tokens/s | Time per output token | End-to-end latency |
| --- | --- | --- | --- | --- |
| 5 | 68.48 ms | 34.65 | 28.86 ms | 3.734 s |
| 199 | 107.85 ms | 33.58 | 29.78 ms | 3.890 s |

The corresponding prefill buckets are 32 and 224 tokens. First-token latency includes request handling, prefill, and first-token sampling; it is not an isolated device prefill time.

One exploratory two-request concurrent batch, following one concurrent warmup batch, produced 256 output tokens in 4.251 s: **60.23 aggregate output tokens/s**. This is one observation, not a repeated concurrency benchmark.

Settings: BF16 activations, TT backend BF8 weight annotation, page size 32, max running requests 2, token pool 1024, max/chunked prefill 256. See `launch.sh` for exact settings. Both model configurations used the same settings except TP size, visible devices, and the explicitly recorded partitioner diagnostic.

Generation sanity checks returned `Paris`, `4`, and `H₂O`. All three pass after Unicode NFKC normalization. These are basic smoke checks, not a model accuracy evaluation. Nine SGLang TT attention metadata tests also passed on CPU.

## Profile

A separate warmed 32-output-token request was captured using `/start_profile` and `/stop_profile`, with host tracing level 2 and Python tracing level 1. Unprofiled requests above supply the throughput measurements.

Chrome trace and XPlane files are under `results/tp1/trace/plugins/profile/2026_09_23_22_55_03/`. `results/tp1/trace-summary.json` contains inclusive host event aggregates.

The trace records 32 scheduler `run_batch` calls totaling 995.7 ms (31.1 ms/call), with 870.0 ms inside the Python `sample` calls. Model dispatch wrappers account for 59.9 ms. These are inclusive host-side timings: asynchronous model work can be charged to a later synchronization in sampling, so these numbers do not establish that the sampling kernel is the bottleneck. Nested events must not be summed.

This plugin advertises a no-op PJRT device profiler, and the Bazel build does not enable Tracy. Therefore these artifacts contain host/Python profiling, not device kernel timings or collective bandwidth measurements.

## Failures and compatibility adjustments

1. **Unmodified SGLang main fails on its declared JAX 0.11.1.** The TT backend passes `optimization_level="1"`; JAX raises `ValueError: '1' is not a valid CompilerEffortLevel` on the first TP1 request. A temporary one-line patch uses `"O1"`, which libtt maps to TT optimization level 1. All TP1 performance results and subsequent TP2 model attempts use this patch. It is saved as `results/sglang-jax-compatibility.patch`.
2. **Single-chip P300 needs an explicit mesh descriptor.** Without it the runtime rejects the CUSTOM cluster type. `tp1.textproto` supplies the required 1x1 mesh.
3. **TP2 with legacy GSPMD (`JAX_USE_SHARDY_PARTITIONER=false`) fails while loading the first Q-projection weight.** Errors: `GSPMD presharded argument missing @Sharding custom call`, missing `@SPMDShardToFullShape`, then `Number of tensor shards must match mesh size`. See `results/tp2-server-o1.log`.
4. **TP2 with Shardy (`true`) loads weights and starts the server, but fails on the first generation.** The weight-annotation reshape reports `number of output elements (622329856) doesn't match expected number of elements (311164928)`. See `results/tp2-server-shardy.log`.
5. **Minimal TP2 reproduction:** Shardy sharded addition fails copying results to host: `Runtime tensor does not match the PJRT shard: 8192 vs 4096 bytes`. This occurs with both JAX 0.11.1 and libtt's test-pinned JAX 0.11.2. The explicit shard-map legacy-GSPMD variant segfaulted. Logs and `tp_smoke.py` are included. JAX 0.11.2 was used only for this diagnostic, then the environment was restored to 0.11.1.

The initial selection `01:00.0,03:00.0` had no direct link and auto-discovery downgraded it to one device. Its apparent arithmetic pass is not a TP2 pass. UMD topology discovery identified the correct cross-board pair `01:00.0,04:00.0`; all subsequent TP2 attempts enumerated two devices. The final smoke script explicitly asserts two devices.

## Reproduction and artifacts

The prepared environment, model cache, patched SGLang checkout, and scripts remain at `/tmp/libtt-profile`. No benchmark servers remain running. Scripts saved beside this report retain those absolute paths.

```bash
# TP1 server; launch.sh supplies the single-chip descriptor
/tmp/libtt-profile/launch.sh 1
# In another terminal
/tmp/libtt-profile/venv/bin/python /tmp/libtt-profile/benchmark.py \
  --tp 1 --output /tmp/libtt-profile/results/tp1-rerun

# TP2 failures, each with its own server invocation
/tmp/libtt-profile/launch.sh 2
PROFILE_USE_SHARDY=true /tmp/libtt-profile/launch.sh 2

# Minimal TP2 shard-copy reproduction
env -u TT_METAL_RUNTIME_ROOT \
  TT_VISIBLE_DEVICES=0000:01:00.0,0000:04:00.0 \
  JAX_PLATFORMS=tt JAX_USE_SHARDY_PARTITIONER=true \
  /tmp/libtt-profile/venv/bin/python /tmp/libtt-profile/tp_smoke.py
```

Raw streaming events, generated text, and per-request metrics are in `results/tp1/requests.json`. Server logs, build log, diagnostic failures, and profile files are preserved under `results/`.
