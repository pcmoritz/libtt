# Qwen3-32B TP2 / TP4 on QuietBox

Decode optimizations for Qwen3-32B on local Blackhole chips. All changes are in
libtt's TT-Metal patch layer; SGLang-JAX and the model are unchanged. Possible
SGLang-JAX-side improvements are listed separately at the end.

## Results

Qwen/Qwen3-32B revision `9216db5781bf21249d130ec9da846c4624c16137`, BF8
weights, BF16 activations. Median decode tokens/s over five 128-token
generations after three warmups (temperature zero, EOS ignored; first token and
prefill excluded). Concurrent is aggregate end-to-end throughput of one
measured pair of short-prompt requests.

| Configuration | TP2 short (5 tok) | TP2 long (199 tok) | TP4 short | TP4 long | TP2 concurrent | TP4 concurrent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline (`908c39b`) | 16.20 | 15.82 | 25.29 | 24.46 | 28.34 | 41.00 |
| This work | **18.41** | **17.98** | **29.54** | **28.45** | 31.68 | 46.96 |
| Change | +13.7% | +13.7% | +16.8% | +16.3% | +11.8% | +14.5% |

TP4/TP2 scaling is 1.60x (short) and 1.58x (long). Median TTFT is unchanged
within noise (TP4 102/142 ms, TP2 149/207 ms).

Quick short-prompt runs attribute the gains roughly as follows (tokens/s):

| Step | TP2 | TP4 |
| --- | ---: | ---: |
| Baseline | 16.4 | 25.2 |
| + TP2/TP4 fused all-reduce, dual-NOC SwiGLU | 17.91 | 28.90 |
| + Decode-stream matmul for QKV, O, down, LM head | 18.43 | 29.61 |

## Changes

### Dual-NOC weight streaming (`matmul_decode_stream.patch`)

Decode matmuls were limited to about 365 GB/s of the 512 GB/s DRAM peak,
independent of K-block size, burst size, bank locality or math: time scaled
linearly with weight bytes, and removing the math did not change it. All
weight reads used one NOC. The fused SwiGLU kernel now splits the weight
stream: RISCV_0 reads even K blocks on its NOC and RISCV_1 reads odd blocks on
the other NOC. TP4 SwiGLU drops from 190 to 153 µs (455 GB/s), and TP2 from
394 to 319 µs.

The activation row block also stays resident in L1. After one readiness
handshake, the sender streams it in chunks of up to eight tiles, overlapping
the DRAM read of the next chunk with the multicast of the current one. A
counting semaphore tells receivers how many chunks have landed.

### Decode-stream matmul (`matmul_decode_stream.patch`)

A new TT-Metal program factory applies the same design to the other decode
projections. It is selected automatically inside `ttnn::prim::matmul` when all
of these hold:

- Blackhole, one tile row of BF16 activations (1–32 logical rows), batch 1;
- BFP8 weights; BF16 or FP32 output;
- all tensors DRAM interleaved;
- no program config, core grid, bias, fused activation, transpose, global CB
  or sub-device;
- K is 8–400 tiles, since the activation stays resident in L1;
- enough output tiles for at least half the worker grid.

`LIBTT_DECODE_STREAM=0` disables it.

- Output tiles are split evenly when a divisor of N uses at least 70% of the
  grid; otherwise every core is used and per-core counts differ by one.
- Wide outputs (the LM head) are processed in groups of up to eight tiles,
  reusing the resident activation.
- Accumulation stays in FP32 DST across all of K. This roughly halves the
  error against an FP32 reference compared with the generic factory.
- When the compute cores leave the last grid row free, two idle cores there
  feed the activation multicast. Otherwise each multicast sender interleaves
  its own odd weight blocks with activation chunks.

| Projection (µs) | Generic | Decode stream |
| --- | ---: | ---: |
| TP4 QKV 5120→2560 | 39.0 | 39.4 |
| TP4 O 2048→5120 | 32.4 | 29.9 |
| TP4 down 6400→5120 | 93.9 | 84.2 |
| TP4 LM head 5120→37984 (FP32) | 548 | 430 |
| TP2 QKV/O 5120→5120, 4096→5120 | 75.7 / 61.4 | 69.6 / 56.0 |
| TP2 down 12800→5120 | 184 | 170 |
| TP2 LM head 5120→75968 (FP32) | ~1095 | 876 |

Remaining cost is about 5 µs fixed per op plus about 450 GB/s marginal
bandwidth; narrow projections with one or two output tiles per core stay near
355–410 GB/s.

### Fused TP2/TP4 all-reduce (`tp_minimal_all_reduce.patch`)

The earlier TP2-only exchange/reduce specialization, which reuses
TT-Metal's `all_reduce_async` with direct DRAM reader and writer kernels, is
restored and extended to four ranks. It replaces reduce-scatter plus
all-gather for decode-sized BF16 sums: one tile row, widths
1,024–8,192 divisible by 1,024, a one-dimensional TP2 or TP4 mesh. It uses FP32
accumulation. At TP4, all-reduce time drops from about 28 µs to about 17–20 µs
per call. `LIBTT_MINIMAL_ALL_REDUCE=0` disables it.

### SwiGLU widths

The fused SwiGLU accepts per-chip intermediate widths 6,400 (TP4) and 12,800
(TP2) on a 10x10 worker grid; other widths keep the 96-core layout.

## Device profile after the changes (TP4, per token)

The device cycle falls from 39.3 to 33.5 ms. The chip-to-chip matmul imbalance
is gone: previously chip 2 spent about 3.4 ms more in matmuls than the others,
and every layer's collective waited for it. Chips now range from 20.3 to
20.7 ms.

| Category | ms/token |
| --- | ---: |
| Decode-stream matmuls (QKV, O, down) | 9.8 |
| Fused SwiGLU | 9.8 |
| LM head | 0.43 |
| Fused all-reduce | 2.2–2.6 |
| Normalization | 1.4 |
| Other small ops (QKV heads, rotary, adds, cache update, sampler) | ~4.2 |
| Attention | 0.54 |
| Devices idle between programs (host) | ~4.3 |

## Remaining bottlenecks

- **Host overhead, about 4 ms/token.** Forward `Execute` spends about 0.2 ms
  validating 839 arguments, 0.4 ms preparing inputs and 0.7 ms submitting.
  Replacing the const-eval cache key's string hashing did not measurably help.
- **Sampler input round trip, about 0.9 ms of device idle.** The sampler is a
  separate traced executable whose `@main` inputs are declared in host memory,
  as are all non-KV-cache inputs. Its logits and greedy predicate are therefore
  read back from the device and uploaded again every token. Declaring such
  inputs in device memory needs matching trace-slot handling: device inputs
  are currently aliased, not copied.
- **Narrow projections** stay near 355–410 GB/s, compared with 450–480 GB/s
  for SwiGLU and the LM head.

## Optional SGLang-JAX changes (not applied)

Measured with the final libtt build; kept separate from the results above.

| Change | TP2 short | TP4 short | Notes |
| --- | ---: | ---: | --- |
| Skip greedy `log_softmax` when logprobs are not requested | 18.73 (+1.6%) | 30.40 (+2.7%) | The greedy branch always computes full-vocabulary log-softmax on 32-row-padded FP32 tiles, about 1 ms/token of device work |
| Overlap scheduler (`--disable-overlap-schedule` removed) | — | 29.33 (−0.9%) | Not beneficial |

Fusing sampling into the forward executable, or giving the sampler
device-resident inputs, would also remove the host round trip described above.
The experiment patch is at
`/tmp/libtt-quietbox/qwen32b-round2/sglang-skip-greedy-logprobs.experiment.patch`.

## Reproduction and validation

Build and launch as in [qwen3-8b-quietbox-tp2.md](qwen3-8b-quietbox-tp2.md),
replacing the model with `Qwen/Qwen3-32B` and adding the revision above:

| TP | TT_VISIBLE_DEVICES |
| --- | --- |
| 2 | `0000:01:00.0,0000:04:00.0` |
| 4 | `0000:01:00.0,0000:02:00.0,0000:03:00.0,0000:04:00.0` |

Serving uses the same settings as the baseline: O1, traced decode, maximum
running requests 2, 1,024-token cache, prefill/chunk size 256, page size 32,
overlap scheduling and radix cache disabled. SGLang-JAX is
`6ca38400e6130bc242b215af52b8984bc859036e` with the pre-existing local
`_select_logits` compatibility fix. JAX/jaxlib is 0.11.1.

Regression tests (September 25, 2026, final wheel):

- Single chip: all 114 libtt tests and 17 JAX smoke tests pass.
- TP2: local-mesh, rotary, SwiGLU, fused all-reduce and decode-stream tests:
  63 passed.
- TP4: rotary, SwiGLU, fused all-reduce and decode-stream tests: 49 passed.
  `local_mesh_regression_test.py` builds a two-chip sub-mesh and fails during
  fabric initialization when four chips are visible, before any kernel runs.
  Run it with two chips visible, as in the TP2 command.

```bash
bazel test -c opt //tests:libtt_test_suite \
  --test_arg=minimal_all_reduce_regression_test.py \
  --test_arg=decode_stream_matmul_regression_test.py \
  --test_arg=swiglu_regression_test.py --test_arg=-xq \
  --test_env=TT_VISIBLE_DEVICES=<2 or 4 chips> \
  --test_env=TT_METAL_OPERATION_TIMEOUT_SECONDS=60 \
  --test_timeout=1800 --test_output=errors
```

The all-reduce test uses every visible chip, so run it with both two and four
chips visible. Its inputs keep every partial sum exact in BF16 for up to four
ranks. Decode-stream tests compare against an FP32 reference within about one
BF16 ulp.

Both full benchmark runs completed all 21 requests per TP size. Measured texts
repeated within each prompt setting. Continuations differ from the baseline in
places because FP32 accumulation changes numerics; they remain coherent. This
is not a model-quality evaluation.

Raw results, wheels, profiles and scripts are under
`/tmp/libtt-quietbox/qwen32b-round2/`.
