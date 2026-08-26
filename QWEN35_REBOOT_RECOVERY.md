# Qwen3.5-9B reboot recovery

Updated: 2026-08-25 (America/Los_Angeles)

## Final direct-state outcome

The real-weight divergence is fixed on this branch. The final design keeps the
known-correct indexed state-pool contract at the JAX and TT-MLIR boundary, but
uses a TTNN-style direct state internally to the runtime operation:

1. gather the selected recurrent-state slot into a batch-sized FP32 tensor;
2. run the direct-state gated-delta kernel, mutating that tensor in place;
3. scatter the updated tensor back into the state pool.

The direct alias never escapes the runtime operation into MLIR SSA or trace
outputs. This is the key difference from the earlier experimental version. Its
recorded trace retained an unsafe external direct-state lifetime: small focused
programs could pass, while full-model capture/replay corrupted recurrent state
and diverged after a few tokens. Keeping the proven state-pool interface makes
the persistent trace boundary explicit while preserving the smaller TTNN
kernel. No additional SGLang-JAX changes are required.

The gather kernel also consumes `has_initial_state`. A false mask zeros the
direct state instead of reading a stale pool slot; a true mask gathers the
indexed continuing state.

Final real-weight validation at revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`:

- cold oracle token IDs: `[11751, 13, 198, 760]`; text: `" Paris.\nThe"`
- cached replay token IDs: `[11751, 13, 198, 760]`; identical text
- cached four-token request: `2.4816 s` end to end
- warm 32-token request: `4.2523 s` end to end, including prefill
- steady decode intervals: `13.83` to `13.86` token/s
- final cold request, including compilation: `342.0885 s`

Focused hardware validation:

- 24-layer traced versus untraced chain: bit-exact outputs and recurrent states
  for three consecutive advances
- fresh-state mask versus NumPy: recurrent max error `1.65e-4`, output max
  error `3.81e-6`, output correlation `0.999983`
- final plugin build: passed in `91.150 s` with `1,417` action-cache hits

The indexed parent remains a useful correctness baseline and is summarized
below.

## Validated indexed parent outcome (historical baseline)

Qwen3.5-9B now decodes correctly and quickly on the P150. The default compiler
path uses the repaired `ttnn.gated_delta_decode` fusion for all 24 recurrent
layers. Full prefill tracing remains disabled because it takes more than 600
seconds to construct, while the much smaller one-token decode specialization
uses tracing and persistent recurrent-state aliases.

Real weights at revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a` exactly
matched the CPU/native four-token oracle for `The capital of France is`:

- token IDs: `[11751, 13, 198, 760]`
- text: `" Paris.\nThe"`
- temperature: 0

Warm real-weight performance:

- prefill-only baseline: `2.1081 s`
- first warm 32-token request: `6.3818 s`
- repeated warm 32-token request: `4.2743 s`
- repeated requests returned identical 32-token sequences
- steady server decode intervals: `14.00` to `14.12` token/s
- baseline-adjusted decode rate: about `14.8` token/s
- prior correct but untraced decode rate: about `1.06` token/s

The decode-only trace is therefore about 14.0 times faster than the untraced
control. The first real-weight request took `335.7426 s`, including ordinary
prefill compilation and the one-time decode trace capture. This replaces the
failed full-model trace configuration, which produced no response in 600
seconds.

## Repository state

- Workspace: `/home/pcmoritz/libtt`
- Branch: `agent/qwen35-9b-ttnn-direct-state-experimental`
- Base: `origin/main` at `0ff8ad0`
- Keep commits on this branch focused on the validated Qwen3.5 implementation.
- Preserve unrelated and user-owned dirty or untracked files.

Tracked files modified by this work:

- `MODULE.bazel`
- `third_party/tt_metal/trace_allocation_state.patch`
- `third_party/tt_xla/tt_mlir_scatter.patch`
- `third_party/tt_xla/tt_mlir_validate_trace_allocation_state.patch`
- `third_party/tt_xla/tt_mlir_while_loop.patch`

New implementation patches:

- `third_party/tt_metal/dynamic_slice_optional_metadata.patch`
- `third_party/tt_metal/recurrent_dataflow_utils.patch`
- `third_party/tt_metal/gated_delta_decode_raw_gating.patch`
- `third_party/tt_metal/causal_conv1d_update.patch`
- `third_party/tt_metal/state_pool_update.patch`
- `third_party/tt_metal/gated_delta_decode_direct_state.patch`
- `third_party/tt_xla/tt_mlir_dynamic_slice_index_type.patch`
- `third_party/tt_xla/tt_mlir_fuse_gated_delta_rule.patch`
- `third_party/tt_xla/tt_mlir_normalization_fusions.patch`
- `third_party/tt_xla/tt_mlir_raw_gating_fusion.patch`
- `third_party/tt_xla/tt_mlir_trace_preserve_output_aliases.patch`
- `third_party/tt_xla/tt_mlir_fuse_matmul_swiglu_dtype_guard.patch`
- `third_party/tt_xla/tt_mlir_paged_update_cache_row_major.patch`
- `third_party/tt_xla/tt_mlir_gated_delta_decode_internal_direct_state.patch`

A local untracked `tt_mlir_safe_gdn_decode_fusion_default.patch` may exist as a
historical artifact, but it is intentionally excluded from this branch and
`MODULE.bazel` does not register it. The repaired fusion is enabled by default.

The validated SGLang-JAX compatibility and test-harness diff is stored in
`.qwen35-sglang-test-harness.patch`. It was regenerated from the exact tested
checkout and passes `git apply --reverse --check` there.

## Fused decode correctness fix

The compiler recognizes the raw decode gating graph and replaces all 24
recurrent-layer subgraphs with `ttnn.gated_delta_decode`. The kernel computes
sigmoid beta gating, stable softplus, decay, and the recurrent update on device.
The resulting TTNN graph has:

- `gated_delta_decode`: 24
- `exp`: 104 to 0
- `log1p`: 24 to 0
- `abs`: 24 to 0
- `where`: 26 to 2
- `add`: 229 to 125
- `multiply`: 195 to 171
- `neg`: 80 to 0

The numerical failure had a concrete ABI-layout cause. The reader and writer
kernels each declare six fixed compile-time arguments:

`kt`, `vt`, `batch_size`, `num_heads`, `workers_per_head`, `num_workers`

Their `TensorAccessorArgs` metadata incorrectly began at slot 5, shifting every
tensor descriptor by one slot. Both offsets now begin at slot 6 in
`third_party/tt_metal/gated_delta_decode_raw_gating.patch`.

Before this fix the recurrent state remained unchanged and output correlation
was effectively zero. After the fix, three sequential production-form fused
decode steps produced:

- step 1: recurrent max error `7.10e-4`; output max error `9.16e-5`;
  output correlation `0.999978`
- step 2: recurrent max error `1.43e-3`; output max error `1.22e-4`;
  output correlation `0.999969`
- step 3: recurrent max error `2.02e-3`; output max error `1.83e-4`;
  output correlation `0.999959`

Both recurrent-state column halves update correctly. The same probe passed with
tracing enabled.

A production-shape prefill-to-fused-decode handoff also passed:

- prefill state: max error `1.16e-4`, correlation `0.999994`
- decode state: max error `3.00e-4`, correlation `0.999975`
- decode output: max error `1.14e-5`, correlation `0.999970`
- state change: max error `2.98e-4`, correlation `0.999928`

The newest focused runtime IR contains `ttnn.gated_delta_decode`, confirming the
validated default path actually lowers to the fused operation.

## Decode-only tracing

Tracing the full hybrid model includes the large prefill graph and remained at
100 percent host CPU for more than 600 seconds. A persistent compilation cache
did not produce an artifact or improve that attempt.

The SGLang harness now exposes two compiler-option sets:

- prefill JIT: `enable_trace=false`, BF16 hybrid model matrices
- decode JIT: `enable_trace=true`, BF16 hybrid model matrices

`ModelRunner` constructs two JIT wrappers around the same model implementation
and selects the decode wrapper only when `forward_mode.is_decode()`. The optional
single AOT dispatcher falls back to stock JIT when decode-specific options are
present. This avoids forcing prefill through tracing while retaining direct
trace replay for every one-token recurrent step.

The historical trace lifetime repair is still required. The original decode
trace copied 48 donated recurrent outputs into separate output slots and copied
them back before replay, violating the donation alias contract.
`third_party/tt_xla/tt_mlir_trace_preserve_output_aliases.patch` detects
`tf.aliasing_output`, preserves donated input slots as outputs, and prevents
recurrent deallocation inside replay. Persistent argument attributes survive
layout decomposition, while capture-local warmup buffers retain normal cleanup.

The allocation-state validation patches snapshot live overlaps after capture,
replay only a safe trace variant, release obsolete variants, and recapture when
the allocation state changes.

## Build and validation

Final plugin build:

- command: `bazel build //:jax_tt_plugin_wheel`
- result: passed
- clean-patch elapsed after dependency rematerialization: `765.090 s`
- clean-patch actions run: 1,056
- cached actions reused: 5,756
- immediate incremental rebuild: `0.761 s`, one internal action
- wheel: `bazel-bin/jax_tt_plugin-0.1.0-py3-none-linux_x86_64.whl`
- installed environment: `/tmp/sglang-jax-qwen35/.venv`
- JAX: 0.8.1
- NumPy: 2.2.6

Real-weight model details:

- model: `Qwen/Qwen3.5-9B`
- revision: `c202236235762e1c871ad0ccb60c8ee5ba337b9a`
- four cached safetensor shards, about 19.3 GB
- load summary: `consumed=427, skipped=348, missing=0, unexpected=0`
- device: `TTDevice(id=0, arch=Blackhole)`

Additional real-device checks that passed:

- focused traced fused decode across three state advances
- production prefill and recurrent-state handoff into fused decode
- paged KV update with exact updated element count and sum
- production-shape paged full-attention versus NumPy GQA: max error `0.00612`,
  correlation `0.999933`
- BF16 SwiGLU versus NumPy: max error `7.69e-5`, correlation `0.999995`
- exact four-token real-weight server oracle
- two identical 32-token warm real-weight generations

## Temporary test setup

The tested checkout is `/tmp/sglang-jax-qwen35`, based on current SGLang-JAX
main plus the compatibility patch. If `/tmp` is lost, recreate the checkout and
virtual environment, then apply `.qwen35-sglang-test-harness.patch`.

Validated real-weight launch:

```bash
JAX_USE_SHARDY_PARTITIONER=false \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
/tmp/sglang-jax-qwen35/.venv/bin/python -u -m sgl_jax.launch_server \
  --model-path Qwen/Qwen3.5-9B \
  --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --device tt \
  --dtype bfloat16 \
  --attention-backend tt \
  --gdn-prefill-impl chunked_jax \
  --host 127.0.0.1 \
  --port 32000 \
  --max-running-requests 1 \
  --max-total-tokens 128 \
  --max-prefill-tokens 32 \
  --chunked-prefill-size 32 \
  --page-size 32 \
  --max-recurrent-state-size 1 \
  --watchdog-timeout 1200 \
  --disable-radix-cache \
  --disable-overlap-schedule \
  --disable-precompile \
  --skip-server-warmup \
  --decode-log-interval 8
```

Oracle request:

```bash
curl http://127.0.0.1:32000/generate \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"The capital of France is\",\"sampling_params\":{\"temperature\":0,\"max_new_tokens\":4}}"
```

The first request incurs compilation. Measure warm throughput with repeated
requests of the same padded shape and `ignore_eos=true`.

## Direct-state implementation decision

The first TTNN-upstream-style attempt exposed the batch-sized direct state as a
public fused-op operand and mutated it across trace capture and replay. The
compute kernel itself was identical to the indexed kernel, and isolated
three-step probes passed, but full real-weight decoding returned token `32`
instead of `760`. Explicitly returning the state alias and several capture,
restore, and warm-replay variants changed the wrong tokens without restoring
correctness. A second fresh request also failed on its first token, proving that
the recorded external direct-state graph itself was unsafe.

The final composite removes that unsafe contract. Public MLIR remains indexed;
TTNN performs gather, direct mutation, and scatter entirely inside one runtime
operation. This is both smaller and safer than teaching JAX, SGLang-JAX, MLIR
alias analysis, and trace ownership about a new persistent direct-state value.
It also retains essentially all performance: `13.83` to `13.86` token/s versus
about `14` token/s for the indexed baseline.

The active overlays are:

- `third_party/tt_metal/gated_delta_decode_direct_state.patch`
- `third_party/tt_xla/tt_mlir_gated_delta_decode_internal_direct_state.patch`
- the existing allocation-state and output-alias trace repairs

The superseded public direct-SSA and scratch trace overlays are intentionally
not registered in `MODULE.bazel` and were removed from the branch. Recoverable
copies from this session are in
`/tmp/libtt-superseded-direct-overlays-20260825`.

## Remaining optional work

1. If startup latency matters, investigate persistent caching or precompilation
   for the separate prefill and decode executables. Do not restore full-prefill
   tracing without first bounding its compile time.
2. Run a larger README-style multi-request benchmark for an apples-to-apples
   comparison with the Qwen3-8B reference.
