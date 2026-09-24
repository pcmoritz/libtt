# TP1/TP2 fusion audit: Qwen3-8B

The portable TP1 fusions are already active in TP2. The meaningful batch-one difference is **72 matmul-plus-residual fusions**: TP1 uses `ttnn.linear`, while TP2 must reduce the partial projections across chips before adding the residual. This requires a collective-aware fusion, not simply enabling the existing matmul/bias rewrite on TP2.

No production compiler/kernel changes were made for this audit. The installed wheel remains `a9426eb5435aad09fb355020a44801031437de6f74d95275db0f6d78de2194e1`, and the SGLang checkout remains unmodified.

## Actual emitted decode graphs, batch one

Counts are in the executable trace body, excluding constant evaluation, allocation helpers, and host wrappers. TP2 counts describe the per-chip SPMD program, not the sum across both chips.

| Existing fusion or specialization | TP1 | TP2 |
| --- | ---: | ---: |
| Shared QKV projection | 36 | 36 |
| Shared up/gate projection | 36 | 36 |
| Matmul + SwiGLU epilogue | 36 | 36 |
| QKV head splitting | 36 | 36 |
| RMSNorm | 145 | 145 |
| RoPE | 72 | 72 |
| LM-head transpose folded into matmul | 1 | 1 |
| Matmul + residual (`ttnn.linear`) | **72** | **0** |
| Separate residual additions after all-reduce | 0 | **72** |

Both graphs have 145 total projection operations (`matmul` plus `linear`). All projection weights remain BF8. RMSNorm's explicit compute configuration matches: HiFi4, approximation disabled, FP32 accumulation enabled, packer accumulation enabled.

The relevant local weight shapes confirm that these are the intended projections:

| Projection | TP1 weight shape | TP2 local weight shape | Count |
| --- | --- | --- | ---: |
| QKV | 4096 x 6144 | 4096 x 3072 | 36 |
| Up/gate, combined | 4096 x 24576 | 4096 x 12288 | 36 |
| Attention output | 4096 x 4096 | 2048 x 4096 | 36 |
| MLP down | 12288 x 4096 | 6144 x 4096 | 36 |
| LM head, transposed by matmul | 151936 x 4096 | 75968 x 4096 | 1 |

The runtime SwiGLU kernel's shape guards accept both local shapes: its half-output tiles per core are four for TP1 and two for TP2. The runtime sharded-RMSNorm eligibility also matches for the 73 hidden-width normalizations: local width 4096 and a small row count. The 72 Q/K head normalizations have width 128 on both configurations. Runtime eligibility was checked against source and emitted layouts; this audit does not provide new per-kernel timing measurements.

## Why the residual epilogue differs

Actual TP1 decode structure, twice per layer:

```text
linear(activation, weight, residual) -> RMSNorm
```

Actual TP2 structure, twice per layer:

```text
local matmul -> reshape -> all_reduce(sum) -> reshape -> add(residual) -> RMSNorm
```

The audit follows SSA operands through reshapes and confirms all 72 TP2 chains. The existing `MatmulWithBiasFusionPattern` in TTIR fusing follows reshapes but does not cross an all-reduce. That is correct: independently adding the replicated residual inside each rank's matmul would produce `sum(partials) + 2 * residual`. Dividing the residual or injecting it on one rank would change the floating-point evaluation order and would need separate numerical and performance validation. Neither transformation was made.

The applicable next optimization is a collective-aware residual epilogue, or a post-reduction residual/RMSNorm fusion that preserves the required residual output. The current TP1 linear fusion cannot simply be reused unchanged.

## Batch-two and prefill cross-checks

Both configurations were also exercised with two concurrent requests. Their batch-two decode graphs agree:

- 36 shared QKV and 36 shared up/gate projections.
- 36 fused SwiGLU operations and 145 RMSNorm operations.
- No QKV-head or RoPE fusion, and no matmul-plus-residual linear epilogues on either TP1 or TP2.

These are shared batch-size limitations, not TP2-specific regressions. Extending the existing decode patterns to batch two is a separate opportunity for both configurations; it does not explain the batch-one scaling gap.

The observed prefill bucket (five prompt tokens padded to 32) also has matching fusion coverage: shared projections, RMSNorm, standalone SiLU, and the folded LM-head transpose. Neither prefill graph uses the decode-only SwiGLU, QKV-head, or RoPE fusions. This audit does not cover every possible prefill length or batch size.

## Validation and reproduction

Both configurations completed two eight-token single-request generations and a pair of concurrent sixteen-token generations. The single-request output-token sequences match the first eight tokens of each configuration's previously validated result. Concurrent runs verify execution and graph capture; they are not an accuracy evaluation or a throughput benchmark.

`sitecustomize.py` is a diagnostic-only JAX `jit` wrapper. It adds the supported `export_path` compiler option while preserving the existing optimization settings. With the serving venv and launcher from the parent directory:

```bash
PYTHONPATH="$PWD/profiling/qwen3-8b-2026-09-24/fusion-audit" \
LIBTT_AUDIT_EXPORT=/tmp/libtt-profile/fusion-audit/tp1 \
PROFILE_VARIANT=fusion-audit \
/tmp/libtt-profile/launch-optimized.sh 1
```

Use `request.py EXPORT_DIR TP_SIZE`, then `request_pair.py EXPORT_DIR`. Stop the server fully before repeating with TP2. Raw exported compiler graphs are compressed under `graphs/tp1/irs` and `graphs/tp2/irs`. Their source paths and hashes are in `graph-manifest.json`; representative SSA chains are in `comparison.json`. Server metadata and generation outputs are saved alongside the graphs. All audit servers were stopped afterward.

The saved graphs can be checked without hardware:

```bash
python3 profiling/qwen3-8b-2026-09-24/fusion-audit/audit_fusions.py \
  profiling/qwen3-8b-2026-09-24/fusion-audit/graphs/tp1 \
  profiling/qwen3-8b-2026-09-24/fusion-audit/graphs/tp2 \
  --check
```

The assertions check the batch-one expected fusion counts, projection weight shapes and BF8 dtype, folded transpose, RMSNorm precision, all 72 TP2 residual/reduction chains, and portable-fusion parity across the captured prefill and batch-one/batch-two graphs. They pass on the saved artifacts. These are structural checks of the emitted program; matching fusion counts do not imply identical kernel efficiency across local shapes.
