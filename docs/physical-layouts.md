# Physical layouts

TTNN stores a tiled tensor's last two dims in 32 x 32 tiles, and pads each
slice of them separately. A tensor with a small dim among its last two takes
many times its size, and every op that makes or reads it moves all of that:

| Shape | Example | Allocation |
|---|---|---|
| `[N]` | FSDP2's flat parameter and gradient buffers | 32x |
| `[2, N]` | FSDP2's reduce-scatter input on two chips | 16x |
| `[T, 1, V]` | Megatron-style `[sequence, batch, vocabulary]` logits | 32x |
| `[S, N, H, 1]` | torch_tpu's `[B, S, N, H]` to `[B, N, S, H]` transpose with B = 1 | 32x |
| `[B, N, T, 1, S]` | `jax.nn.dot_product_attention` with one query head per key head | 32x |

Each of these was found as an out-of-memory error or a slow step in one
framework's program, and fixed by a rewrite for that spelling of the shape:
`ttir-dense-flat` for flat buffers and `[T, 1, V]` (#560, not merged), a
permute canonicalization (#541), a `dot_general` decomposition and fusing
pattern (#412), and two fusing patterns that move elementwise ops across
reshapes to the side with fewer tiles (#357). Each rewrite missed the next
spelling: a `[T, 1, V]` tensor made by a reshape inside a program, rather
than passed in, kept its 32x padding (miles' log-probabilities, 4.6 GB).

## Logical shapes, physical layouts

A program's values keep their logical shapes: the shapes the framework wrote,
which the program's ops are defined on. Each value also gets a *physical
layout*: the shape its elements are stored in on device. The compiler chooses
it, rebuilds each op on its operands' physical layouts, and TTNN only ever
sees physical shapes. That is how XLA separates a `Shape` from its `Layout`.

TTNN's own tensor layout can't express this. tt-metal requires the second
innermost alignment of a tiled tensor to be a multiple of the tile height
(`validate_alignment_tile`), so it always pads each slice, and each TTNN op
reads its operands' shapes. tt-mlir's layout attribute has an affine map from
logical to physical dims (`TTNNLayoutAttr::linear`), but the runtime ignores
it, and the attribute's memref (which collapses the leading dims before
padding) undercounts what TTNN allocates. So the physical layout is the shape
TTNN ops receive, and one TTIR pass applies it: `ttir-physical-layouts`
(`tt_mlir/lib/Dialect/TTIR/Transforms/PhysicalLayouts.cpp`), after the TTIR
fusions, which match the logical shapes.

A physical layout of a tensor of logical shape `S` is a shape `P` holding the
same elements in the same row-major order, so changing between them is a
reshape. Besides `S` itself there are two:

- **Unit dims dropped (U):** `P` is `S` without its size-1 dims. `[T, 1, V]`
  is stored as `[T, V]`, `[S, N, H, 1]` as `[S, N, H]`, `[B, N, T, 1, S]` as
  `[B, N, T, S]`.
- **Folded (F):** `P = [n / 32, 32]` for `n` elements, a multiple of 32: rank-1
  tensors and matrices of a few rows, in whole tiles.

## Op rules

An op is *transparent* for a layout kind when it can run on its operands and
result in that kind. For each such op, the pass knows which operands and
results are *members* (stored in the kind like the op's result) and how to
rebuild the op on physical shapes. Any other operand is an *edge*: it is
reshaped to the shape the physical op needs.

| Op | U | F |
|---|---|---|
| elementwise (unary, binary, ternary) | operands of the result's shape are members; broadcast operands are edges without the result's unit dims | operands of the result's shape are members; one-element operands are `[1, 1]` edges |
| `broadcast` | the input is an edge without the result's unit dims | the input has one element |
| reductions (`sum`, `mean`, `max`, `min`, `prod`, `argmax`, `reduce_and`, `reduce_or`) | the reduced dims, mapped, on the input's U layout; the result's U layout is what remains | no |
| `softmax` | its dim, mapped, if it isn't a unit dim | no |
| `permute` | the permutation of the non-unit dims; none if they keep their order | no |
| `slice_static` | the non-unit dims' bounds, if the slice makes no new unit dim | slices of 32-element runs: whole rows, or aligned column blocks |
| `concat` | its dim, mapped, if the inputs and result have the same unit dims | rank 1, rows of a multiple of 32 elements, or columns of each row |
| `all_gather`, `reduce_scatter` | their dim, mapped | rank 1 along dim 0 |
| `all_reduce` | yes | yes |
| `rms_norm`, `layer_norm` | if the normalized dims aren't unit dims | no |
| constants, `full`, `zeros`, `ones` | edges: their reshapes fold away | same |
| `reshape` | either side takes any layout: one reshape between the physical shapes, none if they are equal | same |
| anything else, and ops with regions | operands and results in their logical shapes | same |

## Choosing layouts

Values connected through transparent ops form a *cluster*; an op's cluster
takes a kind, or none. The pass prices a cluster by the time its ops take, in
the time it takes DRAM to move the same bytes (about 400 GB/s):

- An op takes at least its dispatch time, about 1 MB's worth: a traced program
  advances every 2 to 5 µs whatever its size, so padding a decode step's
  one-token rows costs nothing.
- A reshape that moves tiles runs at about 1/32 of DRAM's rate:
  `[256, 4096]` f32 to `[32768, 32]` takes about 350 µs. A reshape that keeps
  the last two dims (including everything for U) is free.
- Tiles of at most 16 rows by at least 32768 columns (a sampler's rows) are
  stored as 16-row tiles (`top_half_tiles.patch`) and counted as such.

A cluster takes the kind when the time its ops save exceeds the time of the
reshapes at its edges. The saving per op is the difference between logical
and stored allocation; an edge's cost is the same reshape time the two shapes
differ by.
U is tried first, then F on the values U didn't take: a fold only pays where
the shape has no unit dims to drop.

Public functions' arguments and results of compact shape keep it, as the
function's signature: rank-1 arrays of at least 2^20 elements, matrices of at
most 8 rows and at least 2^22 elements, and `[T, 1, V]` of at least 2^20
elements take the folded form, respectively `[T, V]`. The runtime stores
buffers by the same rule, which `tt_mlir/include/ttmlir/PhysicalLayouts.h`
gives both (the runtime half of `tt_mlir_physical_layouts.patch`: `toLayout`
reshapes a buffer stored otherwise, and a program input's `LayoutDesc` records
its shape). `LIBTT_PHYSICAL_LAYOUTS=0` turns both off.

Fused ops formed before this pass are edges: the rotary composites
(`ttcore.composite`), the decode attention prologue
(`ttir.split_qkv_norm_rope_decode`, #561), the fused matmul epilogues
(`ttir.fused_matmul`, #563), and the JAX attention prologue (#564). They take
their operands in their logical shapes, and a decomposition is left alone.
XLA fuses after layout assignment; here the fusions that match logical shapes
move before the pass instead.

## Rewriting

Each op of a taken cluster is rebuilt on its own, with its attributes mapped
(dims, permutation, slice bounds, broadcast multipliers, a reduction's
`keep_dim`): its members are reshaped to the kind's shape where they are made,
its edges to the shapes the op needs, and its result back to its logical
shape. Every other op, and any use inside a region, keeps the logical value.
Reshape folding is left to MLIR: a reshape of a reshape merges and a reshape to
the same shape vanishes, so the pair between two rebuilt ops disappears, and
reshapes of constants and fills fold for free. A value that several rebuilt ops
take gets a reshape from each, side by side where it is made, which the CSE
after the TTIR passes merges. Clusters are decided in the order of the first
value, reproducibly.

## Earlier rewrites

`ttir-dense-flat` (#560, not merged) did this for flat buffers and
`[T, 1, V]`: its flat-buffer views are F clusters here, its unit-row views U
clusters, and its boundary rule is the one above.

These stay, though they overlap with the pass:

- #541's permute canonicalization (`tt_mlir_permute_unit_dims.patch`)
  rewrites and removes permutes that move size-1 dims, also on tensors too
  small for the pass to change.
- #412's `permuteOperand` in the `dot_general` decomposition skips a permute
  that only moves size-1 dims. That is a choice of ops, not of layouts, and
  #541's transposed matmul operands build on it.
- #412's `SinkReshapeThroughPermutePattern` and #357's
  `HoistReshapeToFewerTilesPattern` and `SinkReshapeToFewerTilesPattern` move
  ops to the side of a reshape with fewer tiles inside the TTIR fusions, whose
  other patterns match their results (a weightless `rms_norm`, a typecast
  folded into a multiply).
- #521's wide-row reductions, which are kernels.

## Not yet

- **Permuted layouts.** The largest remaining tensors in torchtitan training
  need dims reordered rather than dropped: the fused gate and up projection
  `[T, 2, F]` (as `[2, T, F]`), grouped-query attention's `[T, 8, 2, 128]`,
  and rotary embeddings' `[T, 1, 128]` broadcast against `[T, 8, 128]`. A
  third kind, small dims moved in front, fits the same clusters, with rules
  for ops that read dims.
- **Scatters and gathers.** miles' log-probability gradient scatters into a
  `[256, 1, 151936]` tensor, 4.7 GB padded: scatters have no rule yet.
- **Fusions after the pass.** TTNN's fusions match logical shapes, so
  clusters next to fusions still in TTNN keep their layout. Moving them
  before layout assignment, as XLA fuses after it assigns layouts, would
  lift that limit.
- **Matmul operands.** Decode matmuls take `[B, 1, K]`, which their kernels
  match; matmuls are edges.
- **Control flow.** Values crossing region boundaries keep their logical
  shapes.

## Tests

- `tests/libtt/physical_layouts_test.py`: exported-program shapes for unit
  rows made inside a program, log-softmax, softmax with slices and concats,
  and a transpose with a trailing unit dim.
- `tests/libtt/dense_flat_test.py`,
  `tests/libtt_torch_tpu/fsdp2_flat_buffers_test.py`: the boundary rule, flat
  buffers at program boundaries, FSDP2's flat buffers.
- The decode fusion suites (`swiglu_regression_test.py`,
  `concat_slices_regression_test.py`,
  `elementwise_on_fewer_tiles_test.py`, `qkv_norm_rope_decode_regression_test.py`)
  check that one-token rows keep their shapes and the fused ops survive.
