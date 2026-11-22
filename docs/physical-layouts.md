# Physical layouts

TTNN stores a tiled tensor's last two dims in 32 x 32 tiles and pads each
slice of them separately. A tensor with a small dim among its last two takes
many times its size:

| Shape | Example | Allocation |
|---|---|---|
| `[N]` | FSDP2's flat parameter and gradient buffers | 32x |
| `[2, N]` | FSDP2's reduce-scatter input on two chips | 16x |
| `[T, 1, V]` | Megatron-style `[sequence, batch, vocabulary]` logits | 32x |

In miles' GRPO trainer these run the chips out of memory: a padded flat
buffer of a 0.6B model's parameters takes 19 GB, a `[256, 1, 151936]` f32
log-probability 4.7 GB.

## The rule

Large tensors of these shapes are stored compactly, holding the same
elements in the same row-major order, so changing between the two shapes is
a reshape:

- rank-1 arrays of at least 2^20 elements, and matrices of at most 8 rows and
  at least 2^22 elements, folded to `[n / 32, 32]` (F);
- `[T, 1, V]` of at least 2^20 elements as `[T, V]` (U), folded too when
  that is a matrix of a few rows, so that a compact shape is its own.

The rule is one function, `compactShape` in
`tt_mlir/include/ttmlir/PhysicalLayouts.h`, and it applies everywhere:

- **Buffers.** The runtime stores device buffers of these shapes compactly
  (`tt_mlir_physical_layouts.patch`): `toLayout` reshapes a buffer into the
  shape a program takes, on device and without a padded copy, and a program
  input's `LayoutDesc` records it.
- **Program boundaries.** `ttir-physical-layouts`
  (`tt_mlir/lib/Dialect/TTIR/Transforms/PhysicalLayouts.cpp`) gives public
  functions' arguments and results of these shapes their compact shapes. It
  marks the arguments it folds (`libtt.folded`), which TTNN's layout pass then
  keeps in tiles, as the runtime stores them, instead of converting them to
  row-major for an embedding or a rows gather.
- **Inside programs.** The same pass rebuilds the ops that have a rule for a
  compact tensor on compact operands.

A tensor's storage follows from its own shape, so there are no layout
decisions to make and none to keep consistent. Smaller tensors, including
everything in a decode step, keep their shapes.

## Rebuilding ops

| Op | F (flat buffers, few-row matrices) | U (`[T, 1, V]`) |
|---|---|---|
| elementwise | into a compact result; one-element operands as `[1, 1]` | into a compact result; operands without their middle dim |
| `broadcast` | of one element into a compact result | |
| `slice_static` | of a compact input: columns between 32-element boundaries, of any rows | of a compact input, in T and V |
| `concat` | into a compact result: stacked, or side by side | into a compact result, along T or V |
| `all_gather`, `reduce_scatter` | rank 1, either side compact | |
| reductions | | of a compact input, over T or V |

The other operands and results of a rebuilt op take the same form whatever
their size: FSDP2 concatenates small parameters into a compact buffer, and
slices them back out of it. Each operand is reshaped where it is made, so its
padded shape lives no longer than before, and the result is reshaped back to
its logical shape. MLIR's reshape folding removes the pairs between rebuilt
ops, and the CSE after the TTIR passes merges the copies a value gets for
several users.

Every other op sees logical shapes. It pays the padding as it did before: a
compact value reshaped to its logical shape is as large as the tensor the
program had there. FSDP2's `dynamic_update_slice` of a shard at a 32-element
boundary of a buffer the rule folds lowers to slices and a concat
(`IndexedRows.cpp`), which run folded. Other updates scatter rows.

`LIBTT_PHYSICAL_LAYOUTS=0` turns off the pass, with its marks, the lowering of
updates to slices and a concat, and the runtime's compact storage of new
buffers. Both read it each time, and the runtime reshapes a
buffer into the shape of the pair a program takes whatever the flag says, so
buffers and programs made on either side of a change work together.

## Not covered

- **Other small dims.** `[T, B, V]` with `B > 1`, the fused gate and up
  projection `[T, 2, F]`, grouped-query attention's `[T, 8, 2, 128]`: their
  small dim is not 1, so no reshape drops it.
- **Ops without a rule**, such as scatters, sorts and top-k, on compact
  tensors.
- **Values crossing control flow.** Only a function's own block is rebuilt.
- **Lengths that aren't a multiple of 32.** A flat buffer or few-row matrix
  whose elements aren't a multiple of 32 has no compact shape and pays the
  full padding, 32 times its size for a flat buffer. One odd piece does it: a
  `[1024]` weight sharded over 3 ranks is 342 elements. The pass logs each
  public function's argument and result that misses the rule for this reason
  alone.

## Tests

- `tests/libtt/physical_layouts_test.py`: flat buffers and few-row matrices
  through FSDP2's patterns (concats, slices, updates, views) and at program
  boundaries; `[T, 1, V]` made inside a program, through reductions,
  log-softmax, and slices and concats, with no padded shape in the exported
  TTNN program; buffers and programs made with the flag set differently.
- `tests/libtt_torch_tpu/fsdp2_flat_buffers_test.py`: FSDP2 on two chips
  against CPU.
