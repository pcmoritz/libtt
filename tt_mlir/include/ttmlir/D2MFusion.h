// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

// libtt's extensions of tt-mlir's D2M subgraph fusion (TTNN ops outlined into
// D2M subgraphs and compiled to generated kernels), in pipeline order. Each
// group is called from the upstream pass named in its heading
// (tt_mlir_d2m_fusion.patch); the definitions are in the libtt sources named
// there. Everything here runs only when D2M subgraphs are enabled
// (enable_create_d2m_subgraphs, or LIBTT_D2M_SUBGRAPHS=1).

#ifndef TTMLIR_D2MFUSION_H
#define TTMLIR_D2MFUSION_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>

namespace mlir::tt::ttnn {
class ReshapeOp;
class SliceStaticOp;
} // namespace mlir::tt::ttnn

namespace mlir::tt::d2m {
class GenericOp;
} // namespace mlir::tt::d2m

namespace mlir::tt::d2m_fusion {

//===----------------------------------------------------------------------===//
// TTNNCreateD2MSubgraphs (lib/Dialect/TTNN/Transforms/D2MSubgraphGroups.cpp)
//===----------------------------------------------------------------------===//

// TTNN rewrites before grouping that let D2M take Inkling 2's per-head clips
// whole: heads sliced from a fused projection as a column slice and a head
// split, and typecasts moved across reshapes next to the ops D2M takes.
void prepareForOutlining(ModuleOp module);

// Makes each fusion group with head reshapes one the head-major rewrite
// (finishD2MSubgraphs) takes, or keeps those reshapes on TTNN.
void fixHeadGroups(SmallVectorImpl<SmallVector<Operation *>> &groups);

// Admission helpers: the tiles of a tensor in TTNN's tile layout, the head
// widths the head-major (multiples of 32) and group-major (16) forms handle,
// and slices of whole rows at tile-aligned columns (read in place).
int64_t tileCount(RankedTensorType type);
bool headDimSupported(int64_t width);
// RMS norms over one tile row ([..., T, N] with T <= 32, leading dims 1) of
// more than 32 tiles: the number of chunks k the row is reduced in, one per
// core of a rectangle of op's device ([k, T, N / k], the same tiles in the
// same order).
std::optional<int64_t> wideNormChunks(Operation *op, RankedTensorType type);
bool isTileColumnSlice(ttnn::SliceStaticOp slice);

// A reshape the outliner drops: one element, the same shape once promoted to
// rank 2 (in TTNNCreateD2MSubgraphs.cpp, next to that promotion).
bool isPromotionReshape(ttnn::ReshapeOp reshape);

//===----------------------------------------------------------------------===//
// ConvertTTNNToTTIR (lib/Conversion/TTNNToTTIR/D2MSubgraphs.cpp)
//===----------------------------------------------------------------------===//

// Whether every activation in a TTNN binary op's activation list (null or
// empty included) has a parameterless TTIR unary counterpart, so the
// conversion can expand the list into separate ops; and that expansion (the
// first entry applied first).
bool canExpandTTNNActivations(ArrayAttr activations);
Value applyTTNNActivations(OpBuilder &builder, Location loc, Value value,
                           ArrayAttr activations);

// Conversions of the ops only D2M subgraphs hold: reshapes (head splits and
// merges), tile-aligned column slices and RMS norms (decomposed).
void populateD2MSubgraphPatterns(MLIRContext *ctx, RewritePatternSet &patterns,
                                 TypeConverter &typeConverter);

// After the conversion: scalar constants inlined, per-head subgraphs in
// head-major or group-major form and wide rows over several tile rows, each
// subgraph's row count recorded, identical subgraphs deduplicated. Fails on a
// head reshape no rewrite took.
LogicalResult finishD2MSubgraphs(ModuleOp module,
                                 ArrayRef<func::FuncOp> d2mSubgraphs);

// Per-head subgraphs in decode: T <= 32 tokens, H heads of width D (a multiple
// of 32, or 16) split from [T, H * D] and merged back to it. The outliner
// keeps head reshapes in a group only if the head-major rewrite takes it,
// using these checks.
struct HeadDims {
  int64_t T, H, D;
};
// The dims of heads [T, H, D] merged to [T, H * D], if supported.
std::optional<HeadDims> headMerge(ArrayRef<int64_t> heads,
                                  ArrayRef<int64_t> merged);
// The reshapes into the rank-3 values of such a subgraph.
enum class HeadReshape {
  Split,    // [T, H * D] -> [T, H, D]
  Scale,    // [1, H * D] or [H * D] -> [1, H, D]
  Token,    // the column [T, 1] -> [T, 1, 1]
  TokenRow, // the row [1, T] or [T] -> [T, 1, 1]
  Row,      // a norm weight [1, D] or [D] -> [1, 1, D] (D % 32 == 0)
};
std::optional<HeadReshape>
classifyHeadReshape(ArrayRef<int64_t> in, ArrayRef<int64_t> out, HeadDims dims);
// The rank-3 values such a subgraph computes: [a, H, D], [a, H, 1], [a, 1, 1].
bool isHeadValue(ArrayRef<int64_t> shape, HeadDims dims);

//===----------------------------------------------------------------------===//
// TTIRToD2M (lib/Conversion/TTIRToD2M/D2MAlgebra.cpp)
//===----------------------------------------------------------------------===//

// TTIR rewrites of the subgraphs before their conversion to D2M.
void simplifyForD2M(ModuleOp module);

//===----------------------------------------------------------------------===//
// D2MGridSelection, D2MGenericFusion (lib/Dialect/D2M/Transforms/)
//===----------------------------------------------------------------------===//

// RowMerge.cpp: marks the generics of functions that hold whole rows per core
// (d2m.unsplit_dims), then merges each such function's generics into one.
void annotateRowMergeCandidates(ModuleOp module);
void mergeRowGroups(ModuleOp module);

// ElementwiseDAGFusion.cpp: fuses one producer generic into an elementwise
// consumer, keeping escaping results as outputs; fails if none fuses.
LogicalResult fuseElementwiseProducer(d2m::GenericOp consumer,
                                      PatternRewriter &rewriter);

//===----------------------------------------------------------------------===//
// ConvertD2MToTTKernel (lib/Conversion/D2MToTTKernel/DstResident.cpp)
//===----------------------------------------------------------------------===//

// Whether a compute kernel uses FP32 DST (32-bit CBs): the kernels the two
// rewrites below apply to.
bool usesFp32Dst(func::FuncOp kernel);
// Keeps a whole kernel's values in DST across its phases if they fit (a chunk
// all-reduce's kernel in two sections, split where the partial leaves);
// returns whether it did.
bool keepKernelInDst(func::FuncOp kernel);
// Fuses runs of elementwise SFPU tile ops into one SFPI function each.
void fuseSFPUChains(func::FuncOp kernel);

// After D2MToTTNN (lib/Conversion/D2MToTTKernel/ChunkAllReduce.cpp): each
// program whose compute kernel has a libtt_chunk_allreduce_tile marker sums
// the marked partial (row sums) over its cores inside the kernel: the reader
// gathers every core's row sums on one core, which multicasts them back
// (new CBs and a new semaphore); the compute phase sums them.
LogicalResult insertChunkAllReduce(ModuleOp module);

//===----------------------------------------------------------------------===//
// TTKernelToCpp (lib/Target/TTKernel/SFPUHelpers.cpp)
//===----------------------------------------------------------------------===//

// C++ emitted into kernels: the helpers fused SFPU chains call, D2M's
// pre-334f0f9 tt-metal call names on the new API, and libtt's tile ops.
llvm::StringRef sfpuChainHelpers();
llvm::StringRef binaryCompatHelpers();
llvm::StringRef tileOpHelpers();
// C++ emitted into data-movement kernels that call libtt_*_dm helpers.
llvm::StringRef dataMovementHelpers();

} // namespace mlir::tt::d2m_fusion

#endif // TTMLIR_D2MFUSION_H
