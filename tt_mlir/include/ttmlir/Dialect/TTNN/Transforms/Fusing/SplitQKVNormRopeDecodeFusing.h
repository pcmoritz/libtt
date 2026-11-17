// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#ifndef LIBTT_TT_MLIR_INCLUDE_TTMLIR_DIALECT_TTNN_TRANSFORMS_FUSING_SPLITQKVNORMROPEDECODEFUSING_H
#define LIBTT_TT_MLIR_INCLUDE_TTMLIR_DIALECT_TTNN_TRANSFORMS_FUSING_SPLITQKVNORMROPEDECODEFUSING_H

#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Fusing/SplitQKVNormRopeDecodeFusing.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/Builders.h"

namespace mlir::tt::ttnn::fusing {

// The composite of a decode step's attention prologue, which the TTIR fusion
// ttir::fusing::SplitQKVNormRopeDecodeFusing makes.
using ttir::fusing::kSplitQKVNormRopeDecode;

// The split_qkv_norm_rope_decode of a kSplitQKVNormRopeDecode composite, if
// the layouts TTNNLayout chose for its operands and results suit the kernel;
// otherwise nullptr, and TTNNResolveComposites inlines the composite's
// decomposition.
Operation *buildSplitQKVNormRopeDecode(ttcore::CompositeOp composite,
                                       OpBuilder &builder);

} // namespace mlir::tt::ttnn::fusing

#endif
