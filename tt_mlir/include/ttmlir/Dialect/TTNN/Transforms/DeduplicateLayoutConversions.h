// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#ifndef LIBTT_TT_MLIR_INCLUDE_TTMLIR_DIALECT_TTNN_TRANSFORMS_DEDUPLICATELAYOUTCONVERSIONS_H
#define LIBTT_TT_MLIR_INCLUDE_TTMLIR_DIALECT_TTNN_TRANSFORMS_DEDUPLICATELAYOUTCONVERSIONS_H

#include "mlir/IR/BuiltinOps.h"

namespace mlir::tt::ttnn {

// Replaces every ttnn.to_tensor_spec that converts a value to the same type as
// an earlier one in its block with that earlier conversion. The passes that
// insert layout conversions add one per use and run after the last CSE, so a
// value every layer reads (such as the recurrent state slot of Qwen3.5) was
// converted once per use. Conversions into L1 are kept, so that L1 lifetimes
// stay as the memory planning chose them.
void deduplicateLayoutConversions(ModuleOp module);

} // namespace mlir::tt::ttnn

#endif
