// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#ifndef LIBTT_TT_MLIR_RUNTIME_LIB_TTNN_OPERATIONS_TRANSFORMER_SPLIT_QKV_NORM_ROPE_DECODE_H
#define LIBTT_TT_MLIR_RUNTIME_LIB_TTNN_OPERATIONS_TRANSFORMER_SPLIT_QKV_NORM_ROPE_DECODE_H

#include "tt/runtime/detail/ttnn/types/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::transformer {
void run(const ::tt::target::ttnn::SplitQKVNormRopeDecodeOp *op,
         ProgramContext &context);
} // namespace tt::runtime::ttnn::operations::transformer

#endif
