// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#ifndef LIBTT_TT_MLIR_RUNTIME_LIB_TTNN_OPERATIONS_TRANSFORMER_NLP_CREATE_QKV_HEADS_DECODE_NORM_ROPE_H
#define LIBTT_TT_MLIR_RUNTIME_LIB_TTNN_OPERATIONS_TRANSFORMER_NLP_CREATE_QKV_HEADS_DECODE_NORM_ROPE_H

#include "tt/runtime/detail/ttnn/types/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::transformer {
void run(const ::tt::target::ttnn::NLPCreateQKVHeadsDecodeNormRopeOp *op,
         ProgramContext &context);
} // namespace tt::runtime::ttnn::operations::transformer

#endif
