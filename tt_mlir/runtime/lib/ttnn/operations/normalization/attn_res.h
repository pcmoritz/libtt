// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#ifndef LIBTT_RUNTIME_LIB_TTNN_OPERATIONS_NORMALIZATION_ATTN_RES_H
#define LIBTT_RUNTIME_LIB_TTNN_OPERATIONS_NORMALIZATION_ATTN_RES_H

#include "tt/runtime/detail/ttnn/types/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::attn_res {
void run(const ::tt::target::ttnn::AttnResOp *op, ProgramContext &context);
} // namespace tt::runtime::ttnn::operations::attn_res

#endif
