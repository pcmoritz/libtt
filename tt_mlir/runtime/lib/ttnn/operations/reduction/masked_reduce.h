// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#ifndef RUNTIME_LIB_TTNN_OPERATIONS_REDUCTION_MASKED_REDUCE_H
#define RUNTIME_LIB_TTNN_OPERATIONS_REDUCTION_MASKED_REDUCE_H

#include "tt/runtime/detail/ttnn/types/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::reduction {
void run(const ::tt::target::ttnn::MaskedReduceOp *op, ProgramContext &context);
} // namespace tt::runtime::ttnn::operations::reduction

#endif
