// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#ifndef RUNTIME_LIB_TTNN_OPERATIONS_QUANTIZATION_QUANTIZE_ROWS_H
#define RUNTIME_LIB_TTNN_OPERATIONS_QUANTIZATION_QUANTIZE_ROWS_H

#include "tt/runtime/detail/ttnn/types/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::quantization {
void run(const ::tt::target::ttnn::QuantizeRowsOp *op, ProgramContext &context);
} // namespace tt::runtime::ttnn::operations::quantization

#endif
