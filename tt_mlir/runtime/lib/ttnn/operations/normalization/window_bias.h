// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#ifndef LIBTT_RUNTIME_LIB_TTNN_OPERATIONS_NORMALIZATION_WINDOW_BIAS_H
#define LIBTT_RUNTIME_LIB_TTNN_OPERATIONS_NORMALIZATION_WINDOW_BIAS_H

#include "tt/runtime/detail/ttnn/types/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::window_bias {
void run(const ::tt::target::ttnn::WindowBiasOp *op, ProgramContext &context);
} // namespace tt::runtime::ttnn::operations::window_bias

#endif
