// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#ifndef LIBTT_RUNTIME_LIB_TTNN_OPERATIONS_NORMALIZATION_RMS_NORM_GATED_H
#define LIBTT_RUNTIME_LIB_TTNN_OPERATIONS_NORMALIZATION_RMS_NORM_GATED_H

#include "tt/runtime/detail/ttnn/types/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::rms_norm_gated {
void run(const ::tt::target::ttnn::RMSNormGatedOp *op, ProgramContext &context);
} // namespace tt::runtime::ttnn::operations::rms_norm_gated

#endif
