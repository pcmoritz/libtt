// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#ifndef RUNTIME_LIB_TTNN_OPERATIONS_DATA_MOVEMENT_INDEXED_ROWS_H
#define RUNTIME_LIB_TTNN_OPERATIONS_DATA_MOVEMENT_INDEXED_ROWS_H

#include "tt/runtime/detail/ttnn/types/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::data_movement {
void run(const ::tt::target::ttnn::RowsGatherOp *op, ProgramContext &context);
void run(const ::tt::target::ttnn::RowsScatterOp *op, ProgramContext &context);
} // namespace tt::runtime::ttnn::operations::data_movement

#endif
