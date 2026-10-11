// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "tt/runtime/detail/ttnn/types/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"
namespace tt::runtime::ttnn::operations::reduction {
void run(const ::tt::target::ttnn::MoeRouterTopKOp *op,
         ProgramContext &context);
}
