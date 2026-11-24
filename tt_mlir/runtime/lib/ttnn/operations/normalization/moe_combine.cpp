// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include "operations/normalization/moe_combine.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/operations/experimental/moe_combine/moe_combine.hpp"

namespace tt::runtime::ttnn::operations::moe_combine {

void run(const ::tt::target::ttnn::MoeCombineOp *op, ProgramContext &context) {
  ProgramTensorPool &pool = context.getTensorPool();
  ::ttnn::Tensor out = ::ttnn::experimental::moe_combine(
      pool.getTTNNTensorAndValidate(op->slots()),
      pool.getTTNNTensorAndValidate(op->weights()),
      utils::getDataType(op->out()));
  pool.insertTTNNTensorAndValidate(op->out(), out);
}

} // namespace tt::runtime::ttnn::operations::moe_combine
