// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include "operations/normalization/moe_route.h"

#include <optional>

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/operations/experimental/moe_route/moe_route.hpp"

namespace tt::runtime::ttnn::operations::moe_route {

void run(const ::tt::target::ttnn::MoeRouteOp *op, ProgramContext &context) {
  ProgramTensorPool &pool = context.getTensorPool();
  std::optional<::ttnn::Tensor> numValid;
  if (op->num_valid()) {
    numValid = pool.getTTNNTensorAndValidate(op->num_valid());
  }
  auto outputs = ::ttnn::experimental::moe_route(
      pool.getTTNNTensorAndValidate(op->logits()),
      pool.getTTNNTensorAndValidate(op->bias()), op->k(), op->scale(),
      numValid);
  pool.insertTTNNTensorAndValidate(op->weights(), outputs.at(0));
  pool.insertTTNNTensorAndValidate(op->ids(), outputs.at(1));
}

} // namespace tt::runtime::ttnn::operations::moe_route
