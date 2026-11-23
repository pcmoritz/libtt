// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include "operations/normalization/attn_res.h"

#include <optional>
#include <vector>

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/operations/experimental/attn_res/attn_res.hpp"

namespace tt::runtime::ttnn::operations::attn_res {

void run(const ::tt::target::ttnn::AttnResOp *op, ProgramContext &context) {
  ProgramTensorPool &pool = context.getTensorPool();
  std::vector<::ttnn::Tensor> sources;
  for (const auto *source : *op->sources()) {
    sources.push_back(pool.getTTNNTensorAndValidate(source));
  }
  std::optional<::ttnn::Tensor> addend;
  if (op->addend()) {
    addend = pool.getTTNNTensorAndValidate(op->addend());
  }
  const ::ttnn::Tensor &queries = pool.getTTNNTensorAndValidate(op->queries());
  std::optional<::ttnn::Tensor> weight;
  if (op->weight()) {
    weight = pool.getTTNNTensorAndValidate(op->weight());
  }
  const ::ttnn::DataType dtype = ::tt::runtime::ttnn::utils::toTTNNDataType(
      op->out()->desc()->layout()->memory_desc()->data_type());
  pool.insertTTNNTensorAndValidate(
      op->out(), ::ttnn::experimental::attn_res(
                     sources, addend, queries, weight, op->eps_res(),
                     op->eps_norm(), op->normalize(), dtype));
}

} // namespace tt::runtime::ttnn::operations::attn_res
