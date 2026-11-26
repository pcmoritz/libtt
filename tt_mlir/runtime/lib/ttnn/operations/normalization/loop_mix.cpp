// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include "operations/normalization/loop_mix.h"

#include <optional>

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/operations/experimental/loop_mix/loop_mix.hpp"

namespace tt::runtime::ttnn::operations::loop_mix {

void run(const ::tt::target::ttnn::LoopMixOp *op, ProgramContext &context) {
  ProgramTensorPool &pool = context.getTensorPool();
  std::optional<::ttnn::Tensor> normWeight;
  if (op->norm_weight()) {
    normWeight = pool.getTTNNTensorAndValidate(op->norm_weight());
  }
  auto outputs = ::ttnn::experimental::loop_mix(
      pool.getTTNNTensorAndValidate(op->dest()),
      pool.getTTNNTensorAndValidate(op->messages()),
      pool.getTTNNTensorAndValidate(op->down()),
      pool.getTTNNTensorAndValidate(op->gain()),
      pool.getTTNNTensorAndValidate(op->forget()), op->epsilon(), normWeight,
      op->eps_norm());
  for (size_t i = 0; i < op->outs()->size(); ++i) {
    pool.insertTTNNTensorAndValidate(op->outs()->Get(i), outputs.at(i));
  }
}

} // namespace tt::runtime::ttnn::operations::loop_mix
