// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include "operations/normalization/window_bias.h"

#include <optional>

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/operations/experimental/window_bias/window_bias.hpp"

namespace tt::runtime::ttnn::operations::window_bias {

void run(const ::tt::target::ttnn::WindowBiasOp *op, ProgramContext &context) {
  ProgramTensorPool &pool = context.getTensorPool();
  std::optional<::ttnn::Tensor> sink;
  if (op->sink()) {
    sink = pool.getTTNNTensorAndValidate(op->sink());
  }
  auto outputs = ::ttnn::experimental::window_bias(
      pool.getTTNNTensorAndValidate(op->rel_r()),
      pool.getTTNNTensorAndValidate(op->table()),
      pool.getTTNNTensorAndValidate(op->positions()), op->heads(), sink,
      op->sink_scale());
  for (size_t i = 0; i < op->outs()->size(); ++i) {
    pool.insertTTNNTensorAndValidate(op->outs()->Get(i), outputs.at(i));
  }
}

} // namespace tt::runtime::ttnn::operations::window_bias
