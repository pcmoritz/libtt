// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include "operations/normalization/l2_clip.h"

#include <optional>

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/operations/experimental/l2_clip/l2_clip.hpp"

namespace tt::runtime::ttnn::operations::l2_clip {

void run(const ::tt::target::ttnn::L2ClipOp *op, ProgramContext &context) {
  ProgramTensorPool &pool = context.getTensorPool();
  const ::ttnn::Tensor &input = pool.getTTNNTensorAndValidate(op->input());
  std::optional<::ttnn::Tensor> scale, rowScale;
  if (op->scale()) {
    scale = pool.getTTNNTensorAndValidate(op->scale());
  }
  if (op->row_scale()) {
    rowScale = pool.getTTNNTensorAndValidate(op->row_scale());
  }
  const ::ttnn::DataType dtype = ::tt::runtime::ttnn::utils::toTTNNDataType(
      op->out()->desc()->layout()->memory_desc()->data_type());
  pool.insertTTNNTensorAndValidate(
      op->out(), ::ttnn::experimental::l2_clip(
                     input, op->group(), op->threshold(), op->epsilon(),
                     op->offset(), op->width(), scale, rowScale, dtype));
}

} // namespace tt::runtime::ttnn::operations::l2_clip
