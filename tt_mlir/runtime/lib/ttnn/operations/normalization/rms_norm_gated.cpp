// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include "operations/normalization/rms_norm_gated.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/operations/experimental/rms_norm_gated/rms_norm_gated.hpp"

namespace tt::runtime::ttnn::operations::rms_norm_gated {

void run(const ::tt::target::ttnn::RMSNormGatedOp *op, ProgramContext &context) {
  ProgramTensorPool &pool = context.getTensorPool();
  const ::ttnn::Tensor &input = pool.getTTNNTensorAndValidate(op->input());
  const ::ttnn::Tensor &gate = pool.getTTNNTensorAndValidate(op->gate());
  const ::ttnn::Tensor &weight = pool.getTTNNTensorAndValidate(op->weight());
  const ::ttnn::DataType dtype = ::tt::runtime::ttnn::utils::toTTNNDataType(
      op->out()->desc()->layout()->memory_desc()->data_type());
  pool.insertTTNNTensorAndValidate(
      op->out(),
      ::ttnn::experimental::rms_norm_gated(input, gate, weight, op->epsilon(),
                                           op->gate_offset(), dtype));
}

} // namespace tt::runtime::ttnn::operations::rms_norm_gated
