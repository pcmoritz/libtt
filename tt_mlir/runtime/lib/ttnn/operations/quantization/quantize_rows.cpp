// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include "operations/quantization/quantize_rows.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/operations/experimental/quantize_rows/quantize_rows.hpp"

namespace tt::runtime::ttnn::operations::quantization {

void run(const ::tt::target::ttnn::QuantizeRowsOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->input());
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          ::tt::runtime::ttnn::utils::getTensorRefMemoryConfig(op->output()));
  std::vector<::ttnn::Tensor> outputs = ::ttnn::experimental::quantize_rows(
      input, ::tt::runtime::ttnn::utils::toTTNNDataType(op->float8_type()),
      op->min_scale(), outputMemoryConfig);
  tensorPool.insertTTNNTensorAndValidate(op->output(), outputs[0]);
  tensorPool.insertTTNNTensorAndValidate(op->scale(), outputs[1]);
}

} // namespace tt::runtime::ttnn::operations::quantization
