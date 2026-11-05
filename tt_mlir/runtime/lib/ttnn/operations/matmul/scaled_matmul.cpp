// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include "operations/matmul/scaled_matmul.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/operations/matmul/matmul.hpp"

namespace tt::runtime::ttnn::operations::matmul {

void run(const ::tt::target::ttnn::ScaledMatmulOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &a = tensorPool.getTTNNTensorAndValidate(op->a());
  const ::ttnn::Tensor &b = tensorPool.getTTNNTensorAndValidate(op->b());
  const ::ttnn::Tensor &rowScale =
      tensorPool.getTTNNTensorAndValidate(op->row_scale());
  const ::ttnn::Tensor &colScale =
      tensorPool.getTTNNTensorAndValidate(op->col_scale());
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          ::tt::runtime::ttnn::utils::getTensorRefMemoryConfig(op->out()));
  const ::ttnn::DataType dtype = ::tt::runtime::ttnn::utils::toTTNNDataType(
      op->out()->desc()->layout()->memory_desc()->data_type());
  std::optional<::ttnn::DeviceComputeKernelConfig> computeConfig;
  if (op->compute_config()) {
    computeConfig = ::tt::runtime::ttnn::operations::utils::createDeviceComputeKernelConfig(
        op->compute_config());
  }
  ::ttnn::Tensor out = ::ttnn::matmul_scaled(
      a, b, rowScale, colScale, op->transpose_b(), outputMemoryConfig, dtype,
      computeConfig, op->swiglu());
  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

} // namespace tt::runtime::ttnn::operations::matmul
