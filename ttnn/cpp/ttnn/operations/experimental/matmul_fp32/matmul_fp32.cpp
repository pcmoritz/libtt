// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/matmul_fp32/matmul_fp32.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/matmul_fp32/device/matmul_fp32_device_operation.hpp"

namespace ttnn::experimental {

ttnn::Tensor matmul_fp32(
    const ttnn::Tensor& a, const ttnn::Tensor& b, const std::optional<ttnn::MemoryConfig>& memory_config) {
    using Op = prim::MatmulFp32Operation;
    return ttnn::device_operation::launch<Op>(
        {.output_mem_config = memory_config.value_or(a.memory_config())}, {.a = a, .b = b})[0];
}

}  // namespace ttnn::experimental
