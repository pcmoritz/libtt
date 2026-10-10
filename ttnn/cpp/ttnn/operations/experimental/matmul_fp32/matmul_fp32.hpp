// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental {

// a @ b for FLOAT32 TILE tensors, computed on the vector unit with FP32
// multiply-adds: much slower than ttnn::matmul, whose matrix unit reads FP32
// inputs as TF32, but FP32 accurate. a is [..., M, K]; b is [K, N] (broadcast
// over a's batch) or has a's batch dimensions.
ttnn::Tensor matmul_fp32(
    const ttnn::Tensor& a,
    const ttnn::Tensor& b,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::experimental
