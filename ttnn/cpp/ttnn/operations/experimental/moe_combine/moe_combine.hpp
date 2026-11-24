// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <optional>
#include <string>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental {

// Each token's routing-weighted sum of its experts' outputs, as moe_compute
// leaves them: slots [k * T, H] BF16 row-major in DRAM, row j * T + i token
// i's j-th expert output (zero for experts other devices own), weights
// [T, k] (BF16 or FLOAT32, tiled),
//
//   out[i, :] = sum_j weights[i, j] * slots[j * T + i, :]   accumulated in FP32
//
// [T, H] tiled in dtype, for k * T <= 32. One core per tile column of H: a
// placement tile P[i, j * T + i] = weights[i, j], the column's slot rows as a
// tile, and one matmul P @ slots.
ttnn::Tensor moe_combine(const ttnn::Tensor& slots, const ttnn::Tensor& weights, DataType dtype);

std::optional<std::string> moe_combine_unsupported_reason(const ttnn::Tensor& slots, const ttnn::Tensor& weights);

}  // namespace ttnn::experimental
