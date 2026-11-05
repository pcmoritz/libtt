// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental {

// Per-row dynamic quantization of a [..., M, K] tensor (FP32 or BF16, tiled,
// interleaved; one tile row, so M <= 32 and every leading dimension 1; K a
// whole number of tiles) to float8:
//
//   scale[m] = max(float(max_k |input[m, k]|) / fp8_max, min_scale)
//   q[m, k]  = float8(float(input[m, k]) / scale[m])
//
// with fp8_max the format's largest finite value (448 for FP8_E4M3, 57344 for
// FP8_E5M2) and the cast rounded to nearest even and not saturated. Returns
// {q in `dtype`, shaped like the input; scale, [..., M, 1] FP32}. The divides
// and the cast are ttnn.divide's and ttnn.typecast's, so the results equal
// those ops' bit for bit.
//
// One program: each core reads a run of the row's tiles (only their data rows,
// packed into dense tiles), the cores exchange their row maxima over the NoC,
// and each core divides and converts its run.
std::vector<ttnn::Tensor> quantize_rows(
    const ttnn::Tensor& input,
    tt::tt_metal::DataType dtype,
    float min_scale,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::experimental
