// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/experimental/matmul_fp32/device/matmul_fp32_device_operation_types.hpp"

namespace ttnn::experimental::prim {

// Splits the output tiles evenly over the cores. Each core computes its tiles
// on the vector unit in runs along an output row, reading the run's A row
// once; B streams one tile per k.
struct MatmulFp32ProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const MatmulFp32Params&, const MatmulFp32Inputs&, std::vector<Tensor>&);
};

}  // namespace ttnn::experimental::prim
