// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/experimental/matmul_fp32/device/matmul_fp32_device_operation_types.hpp"

namespace ttnn::experimental::prim {

// matmul's multi-core program (MatmulMultiCoreProgramFactory) with the vector
// unit compute kernel: output tiles split over the cores, each core streaming
// its tiles' A rows and B columns through matmul's reader and writer.
struct MatmulFp32ProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const MatmulFp32Params&, const MatmulFp32Inputs&, std::vector<Tensor>&);
};

}  // namespace ttnn::experimental::prim
