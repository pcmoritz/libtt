// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/experimental/rng_bits/device/rng_bits_device_operation_types.hpp"

namespace ttnn::experimental::prim {

// Splits the output tiles evenly over the cores. Each core's data movement
// kernel reads the state's first page and hands it to the compute kernel,
// which computes the core's tiles on the vector unit; the first core also
// writes the advanced state.
struct RngBitsProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const RngBitsParams&, const RngBitsInputs&, std::vector<Tensor>&);
};

}  // namespace ttnn::experimental::prim
