// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/experimental/indexed_rows/device/indexed_rows_device_operation_types.hpp"

namespace ttnn::experimental::prim {

// Splits the index positions over the cores; each core gathers its positions'
// rows (kernels/dataflow/rows_gather_*.cpp).
struct RowsGatherProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const IndexedRowsParams&, const IndexedRowsInputs&, std::vector<Tensor>&);
};

// Splits the table rows over the cores; each core copies its rows and applies
// the updates that land in them (kernels/dataflow/rows_scatter_*.cpp).
struct RowsScatterProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const IndexedRowsParams&, const IndexedRowsInputs&, std::vector<Tensor>&);
};

}  // namespace ttnn::experimental::prim
