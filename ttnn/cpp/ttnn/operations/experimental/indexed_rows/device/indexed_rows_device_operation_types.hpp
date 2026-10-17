// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt_stl/small_vector.hpp>

#include "ttnn/operations/experimental/indexed_rows/indexed_rows.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct IndexedRowsParams {
    ttsl::SmallVector<uint32_t> sizes;
    ttsl::SmallVector<uint32_t> windows;
    bool column = false;
    // rows_gather: write zeros for clamped positions. rows_scatter: unused.
    bool fill_out_of_bounds = false;
    // rows_scatter only.
    RowsReduce reduce = RowsReduce::None;
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct IndexedRowsInputs {
    Tensor table;
    Tensor indices;
    // rows_scatter only.
    std::optional<Tensor> source;
};

}  // namespace ttnn::experimental::prim
