// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt_stl/small_vector.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental {

// How an index tensor addresses a table's rows. A table [B, N, R] is N rows of
// R elements per batch; its rows are the row-major flattening of K dimensions
// of extents `sizes` (prod(sizes) == N). Index component k picks a start along
// dimension k and `windows[k]` consecutive positions follow it, so each index
// position names W = prod(windows) rows. Components are clamped into
// [0, sizes[k] - windows[k]] like StableHLO's gather start indices.
struct IndexedRowsSpec {
    ttsl::SmallVector<uint32_t> sizes;
    ttsl::SmallVector<uint32_t> windows;
};

enum class RowsReduce : uint32_t { None = 0, Add = 1, Mul = 2, Max = 3, Min = 4 };

// out[b, m * W + j, :] = table[b, row_j(indices[bi, m, :]), :] for the W window
// offsets j, with bi = b, or 0 when the index tensor has one batch. table is a
// row-major [B, N, R] tensor of any element type, indices a row-major
// [Bi, M, K] int32 or uint32 tensor; the output is row-major [B, M * W, R] in
// the table's type. With fill_out_of_bounds, positions whose components needed
// clamping read as zeros instead.
ttnn::Tensor rows_gather(
    const ttnn::Tensor& table,
    const ttnn::Tensor& indices,
    const IndexedRowsSpec& spec,
    bool fill_out_of_bounds = false,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt);

// A copy of table with out[b, row_j(indices[bi, m, :]), :] = source[b, m * W + j, :]
// for every in-bounds position (out-of-bounds positions are dropped, as in
// StableHLO's scatter), or reduced into it with `reduce`. Duplicate positions
// apply in increasing m; the result is deterministic. source is row-major
// [B, M * W, R] in the table's type.
ttnn::Tensor rows_scatter(
    const ttnn::Tensor& table,
    const ttnn::Tensor& indices,
    const ttnn::Tensor& source,
    const IndexedRowsSpec& spec,
    RowsReduce reduce = RowsReduce::None,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::experimental
