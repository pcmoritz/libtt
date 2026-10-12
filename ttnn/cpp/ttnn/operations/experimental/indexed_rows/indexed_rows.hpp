// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt_stl/small_vector.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental {

// How an index tensor addresses a table's rows. A table [B, N, S..., R] is, per
// batch, N rows of S consecutive pages of R elements (S is the product of any
// dimensions between N and the last; a table keeps its operand's page size).
// The index tensor [Bi, M * K] holds K int32 components per position (one page
// per batch, so the kernels stream it in chunks). The row
// components (all K, or the first K - 1 with `column`) pick a start along a
// dimension of extent sizes[k] and `windows[k]` consecutive positions follow
// it; the dimensions are laid out row-major in the N rows, so each position
// names W = prod(row windows) rows. With `column`, the last component picks a
// window of windows[K - 1] elements inside each row's single page (S must be 1)
// instead of the whole row. Components are clamped into
// [0, sizes[k] - windows[k]] like StableHLO's gather start indices.
struct IndexedRowsSpec {
    ttsl::SmallVector<uint32_t> sizes;
    ttsl::SmallVector<uint32_t> windows;
    bool column = false;
};

enum class RowsReduce : uint32_t { None = 0, Add = 1, Mul = 2, Max = 3, Min = 4 };

// out[b, m * W + j, ...] = table[b, row_j(indices[bi, :, m]), ...] for the W
// window offsets j, with bi = b, or 0 when the index tensor has one batch. table
// is a row-major [B, N, S..., R] tensor of any element type, indices a row-major
// [Bi, K, M] int32 or uint32 tensor (one page per component, so that assembling
// them never makes K-element pages); the output is row-major [B, M * W, S..., R]
// in the table's type, or [B, M * W * w] with a column window of w elements.
// With fill_out_of_bounds, positions whose components needed clamping read as
// zeros instead.
ttnn::Tensor rows_gather(
    const ttnn::Tensor& table,
    const ttnn::Tensor& indices,
    const IndexedRowsSpec& spec,
    bool fill_out_of_bounds = false,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt);

// A copy of table with out[b, row_j(indices[bi, :, m]), ...] = source[b, m * W + j, ...]
// for every in-bounds position (out-of-bounds positions are dropped, as in
// StableHLO's scatter), or reduced into it with `reduce`. Duplicate positions
// apply in increasing m; the result is deterministic. source is row-major
// [B, M * W, S..., R] in the table's type, or [B, M * W * w] with a column
// window.
ttnn::Tensor rows_scatter(
    const ttnn::Tensor& table,
    const ttnn::Tensor& indices,
    const ttnn::Tensor& source,
    const IndexedRowsSpec& spec,
    RowsReduce reduce = RowsReduce::None,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::experimental
