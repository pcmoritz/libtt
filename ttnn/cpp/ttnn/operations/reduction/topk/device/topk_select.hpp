// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

// TopK's selection path (TopKDeviceOperation::TopKSelectProgramFactory): the
// data-movement cores scan their rows once and keep the best k by insertion,
// comparing values as unsigned keys in IEEE total order and keeping the lower
// index first among equal ones. It takes the calls the sort networks are slow
// at: up to two tile rows (a decode step's router or logits) with k up to a
// tile, and wide rows, which split into column segments whose candidates one
// core per row merges.
//
// The composite in topk.cpp, select_program_factory and validation all decide
// with topk_select_plan, which only gets stricter as k grows: a call that does
// not take the path with its own k does not take it with the tile-rounded one
// the sort networks get either.
struct TopKSelectPlan {
    uint32_t tile_rows = 0;      // tile rows of the input, over all its [H, W] slabs
    uint32_t Ht = 0;             // tile rows per slab
    uint32_t height = 0;         // logical rows per slab
    uint32_t width = 0;          // logical width
    uint32_t Wt = 0;             // width in tiles
    uint32_t segments = 0;       // column segments per row, merged by the first segment's core
    uint32_t segment_tiles = 0;  // width tiles per segment; the last one may be shorter
    uint32_t row_groups = 0;     // cores sharing a tile row's rows within a segment
    uint32_t rows_per_core = 0;  // most rows one core selects
    uint32_t k = 0;

    uint32_t cores_per_tile_row() const { return segments * row_groups; }
    uint32_t cores() const { return tile_rows * cores_per_tile_row(); }
    // Most logical rows in a tile row.
    uint32_t rows() const { return Ht == 1 ? height : 32; }
};

std::optional<TopKSelectPlan> topk_select_plan(
    const Tensor& input,
    uint32_t k,
    bool custom_indices,
    bool preallocated_outputs,
    const tt::tt_metal::MemoryConfig& output_memory_config,
    const tt::tt_metal::CoreRangeSet& grid);

}  // namespace ttnn::prim
