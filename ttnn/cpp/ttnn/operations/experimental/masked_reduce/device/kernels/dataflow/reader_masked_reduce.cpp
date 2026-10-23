// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
//
// Reads this core's threshold tile and spreads each row's threshold (column 0)
// across the tile's 32 columns, so the compute kernel can subtract it
// elementwise; then reads the core's run of input tiles in batches of
// batch_tiles with one barrier per batch, so the reads of a batch overlap.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

namespace {
// A 32 x 32 tile is four 16 x 16 faces: rows 0-15 in faces 0 (columns 0-15)
// and 1 (columns 16-31), rows 16-31 in faces 2 and 3.
constexpr uint32_t face_elems = 256;
// 1.0 for the first `valid` columns of a tile, 0.0 after them.
template <typename T, T one>
inline void write_validity(uint32_t base, uint32_t valid) {
    auto* tile = reinterpret_cast<volatile tt_l1_ptr T*>(base);
    for (uint32_t r = 0; r < 32; ++r) {
        for (uint32_t c = 0; c < 32; ++c) {
            const uint32_t at = ((r / 16) * 2 + c / 16) * face_elems + (r % 16) * 16 + c % 16;
            tile[at] = c < valid ? one : T(0);
        }
    }
}

template <typename T>
inline void spread_column(uint32_t base) {
    auto* tile = reinterpret_cast<volatile tt_l1_ptr T*>(base);
    for (uint32_t r = 0; r < 32; ++r) {
        const uint32_t left = (r / 16) * 2 * face_elems + (r % 16) * 16;
        const T value = tile[left];
        for (uint32_t c = 1; c < 16; ++c) {
            tile[left + c] = value;
        }
        for (uint32_t c = 0; c < 16; ++c) {
            tile[left + face_elems + c] = value;
        }
    }
}
}  // namespace

void kernel_main() {
    const uint32_t tile_row = get_arg(args::tile_row);
    const uint32_t first_tile = get_arg(args::first_tile);
    const uint32_t num_tiles = get_arg(args::num_tiles);
    constexpr uint32_t batch_tiles = get_arg(args::batch_tiles);
    constexpr uint32_t elem_bytes = get_arg(args::elem_bytes);
    constexpr uint32_t last_valid = get_arg(args::last_valid);
    const uint32_t has_last = get_arg(args::has_last);

    DataflowBuffer in_dfb(dfb::in);
    DataflowBuffer t_dfb(dfb::t);
    DataflowBuffer v_dfb(dfb::v);
    const uint32_t tile_bytes = in_dfb.get_tile_size();
    const auto input = TensorAccessor(tensor::input);
    const auto threshold = TensorAccessor(tensor::threshold);
    Noc noc;

    // The threshold tensor has one tile column: tile `tile_row` holds the
    // row's thresholds in its first column.
    t_dfb.reserve_back(1);
    const uint32_t t_addr = t_dfb.get_write_ptr();
    noc.async_read(threshold, t_dfb, tile_bytes, {.page_id = tile_row}, {.offset_bytes = 0});
    noc.async_read_barrier();
    if constexpr (elem_bytes == 4) {
        spread_column<uint32_t>(t_addr);
    } else {
        spread_column<uint16_t>(t_addr);
    }
    t_dfb.push_back(1);

    if constexpr (last_valid != 0) {
        if (has_last) {
            v_dfb.reserve_back(1);
            if constexpr (elem_bytes == 4) {
                write_validity<uint32_t, 0x3F800000u>(v_dfb.get_write_ptr(), last_valid);
            } else {
                write_validity<uint16_t, 0x3F80u>(v_dfb.get_write_ptr(), last_valid);
            }
            v_dfb.push_back(1);
        }
    }

    for (uint32_t done = 0; done < num_tiles;) {
        const uint32_t count = num_tiles - done < batch_tiles ? num_tiles - done : batch_tiles;
        in_dfb.reserve_back(count);
        for (uint32_t i = 0; i < count; ++i) {
            noc.async_read(input, in_dfb, tile_bytes, {.page_id = first_tile + done + i}, {.offset_bytes = i * tile_bytes});
        }
        noc.async_read_barrier();
        in_dfb.push_back(count);
        done += count;
    }
}
