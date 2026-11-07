// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

// Rows r < R of this core's input tiles c (4-byte elements) to row 0 of output
// tiles r * Nt + c. A batch of input tiles comes in behind one barrier, their
// rows' face pieces (64 bytes each) go out behind another.

#include <stdint.h>

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"

void kernel_main() {
    constexpr uint32_t rows = get_compile_time_arg_val(0);
    constexpr uint32_t Nt = get_compile_time_arg_val(1);
    constexpr uint32_t batch_tiles = get_compile_time_arg_val(2);
    constexpr auto input_args = TensorAccessorArgs<3>();
    constexpr auto output_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    // A face of 16 x 16 four-byte elements, and one of its rows.
    constexpr uint32_t face_bytes = 1024;
    constexpr uint32_t row_bytes = 64;
    constexpr uint32_t tile_bytes = 4 * face_bytes;
    constexpr uint32_t top_rows = rows < 16 ? rows : 16;
    constexpr uint32_t bottom_rows = rows - top_rows;

    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);
    const auto input = TensorAccessor(input_args, get_arg_val<uint32_t>(2), tile_bytes);
    const auto output = TensorAccessor(output_args, get_arg_val<uint32_t>(3), tile_bytes);

    CircularBuffer scratch(tt::CBIndex::c_0);
    Noc noc;
    scratch.reserve_back(batch_tiles);
    const uint32_t base = scratch.get_write_ptr();

    for (uint32_t first = start; first < start + count; first += batch_tiles) {
        const uint32_t tiles = start + count - first < batch_tiles ? start + count - first : batch_tiles;
        // The rows' pieces of each face, at their places in a scratch tile.
        for (uint32_t i = 0; i < tiles; ++i) {
            for (uint32_t face = 0; face < 4; ++face) {
                const uint32_t face_rows = face < 2 ? top_rows : bottom_rows;
                if (face_rows != 0) {
                    noc.async_read(
                        input, scratch, face_rows * row_bytes,
                        {.page_id = first + i, .offset_bytes = face * face_bytes},
                        {.offset_bytes = i * tile_bytes + face * face_bytes});
                }
            }
        }
        noc.async_read_barrier();
        // Row r of a tile is row r % 16 of its faces 2 (r / 16) and 2 (r / 16) + 1,
        // and row 0 of faces 0 and 1 of its output tile.
        for (uint32_t i = 0; i < tiles; ++i) {
            for (uint32_t row = 0; row < rows; ++row) {
                const uint32_t left = i * tile_bytes + (row / 16) * 2 * face_bytes + (row % 16) * row_bytes;
                for (uint32_t half = 0; half < 2; ++half) {
                    noc.async_write(
                        CoreLocalMem<uint32_t>(base + left + half * face_bytes), output, row_bytes, {},
                        {.page_id = row * Nt + first + i, .offset_bytes = half * face_bytes});
                }
            }
        }
        noc.async_write_barrier();
    }
}
