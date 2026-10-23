// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
//
// Reads this core's thresholds, then its run of input tiles as dense tiles.
//
// A tile row with `rows` rows of data (at most 16) is read `pack` = 32 / stride
// input tiles at a time: input tile jj's first `stride` rows (the data rows,
// plus one padding row when 16-bit rows need an even stride for 64-byte
// aligned chunks) become rows jj * stride .. jj * stride + stride - 1 of one
// dense tile, so the kernel reads next to no padding. With more than 16 data
// rows every input tile is read whole (pack = 1).
//
// The threshold tile reaches the compute kernel with the dense tile's row k
// threshold, t[k % stride], in column 0 of row k; the compute kernel spreads
// that column across the tile. Reads of a batch of dense tiles share one
// barrier, so they overlap.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

namespace {
// A 32 x 32 tile is four 16 x 16 faces: rows 0-15 in faces 0 (columns 0-15)
// and 1 (columns 16-31), rows 16-31 in faces 2 and 3.
constexpr uint32_t face_rows = 16;
constexpr uint32_t face_elems = 256;

// Column 0 of dense row k gets the threshold of data row k % stride.
template <typename T>
inline void dense_threshold_column(uint32_t base, uint32_t rows, uint32_t stride) {
    auto* tile = reinterpret_cast<volatile tt_l1_ptr T*>(base);
    for (uint32_t k = 0; k < 32; ++k) {
        const uint32_t m = k % stride < rows ? k % stride : 0;
        const T value = tile[m * face_rows];  // face 0, row m, column 0
        tile[(k / face_rows) * 2 * face_elems + (k % face_rows) * face_rows] = value;
    }
}
}  // namespace

void kernel_main() {
    const uint32_t tile_row = get_arg(args::tile_row);
    const uint32_t first_tile = get_arg(args::first_tile);
    const uint32_t num_tiles = get_arg(args::num_tiles);
    const uint32_t rows = get_arg(args::rows);
    const uint32_t stride = get_arg(args::stride);
    const uint32_t pack = get_arg(args::pack);
    constexpr uint32_t batch_tiles = get_arg(args::batch_tiles);
    constexpr uint32_t elem_bytes = get_arg(args::elem_bytes);
    constexpr uint32_t row_bytes = face_rows * elem_bytes;
    constexpr uint32_t face_bytes = face_elems * elem_bytes;
    constexpr uint32_t tile_bytes = 4 * face_bytes;

    DataflowBuffer in_dfb(dfb::in);
    DataflowBuffer t_dfb(dfb::t);
    const auto input = TensorAccessor(tensor::input);
    const auto threshold = TensorAccessor(tensor::threshold);
    Noc noc;

    // The threshold tensor has one tile column: tile `tile_row` holds the
    // row's thresholds in its first column.
    t_dfb.reserve_back(1);
    const uint32_t t_addr = t_dfb.get_write_ptr();
    if (pack == 1) {
        noc.async_read(threshold, t_dfb, tile_bytes, {.page_id = tile_row}, {.offset_bytes = 0});
        noc.async_read_barrier();
    } else {
        // The data rows' thresholds sit in face 0; spread them over the dense rows.
        noc.async_read(threshold, t_dfb, stride * row_bytes, {.page_id = tile_row}, {.offset_bytes = 0});
        noc.async_read_barrier();
        if constexpr (elem_bytes == 4) {
            dense_threshold_column<uint32_t>(t_addr, rows, stride);
        } else {
            dense_threshold_column<uint16_t>(t_addr, rows, stride);
        }
    }
    t_dfb.push_back(1);

    const uint32_t dense_tiles = (num_tiles + pack - 1) / pack;
    for (uint32_t done = 0; done < dense_tiles;) {
        const uint32_t count = dense_tiles - done < batch_tiles ? dense_tiles - done : batch_tiles;
        in_dfb.reserve_back(count);
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t dense = done + i;
            const uint32_t dst_tile = i * tile_bytes;
            const uint32_t in_dense = num_tiles - dense * pack < pack ? num_tiles - dense * pack : pack;
            for (uint32_t jj = 0; jj < in_dense; ++jj) {
                const uint32_t page = first_tile + dense * pack + jj;
                if (pack == 1) {
                    noc.async_read(input, in_dfb, tile_bytes, {.page_id = page}, {.offset_bytes = dst_tile});
                    continue;
                }
                // Rows 0 .. stride - 1 of faces 0 and 1 go to dense rows
                // r0 .. r1 - 1, split where those cross into faces 2 and 3.
                const uint32_t r0 = jj * stride;
                const uint32_t r1 = r0 + stride;
                for (uint32_t f = 0; f < 2; ++f) {
                    if (r0 < face_rows) {
                        const uint32_t n = (r1 < face_rows ? r1 : face_rows) - r0;
                        noc.async_read(
                            input,
                            in_dfb,
                            n * row_bytes,
                            {.page_id = page, .offset_bytes = f * face_bytes},
                            {.offset_bytes = dst_tile + f * face_bytes + r0 * row_bytes});
                    }
                    if (r1 > face_rows) {
                        const uint32_t r2 = r0 > face_rows ? r0 : face_rows;
                        noc.async_read(
                            input,
                            in_dfb,
                            (r1 - r2) * row_bytes,
                            {.page_id = page, .offset_bytes = f * face_bytes + (r2 - r0) * row_bytes},
                            {.offset_bytes = dst_tile + (f + 2) * face_bytes + (r2 - face_rows) * row_bytes});
                    }
                }
            }
        }
        noc.async_read_barrier();
        in_dfb.push_back(count);
        done += count;
    }
}
