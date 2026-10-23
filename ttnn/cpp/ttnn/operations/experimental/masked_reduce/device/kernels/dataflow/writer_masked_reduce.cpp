// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
//
// Turns this core's accumulator tile into its slot of the partials.
//
// The compute kernel leaves dense row k's sum in column 16 of the tile. Data
// row m's sum is the sum over the dense rows jj * stride + m; it goes into
// column `col` (= 4 * core, so the 16-byte slot is NoC-aligned) of row m of
// the partials' tile row, with the slot's other three columns zero. The row's
// last core also zeroes the partials' padding columns.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

namespace {
constexpr uint32_t face_rows = 16;
constexpr uint32_t face_elems = 256;
constexpr uint32_t slot_bytes = 16;
// Byte offset of element (r, c) in an FP32 tile.
constexpr uint32_t elem_offset(uint32_t r, uint32_t c) {
    return (((r / face_rows) * 2 + c / face_rows) * face_elems + (r % face_rows) * face_rows + c % face_rows) * 4;
}
}  // namespace

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);
    const uint32_t rows = get_arg(args::rows);
    const uint32_t stride = get_arg(args::stride);
    const uint32_t pack = get_arg(args::pack);
    const uint32_t out_tile = get_arg(args::out_tile);
    const uint32_t col = get_arg(args::col);
    const uint32_t last_worker = get_arg(args::last_worker);
    constexpr uint32_t cols = get_arg(args::cols);

    DataflowBuffer out_dfb(dfb::out);
    const auto partials = TensorAccessor(tensor::partials);
    Noc noc;
    out_dfb.wait_front(1);
    const uint32_t base = out_dfb.get_read_ptr();
    auto* tile = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);

    // Row sums, read before the staging area (face 0's first rows) is reused.
    float sums[32];
    for (uint32_t m = 0; m < 32; ++m) {
        float s = 0.0f;
        if (num_tiles != 0 && m < rows) {
            for (uint32_t jj = 0; jj < pack; ++jj) {
                const uint32_t bits = tile[elem_offset(jj * stride + m, face_rows) / 4];
                s += __builtin_bit_cast(float, bits);
            }
        }
        sums[m] = s;
    }
    // Staging: row m's 16-byte slot at m * 16, then 64 zero bytes for padding.
    for (uint32_t m = 0; m < 32; ++m) {
        tile[4 * m] = __builtin_bit_cast(uint32_t, sums[m]);
        tile[4 * m + 1] = 0;
        tile[4 * m + 2] = 0;
        tile[4 * m + 3] = 0;
    }
    const uint32_t zeros = 32 * slot_bytes;
    for (uint32_t i = 0; i < 16; ++i) {
        tile[zeros / 4 + i] = 0;
    }

    const uint32_t page = out_tile + col / 32;
    const uint32_t c = col % 32;
    for (uint32_t m = 0; m < 32; ++m) {
        noc.async_write(
            out_dfb, partials, slot_bytes, {.offset_bytes = m * slot_bytes}, {.page_id = page, .offset_bytes = elem_offset(m, c)});
    }
    if (last_worker && col + 4 < cols) {
        // Padding columns col + 4 .. cols - 1 lie in this tile: zero them, a
        // face's run of columns per row at a time.
        const uint32_t c0 = c + 4;
        for (uint32_t m = 0; m < 32; ++m) {
            if (c0 < face_rows) {
                noc.async_write(
                    out_dfb,
                    partials,
                    (face_rows - c0) * 4,
                    {.offset_bytes = zeros},
                    {.page_id = page, .offset_bytes = elem_offset(m, c0)});
            }
            const uint32_t c1 = c0 > face_rows ? c0 : face_rows;
            noc.async_write(
                out_dfb, partials, (32 - c1) * 4, {.offset_bytes = zeros}, {.page_id = page, .offset_bytes = elem_offset(m, c1)});
        }
    }
    noc.async_write_barrier();
    out_dfb.pop_front(1);
}
