// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
//
// Assembles this core's run of output tiles row by row. Row r of a tile is
// row r of its two upper faces (r < 16) or lower ones, 16 elements each; every
// output row comes from one input row:
//
//   Pack:   output tile T of chunk matrix (o, h) of a [..., H, N, 32] result,
//           row r <- chunk 32 T + r of row h: input tile (o, h / 32, 32 T + r),
//           row h % 32.
//   Unpack: output tile (o, ht, n) of a [..., H, N * 32] result, row i <- row
//           n % 32 of input tile n / 32 of chunk matrix (o, 32 ht + i).
//
// Rows past the data (a chunk matrix's last rows, or the rows past H) are tile
// padding and get zeros. A buffer slot keeps the zeros once written: they are
// written again only into slots that have held a tile with more data rows.
// The reads of a batch of tiles share one barrier, so they overlap.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t first_page = get_arg(args::first_page);
    const uint32_t num_pages = get_arg(args::num_pages);
    const uint32_t H = get_arg(args::H);
    const uint32_t N = get_arg(args::N);
    const uint32_t Ht = get_arg(args::Ht);
    const uint32_t Nt = get_arg(args::Nt);
    constexpr uint32_t unpack = get_arg(args::unpack);
    constexpr uint32_t batch_tiles = get_arg(args::batch_tiles);
    constexpr uint32_t row_bytes = get_arg(args::row_bytes);
    constexpr uint32_t face_bytes = 16 * row_bytes;
    constexpr uint32_t tile_bytes = 4 * face_bytes;
    constexpr uint32_t slots = 2 * batch_tiles;

    DataflowBuffer out_dfb(dfb::out);
    const auto input = TensorAccessor(tensor::input);
    Noc noc;

    // Byte offset of row r's upper-left (f = 0) or upper-right (f = 1) face row.
    auto row_offset = [](uint32_t r) { return (r / 16) * 2 * face_bytes + (r % 16) * row_bytes; };

    uint32_t max_valid = 0;
    for (uint32_t done = 0; done < num_pages;) {
        const uint32_t count = num_pages - done < batch_tiles ? num_pages - done : batch_tiles;
        out_dfb.reserve_back(count);
        bool zeroed = false;
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t page = first_page + done + i;
            uint32_t valid, src_page, page_step, src_row;
            if constexpr (unpack) {
                const uint32_t o = page / (Ht * N);
                const uint32_t rem = page - o * Ht * N;
                const uint32_t ht = rem / N;
                const uint32_t n = rem - ht * N;
                valid = H - 32 * ht < 32 ? H - 32 * ht : 32;
                src_page = (o * H + 32 * ht) * Nt + n / 32;
                page_step = Nt;
                src_row = n % 32;
            } else {
                const uint32_t m = page / Nt;
                const uint32_t t = page - m * Nt;
                const uint32_t o = m / H;
                const uint32_t h = m - o * H;
                valid = N - 32 * t < 32 ? N - 32 * t : 32;
                src_page = (o * Ht + h / 32) * N + 32 * t;
                page_step = 1;
                src_row = h % 32;
            }
            const uint32_t dst = i * tile_bytes;
            if (valid < 32 && (done + i < slots || max_valid > valid)) {
                if (valid < 16) {
                    for (uint32_t f = 0; f < 2; ++f) {
                        noc.async_write_zeros(
                            out_dfb, (16 - valid) * row_bytes, {.offset_bytes = dst + f * face_bytes + valid * row_bytes});
                    }
                    noc.async_write_zeros(out_dfb, 2 * face_bytes, {.offset_bytes = dst + 2 * face_bytes});
                } else {
                    for (uint32_t f = 2; f < 4; ++f) {
                        noc.async_write_zeros(
                            out_dfb,
                            (32 - valid) * row_bytes,
                            {.offset_bytes = dst + f * face_bytes + (valid - 16) * row_bytes});
                    }
                }
                zeroed = true;
            }
            max_valid = valid > max_valid ? valid : max_valid;
            const uint32_t src_offset = row_offset(src_row);
            for (uint32_t r = 0; r < valid; ++r) {
                const uint32_t dst_offset = dst + row_offset(r);
                for (uint32_t f = 0; f < 2; ++f) {
                    noc.async_read(
                        input,
                        out_dfb,
                        row_bytes,
                        {.page_id = src_page + r * page_step, .offset_bytes = src_offset + f * face_bytes},
                        {.offset_bytes = dst_offset + f * face_bytes});
                }
            }
        }
        noc.async_read_barrier();
        if (zeroed) {
            noc.write_zeros_l1_barrier();
        }
        out_dfb.push_back(count);
        done += count;
    }
}
