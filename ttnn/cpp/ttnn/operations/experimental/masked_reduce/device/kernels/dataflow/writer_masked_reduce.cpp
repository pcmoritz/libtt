// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
//
// Sends this core's row sums to its tile row's master core, which adds up the
// row's partials and writes the result.
//
// The compute kernel leaves dense row k's sum in column 16 of the accumulator
// tile. Data row m's sum is the sum over the dense rows jj * stride + m; it
// goes into column `col` (= 4 * core, so the 16-byte slot is NoC-aligned) of
// row m of the master's partial tiles, with the slot's other three columns
// zero, and the core counts itself on the master's semaphore. The partial
// tiles are a dataflow buffer every core allocates alike, so the master's
// buffer sits at this core's own address for it. The master zeroes the
// partials' padding columns, waits for the row's other cores, hands the
// partials to its compute kernel, and writes the summed rows as the result:
// a [..., H, 1] result's tile t_page with the sums in column 0 and zeros
// elsewhere, or row t_row of a [..., H] result's tile t_page. Counts can go
// out as int32.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

namespace {
constexpr uint32_t face_rows = 16;
constexpr uint32_t face_elems = 256;
constexpr uint32_t slot_bytes = 16;
constexpr uint32_t tile_bytes = 4 * face_elems * 4;
// Byte offset of element (r, c) in an FP32 tile.
constexpr uint32_t elem_offset(uint32_t r, uint32_t c) {
    return (((r / face_rows) * 2 + c / face_rows) * face_elems + (r % face_rows) * face_rows + c % face_rows) * 4;
}
}  // namespace

void kernel_main() {
    const uint32_t t_page = get_arg(args::t_page);
    const uint32_t t_row = get_arg(args::t_row);
    const uint32_t num_tiles = get_arg(args::num_tiles);
    const uint32_t rows = get_arg(args::rows);
    const uint32_t stride = get_arg(args::stride);
    const uint32_t pack = get_arg(args::pack);
    const uint32_t col = get_arg(args::col);
    const uint32_t is_master = get_arg(args::is_master);
    const uint32_t master_x = get_arg(args::master_x);
    const uint32_t master_y = get_arg(args::master_y);
    const uint32_t remote_workers = get_arg(args::remote_workers);
    constexpr uint32_t cols = get_arg(args::cols);
    constexpr uint32_t partial_tiles = get_arg(args::partial_tiles);
    constexpr uint32_t row_form = get_arg(args::row_form);
    constexpr uint32_t int_output = get_arg(args::int_output);

    DataflowBuffer out_dfb(dfb::out);
    DataflowBuffer part_dfb(dfb::part);
    Semaphore done(sem::done);
    const auto output = TensorAccessor(tensor::output);
    Noc noc;

    // The partials' address, the same on every core of the program.
    const uint32_t part_base = part_dfb.get_write_ptr();
    const uint32_t slot_addr = part_base + (col / 32) * tile_bytes;
    const uint32_t c = col % 32;
    if (is_master) {
        part_dfb.reserve_back(partial_tiles);
        // Columns past the row's cores in the last partial tile are never
        // written: zero them before counting the others' slots.
        if (cols > col + 4 * (remote_workers + 1)) {
            auto* last = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(part_base + (partial_tiles - 1) * tile_bytes);
            for (uint32_t m = 0; m < 32; ++m) {
                for (uint32_t cc = 4 * (remote_workers + 1) % 32; cc < 32; ++cc) {
                    last[elem_offset(m, cc) / 4] = 0;
                }
            }
        }
    }

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
    if (is_master) {
        // This core's own slots go straight into its partials.
        auto* part = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot_addr);
        for (uint32_t m = 0; m < 32; ++m) {
            const uint32_t at = elem_offset(m, c) / 4;
            part[at] = __builtin_bit_cast(uint32_t, sums[m]);
            part[at + 1] = 0;
            part[at + 2] = 0;
            part[at + 3] = 0;
        }
    } else {
        // Staging: row m's 16-byte slot at m * 16, written to the master.
        for (uint32_t m = 0; m < 32; ++m) {
            tile[4 * m] = __builtin_bit_cast(uint32_t, sums[m]);
            tile[4 * m + 1] = 0;
            tile[4 * m + 2] = 0;
            tile[4 * m + 3] = 0;
        }
        for (uint32_t m = 0; m < 32; ++m) {
            noc.async_write(
                out_dfb,
                UnicastEndpoint{},
                slot_bytes,
                {.offset_bytes = m * slot_bytes},
                {.noc_x = master_x, .noc_y = master_y, .addr = slot_addr + elem_offset(m, c)});
        }
        noc.async_write_barrier();
        done.up(noc, master_x, master_y, 1);
        noc_async_atomic_barrier();
    }
    out_dfb.pop_front(1);
    if (!is_master) {
        return;
    }

    done.wait(remote_workers);
    done.set(0);
    part_dfb.push_back(partial_tiles);

    // The summed partials: row m's total in column 16.
    out_dfb.wait_front(1);
    const uint32_t result = out_dfb.get_read_ptr();
    auto* res = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(result);
    uint32_t totals[32];
    for (uint32_t m = 0; m < 32; ++m) {
        const float total = __builtin_bit_cast(float, res[elem_offset(m, face_rows) / 4]);
        if constexpr (int_output) {
            totals[m] = static_cast<uint32_t>(static_cast<int32_t>(total));
        } else {
            totals[m] = __builtin_bit_cast(uint32_t, total);
        }
    }
    if constexpr (row_form) {
        // Row t_row of the result's tile, a face's run of columns at a time.
        for (uint32_t m = 0; m < 32; ++m) {
            res[elem_offset(t_row, m) / 4] = totals[m];
        }
        for (uint32_t f = 0; f < 2; ++f) {
            const uint32_t at = elem_offset(t_row, f * face_rows);
            noc.async_write(out_dfb, output, face_rows * 4, {.offset_bytes = at}, {.page_id = t_page, .offset_bytes = at});
        }
    } else {
        // The result's tile: the totals in column 0 and zeros elsewhere.
        for (uint32_t i = 0; i < tile_bytes / 4; ++i) {
            res[i] = 0;
        }
        for (uint32_t m = 0; m < 32; ++m) {
            res[elem_offset(m, 0) / 4] = totals[m];
        }
        noc.async_write(out_dfb, output, tile_bytes, {.offset_bytes = 0}, {.page_id = t_page, .offset_bytes = 0});
    }
    noc.async_write_barrier();
    out_dfb.pop_front(1);
}
