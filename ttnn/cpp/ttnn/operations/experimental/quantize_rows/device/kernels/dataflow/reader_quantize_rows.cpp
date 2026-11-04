// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
//
// Reads this core's run of the row's tiles as dense tiles, then trades row
// maxima with the other cores.
//
// Input tile jj of a dense tile has its rows 0 .. stride - 1 (the data rows and
// the padding up to the stride) at dense rows jj * stride .. jj * stride +
// stride - 1: two chunks, one from each left/right face pair. A stride of 32
// reads whole tiles.
//
// The compute kernel then packs this core's row maxima: row m's maximum in
// columns 0-15 of row m of an FP32 tile's left faces. Every core writes those
// rows (stride * 64 bytes, or both left faces for a stride of 32) into its slot
// of every core's gather buffer, which every core allocates alike, so the
// remote slot is at this core's own address. Unicast: concurrent multicasts
// from all the cores serialize (16 cores: 22 us for the data, as much again
// for multicast semaphore increments, against about 1 us so). Once its data
// has landed everywhere, a core counts itself on the other cores' semaphores,
// and when all others have counted themselves here, the gather buffer goes to
// the compute kernel.
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
}  // namespace

void kernel_main() {
    const uint32_t first_dense = get_arg(args::first_dense);
    const uint32_t num_dense = get_arg(args::num_dense);
    const uint32_t last_count = get_arg(args::last_count);
    const uint32_t core = get_arg(args::core);
    const uint32_t self_x = get_arg(args::self_x);
    const uint32_t self_y = get_arg(args::self_y);
    constexpr uint32_t stride = get_arg(args::stride);
    constexpr uint32_t pack = get_arg(args::pack);
    constexpr uint32_t elem_bytes = get_arg(args::elem_bytes);
    constexpr uint32_t cores = get_arg(args::cores);
    constexpr uint32_t slot_bytes = get_arg(args::slot_bytes);
    constexpr uint32_t gather_tiles = get_arg(args::gather_tiles);
    constexpr uint32_t x0 = get_arg(args::x0);
    constexpr uint32_t y0 = get_arg(args::y0);
    constexpr uint32_t x1 = get_arg(args::x1);
    constexpr uint32_t y1 = get_arg(args::y1);
    constexpr uint32_t row_bytes = face_rows * elem_bytes;
    constexpr uint32_t face_bytes = face_elems * elem_bytes;
    constexpr uint32_t tile_bytes = 4 * face_bytes;

    DataflowBuffer in_dfb(dfb::in);
    DataflowBuffer part_dfb(dfb::part);
    DataflowBuffer gather_dfb(dfb::gather);
    Semaphore done(sem::done);
    const auto input = TensorAccessor(tensor::input);
    Noc noc;

    in_dfb.reserve_back(num_dense);
    for (uint32_t d = 0; d < num_dense; ++d) {
        const uint32_t count = d + 1 == num_dense ? last_count : pack;
        const uint32_t dense_offset = d * tile_bytes;
        for (uint32_t jj = 0; jj < count; ++jj) {
            const uint32_t page = (first_dense + d) * pack + jj;
            if constexpr (pack == 1) {
                noc.async_read(input, in_dfb, tile_bytes, {.page_id = page}, {.offset_bytes = dense_offset});
            } else {
                // A run of `stride` rows never crosses from faces 0-1 into 2-3.
                const uint32_t r0 = jj * stride;
                const uint32_t face = r0 < face_rows ? 0 : 2;
                const uint32_t at = dense_offset + face * face_bytes + (r0 % face_rows) * row_bytes;
                for (uint32_t f = 0; f < 2; ++f) {
                    noc.async_read(
                        input,
                        in_dfb,
                        stride * row_bytes,
                        {.page_id = page, .offset_bytes = f * face_bytes},
                        {.offset_bytes = at + f * face_bytes});
                }
            }
        }
    }
    noc.async_read_barrier();
    in_dfb.push_back(num_dense);

    // This core's row maxima into its slot on every core.
    gather_dfb.reserve_back(gather_tiles);
    const uint32_t slot = gather_dfb.get_write_ptr() + core * slot_bytes;
    // Rows 0-15 of the left faces 0 and 2: face 2 holds rows 16-31.
    constexpr uint32_t pieces = stride <= face_rows ? 1 : 2;
    constexpr uint32_t piece_bytes = (stride <= face_rows ? stride : face_rows) * 64;
    part_dfb.wait_front(1);
    for (uint32_t y = y0; y <= y1; ++y) {
        for (uint32_t x = x0; x <= x1; ++x) {
            for (uint32_t p = 0; p < pieces; ++p) {
                noc.async_write(
                    part_dfb,
                    UnicastEndpoint{},
                    piece_bytes,
                    {.offset_bytes = p * 2 * face_elems * 4},
                    {.noc_x = x, .noc_y = y, .addr = slot + p * piece_bytes});
            }
        }
    }
    // Every copy acknowledged before any core is told.
    noc.async_write_barrier();
    part_dfb.pop_front(1);
    if constexpr (cores > 1) {
        for (uint32_t y = y0; y <= y1; ++y) {
            for (uint32_t x = x0; x <= x1; ++x) {
                if (x != self_x || y != self_y) {
                    done.up(noc, x, y, 1);
                }
            }
        }
        done.wait(cores - 1);
        done.set(0);
    }
    gather_dfb.push_back(gather_tiles);
    noc_async_atomic_barrier();
}
