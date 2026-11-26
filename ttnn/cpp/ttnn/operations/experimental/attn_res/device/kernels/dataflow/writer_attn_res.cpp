// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// The exchanges between cores and the output rows.
//
// An exchange: the compute kernel packs this core's row tile, whose rows
// 0..3 (one per token) of faces 0 and 1 carry its partial sums; they go to
// slot `worker` of the first core's gather tiles (8 slots a tile: faces
// 2 * (slot / 4) and the next, rows 4 * (slot % 4) ..), and the core counts
// itself in on the first core's `written` semaphore. The first core
// multicasts the gather tiles to every other core's gather buffer (the
// buffers have the same address everywhere) and sets their `valid`
// semaphore.
//
// The output: the compute kernel packs one stage tile whose band b of faces
// A and A + 1 (A = 2 * (u / 4), b = u % 4) holds output tile u = q * count +
// c's rows; the writer copies the token rows to faces 0 and 1 of output tile
// (q, start + c).
void kernel_main() {
    const uint32_t worker = get_arg_val<uint32_t>(0);
    const uint32_t start = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(2);
    const uint32_t output_addr = get_arg_val<uint32_t>(3);
    const uint32_t gather_x = get_arg_val<uint32_t>(4);
    const uint32_t gather_y = get_arg_val<uint32_t>(5);
    // Two multicast rectangles (x0, y0, x1, y1, destinations), the second one
    // possibly empty.
    constexpr uint32_t rectangles_arg = 6;

    constexpr uint32_t cb_row1 = get_named_compile_time_arg_val("cb_row1");
    constexpr uint32_t cb_gather1 = get_named_compile_time_arg_val("cb_gather1");
    constexpr uint32_t cb_row2 = get_named_compile_time_arg_val("cb_row2");
    constexpr uint32_t cb_gather2 = get_named_compile_time_arg_val("cb_gather2");
    constexpr uint32_t cb_output = get_named_compile_time_arg_val("cb_output");
    constexpr uint32_t num_cores = get_named_compile_time_arg_val("num_cores");
    constexpr uint32_t gather_tiles = get_named_compile_time_arg_val("gather_tiles");
    constexpr uint32_t tokens = get_named_compile_time_arg_val("tokens");
    constexpr uint32_t num_queries = get_named_compile_time_arg_val("num_queries");
    constexpr uint32_t Wt = get_named_compile_time_arg_val("Wt");
    constexpr bool has_weight = get_named_compile_time_arg_val("has_weight") != 0;
    constexpr uint32_t written_sem = get_named_compile_time_arg_val("written_sem");
    constexpr uint32_t valid_sem = get_named_compile_time_arg_val("valid_sem");
    constexpr auto output_args = TensorAccessorArgs<0>();
    constexpr uint32_t f32_tile = 4096, f32_face = 1024, f32_band = 256;

    const uint32_t written_addr = get_semaphore(written_sem);
    const uint32_t valid_addr = get_semaphore(valid_sem);
    volatile tt_l1_ptr uint32_t* written = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(written_addr);
    volatile tt_l1_ptr uint32_t* valid = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(valid_addr);
    auto rectangle = [&](uint32_t r, uint32_t field) { return get_arg_val<uint32_t>(rectangles_arg + 5 * r + field); };
    // `source` of the first core goes to `address` on every other core, then
    // the first core's `sem_addr` value.
    auto multicast = [&](uint32_t source, uint32_t address, uint32_t bytes, uint32_t sem_addr) {
        for (uint32_t r = 0; r < 2; ++r) {
            const uint32_t dests = rectangle(r, 4);
            if (dests == 0) {
                continue;
            }
            const uint32_t x0 = rectangle(r, 0), y0 = rectangle(r, 1), x1 = rectangle(r, 2), y1 = rectangle(r, 3);
            noc_async_write_multicast(source, get_noc_multicast_addr(x0, y0, x1, y1, address), bytes, dests, true);
#ifdef ARCH_BLACKHOLE
            noc_async_writes_flushed();
#endif
            noc_semaphore_set_multicast(sem_addr, get_noc_multicast_addr(x0, y0, x1, y1, sem_addr), dests);
        }
    };

    auto exchange = [&](uint32_t cb_row, uint32_t cb_gather, uint32_t round) {
        const uint32_t gather = get_write_ptr(cb_gather);
        const uint32_t slot = worker % 8;
        const uint32_t dest = gather + (worker / 8) * f32_tile + 2 * (slot / 4) * f32_face + (slot % 4) * f32_band;
        cb_wait_front(cb_row, 1);
        const uint32_t row = get_read_ptr(cb_row);
        noc_async_write(row, get_noc_addr(gather_x, gather_y, dest), f32_band);
        noc_async_write(row + f32_face, get_noc_addr(gather_x, gather_y, dest + f32_face), f32_band);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(gather_x, gather_y, written_addr), 1);
        cb_pop_front(cb_row, 1);

        cb_reserve_back(cb_gather, gather_tiles);
        if (worker == 0) {
            noc_semaphore_wait_min(written, round * num_cores);
            *valid = 1;
            multicast(gather, gather, gather_tiles * f32_tile, valid_addr);
            noc_async_write_barrier();
        } else {
            noc_semaphore_wait(valid, 1);
            noc_semaphore_set(valid, 0);
        }
        cb_push_back(cb_gather, gather_tiles);
    };

    exchange(cb_row1, cb_gather1, 1);
    if constexpr (has_weight) {
        exchange(cb_row2, cb_gather2, 2);
    }

    const uint32_t tile_bytes = get_tile_size(cb_output);
    const uint32_t face_bytes = tile_bytes / 4;
    const uint32_t row_bytes = face_bytes / 16;
    const auto output = TensorAccessor(output_args, output_addr, tile_bytes);
    cb_wait_front(cb_output, 1);
    const uint32_t stage = get_read_ptr(cb_output);
    for (uint32_t q = 0; q < num_queries; ++q) {
        for (uint32_t c = 0; c < count; ++c) {
            const uint32_t u = q * count + c;
            const uint32_t source = stage + 2 * (u / 4) * face_bytes + 4 * (u % 4) * row_bytes;
            const uint64_t dest = output.get_noc_addr(q * Wt + start + c);
            noc_async_write(source, dest, tokens * row_bytes);
            noc_async_write(source + face_bytes, dest + face_bytes, tokens * row_bytes);
        }
    }
    noc_async_write_barrier();
    cb_pop_front(cb_output, 1);
}
