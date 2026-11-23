// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// The exchanges between cores and the output tiles.
//
// A gather: the compute kernel packs tiles whose rows 0..3 (one per token) of
// faces 0 and 1 carry this core's partial sums; each goes to a slot s of one
// of the first core's gather tiles (8 slots a tile: faces 2 * (s / 4) and the
// next, rows 4 * (s % 4) ..), and the core counts itself in on the first
// core's `written` semaphore. The first exchange (sums of squares) is
// multicast back raw; for the second (the down projection's shares) the
// first core's compute kernel sums the slots into the hidden tiles, which
// its writer multicasts.
void kernel_main() {
    const uint32_t worker = get_arg_val<uint32_t>(0);
    const uint32_t start = get_arg_val<uint32_t>(1);
    const uint32_t gather_x = get_arg_val<uint32_t>(2);
    const uint32_t gather_y = get_arg_val<uint32_t>(3);
    constexpr uint32_t rectangles_arg = 4;
    // Then the output addresses: [Q, T, W], or one [T, W] per destination.
    constexpr uint32_t outputs_arg = 14;

    constexpr uint32_t cb_row1 = get_named_compile_time_arg_val("cb_row1");
    constexpr uint32_t cb_gather1 = get_named_compile_time_arg_val("cb_gather1");
    constexpr uint32_t cb_partial = get_named_compile_time_arg_val("cb_partial");
    constexpr uint32_t cb_gather2 = get_named_compile_time_arg_val("cb_gather2");
    constexpr uint32_t cb_hidden = get_named_compile_time_arg_val("cb_hidden");
    constexpr uint32_t cb_output = get_named_compile_time_arg_val("cb_output");
    constexpr uint32_t Q = get_named_compile_time_arg_val("num_queries");
    constexpr uint32_t Rt = get_named_compile_time_arg_val("Rt");
    constexpr uint32_t Wt = get_named_compile_time_arg_val("Wt");
    constexpr uint32_t num_cores = get_named_compile_time_arg_val("num_cores");
    constexpr uint32_t gather1_tiles = get_named_compile_time_arg_val("gather1_tiles");
    constexpr uint32_t gather2_tiles = get_named_compile_time_arg_val("gather2_tiles");
    constexpr uint32_t written_sem = get_named_compile_time_arg_val("written_sem");
    constexpr uint32_t valid_sem = get_named_compile_time_arg_val("valid_sem");
    constexpr uint32_t cb_row3 = get_named_compile_time_arg_val("cb_row3");
    constexpr uint32_t cb_gather3 = get_named_compile_time_arg_val("cb_gather3");
    constexpr bool has_norm = get_named_compile_time_arg_val("has_norm") != 0;
    constexpr auto output_args = TensorAccessorArgs<0>();
    constexpr uint32_t f32_tile = 4096, f32_face = 1024, f32_band = 256;
    constexpr uint32_t cnt = 2;

    const uint32_t written_addr = get_semaphore(written_sem);
    const uint32_t valid_addr = get_semaphore(valid_sem);
    volatile tt_l1_ptr uint32_t* written = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(written_addr);
    volatile tt_l1_ptr uint32_t* valid = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(valid_addr);
    auto rectangle = [&](uint32_t r, uint32_t field) { return get_arg_val<uint32_t>(rectangles_arg + 5 * r + field); };
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
    // Tile `tile` of cb's front, rows 0..3 of faces 0 and 1, to slot s (0..7)
    // of gather tile g.
    auto send_slot = [&](uint32_t cb, uint32_t tile, uint32_t gather, uint32_t g, uint32_t s) {
        const uint32_t dest = gather + g * f32_tile + 2 * (s / 4) * f32_face + (s % 4) * f32_band;
        const uint32_t source = get_read_ptr(cb) + tile * f32_tile;
        noc_async_write(source, get_noc_addr(gather_x, gather_y, dest), f32_band);
        noc_async_write(source + f32_face, get_noc_addr(gather_x, gather_y, dest + f32_face), f32_band);
    };
    // The first core's `valid` value multicast after `bytes` of `cb`, or the
    // other cores' wait for it.
    auto distribute = [&](uint32_t cb, uint32_t tiles, uint32_t bytes) {
        cb_reserve_back(cb, tiles);
        if (worker == 0) {
            *valid = 1;
            multicast(get_write_ptr(cb), get_write_ptr(cb), bytes, valid_addr);
            noc_async_write_barrier();
        } else {
            noc_semaphore_wait(valid, 1);
            noc_semaphore_set(valid, 0);
        }
        cb_push_back(cb, tiles);
    };

    // Exchange 1: the sums of squares.
    const uint32_t gather1 = get_write_ptr(cb_gather1);
    cb_wait_front(cb_row1, 1);
    send_slot(cb_row1, 0, gather1, worker / 8, worker % 8);
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(gather_x, gather_y, written_addr), 1);
    cb_pop_front(cb_row1, 1);
    if (worker == 0) {
        noc_semaphore_wait_min(written, num_cores);
    }
    distribute(cb_gather1, gather1_tiles, gather1_tiles * f32_tile);

    // Exchange 2: the down projection's shares, summed by the first core.
    // Share j of core i goes to slot i % 8 of gather tile (i / 8) * Q Rt + j.
    const uint32_t gather2 = get_write_ptr(cb_gather2);
    cb_wait_front(cb_partial, Q * Rt);
    for (uint32_t j = 0; j < Q * Rt; ++j) {
        send_slot(cb_partial, j, gather2, (worker / 8) * Q * Rt + j, worker % 8);
    }
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(gather_x, gather_y, written_addr), 1);
    cb_pop_front(cb_partial, Q * Rt);
    if (worker == 0) {
        cb_reserve_back(cb_gather2, gather2_tiles);
        noc_semaphore_wait_min(written, 2 * num_cores);
        cb_push_back(cb_gather2, gather2_tiles);
        // The compute kernel's hidden tiles, then to every other core.
        cb_wait_front(cb_hidden, Q * Rt);
        *valid = 1;
        multicast(get_read_ptr(cb_hidden), get_read_ptr(cb_hidden), Q * Rt * get_tile_size(cb_hidden), valid_addr);
        noc_async_write_barrier();
    } else {
        cb_reserve_back(cb_hidden, Q * Rt);
        noc_semaphore_wait(valid, 1);
        noc_semaphore_set(valid, 0);
        cb_push_back(cb_hidden, Q * Rt);
    }

    // Exchange 3 (with a norm): the outputs' sums of squares, multicast back raw.
    if constexpr (has_norm) {
        const uint32_t gather3 = get_write_ptr(cb_gather3);
        cb_wait_front(cb_row3, 1);
        send_slot(cb_row3, 0, gather3, worker / 8, worker % 8);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(gather_x, gather_y, written_addr), 1);
        cb_pop_front(cb_row3, 1);
        if (worker == 0) {
            noc_semaphore_wait_min(written, 3 * num_cores);
        }
        distribute(cb_gather3, gather1_tiles, gather1_tiles * f32_tile);
    }

    // The output tiles, q-major: tile (q, start + c) of [Q, T, W], or tile
    // start + c of output q.
    const uint32_t tile_bytes = get_tile_size(cb_output);
    for (uint32_t q = 0; q < Q; ++q) {
        const auto output = TensorAccessor(
            output_args, get_arg_val<uint32_t>(outputs_arg + (has_norm ? q : 0)), tile_bytes);
        for (uint32_t c = 0; c < cnt; ++c) {
            cb_wait_front(cb_output, 1);
            noc_async_write(
                get_read_ptr(cb_output), output.get_noc_addr((has_norm ? 0 : q * Wt) + start + c), tile_bytes);
            noc_async_write_barrier();
            cb_pop_front(cb_output, 1);
        }
    }
}
