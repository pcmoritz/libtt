// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"

// The writes of a gate core of moe_router_topk (see router_gate.cpp). Each
// core of a column of experts sends row r of its partial logits into row
// `slot` of gather tile r on the column's owner, its core of slot 0. The owner
// zeroes the gather rows that no core sends, hands the gather tiles to its
// compute kernel once every partial has landed, and sends row r of the sums
// into the logits tiles of the top-k core of row r (see router_topk.cpp).
// Every circular buffer has the same address on every core of the op.
//
// A 32-bit tile holds four 16 x 16 faces, row-major, of 1 KB each: row r of
// its left or right half is the 64 bytes at face_row_offset(r, half).
namespace {
constexpr uint32_t face_row_offset(uint32_t row, uint32_t half) {
    return ((row / 16) * 2 + half) * 1024 + (row % 16) * 64;
}
}  // namespace

void kernel_main() {
    const uint32_t owner_x = get_arg_val<uint32_t>(0);
    const uint32_t owner_y = get_arg_val<uint32_t>(1);
    const uint32_t slot = get_arg_val<uint32_t>(2);
    const uint32_t n_tile = get_arg_val<uint32_t>(3);
    // On owners: the NOC coordinates of the top-k cores, x then y.

    constexpr uint32_t rows = get_named_compile_time_arg_val("rows");
    constexpr uint32_t group = get_named_compile_time_arg_val("group");
    constexpr uint32_t gate_kc = get_named_compile_time_arg_val("gate_kc");
    constexpr uint32_t cb_in = get_named_compile_time_arg_val("cb_in");
    constexpr uint32_t cb_partial = get_named_compile_time_arg_val("cb_partial");
    constexpr uint32_t cb_gather = get_named_compile_time_arg_val("cb_gather");
    constexpr uint32_t cb_sums = get_named_compile_time_arg_val("cb_sums");
    constexpr uint32_t gather_sem_id = get_named_compile_time_arg_val("gather_sem");
    constexpr uint32_t logits_sem_id = get_named_compile_time_arg_val("logits_sem");
    constexpr uint32_t tile_bytes = 4096;
    static_assert(gate_kc >= 1 && gate_kc <= 32, "one gather tile row per core");
    const bool owner = slot == 0;
    const uint32_t gather = get_write_ptr(cb_gather);

    if (owner && gate_kc < 32) {
        Noc noc;
        CircularBuffer gather_cb(cb_gather);
        for (uint32_t row = 0; row < rows; ++row) {
            const uint32_t base = row * tile_bytes;
            if constexpr (gate_kc < 16) {
                for (uint32_t face = 0; face < 2; ++face) {
                    noc.async_write_zeros(
                        gather_cb, (16 - gate_kc) * 64, {.offset_bytes = base + face * 1024 + gate_kc * 64});
                }
                noc.async_write_zeros(gather_cb, 2048, {.offset_bytes = base + 2048});
            } else {
                for (uint32_t face = 2; face < 4; ++face) {
                    noc.async_write_zeros(
                        gather_cb, (32 - gate_kc) * 64, {.offset_bytes = base + face * 1024 + (gate_kc - 16) * 64});
                }
            }
        }
        noc.write_zeros_l1_barrier();
    }

    cb_wait_front(cb_partial, 1);
    const uint32_t partial = get_read_ptr(cb_partial);
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t half = 0; half < 2; ++half) {
            noc_async_write(
                partial + face_row_offset(row, half),
                get_noc_addr(owner_x, owner_y, gather + row * tile_bytes + face_row_offset(slot, half)),
                64);
        }
    }
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(owner_x, owner_y, get_semaphore(gather_sem_id)), 1);
    noc_async_atomic_barrier();
    cb_pop_front(cb_partial, 1);
    if (!owner) {
        return;
    }

    volatile tt_l1_ptr uint32_t* gather_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(gather_sem_id));
    noc_semaphore_wait_min(gather_sem, gate_kc);
    noc_semaphore_set(gather_sem, 0);
    cb_reserve_back(cb_gather, rows);
    cb_push_back(cb_gather, rows);

    const uint32_t logits = get_write_ptr(cb_in) + n_tile * tile_bytes;
    cb_wait_front(cb_sums, rows);
    const uint32_t sums = get_read_ptr(cb_sums);
    for (uint32_t row = 0; row < rows; ++row) {
        const uint32_t core = row % group;
        const uint32_t x = get_arg_val<uint32_t>(4 + 2 * core);
        const uint32_t y = get_arg_val<uint32_t>(5 + 2 * core);
        for (uint32_t half = 0; half < 2; ++half) {
            noc_async_write(
                sums + row * tile_bytes + face_row_offset(0, half),
                get_noc_addr(x, y, logits + face_row_offset(row, half)),
                64);
        }
    }
    noc_async_write_barrier();
    for (uint32_t core = 0; core < group && core < rows; ++core) {
        const uint32_t x = get_arg_val<uint32_t>(4 + 2 * core);
        const uint32_t y = get_arg_val<uint32_t>(5 + 2 * core);
        noc_semaphore_inc(get_noc_addr(x, y, get_semaphore(logits_sem_id)), 1);
    }
    noc_async_atomic_barrier();
    cb_pop_front(cb_sums, rows);
}
