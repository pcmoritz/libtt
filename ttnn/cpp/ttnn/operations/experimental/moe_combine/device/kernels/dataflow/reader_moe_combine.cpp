// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// The placement tile P[i, j * T + i] = weights[i, j] (FP32), then for each of
// this core's tile columns of H the slot rows' 32 values as a BF16 tile: each
// row read whole (64 aligned bytes of its row-major page) into a staging row,
// its two halves copied to their faces.

namespace {

constexpr uint32_t offset_of(uint32_t row, uint32_t col, uint32_t elem) {
    return ((row / 16) * 2 + (col / 16)) * 256 * elem + ((row % 16) * 16 + col % 16) * elem;
}

void zero(uint32_t addr, uint32_t bytes) {
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t done = 0; done < bytes; done += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, addr + done, MEM_ZEROS_SIZE);
    }
}

}  // namespace

void kernel_main() {
    const uint32_t core = get_arg_val<uint32_t>(0);
    const uint32_t slots_addr = get_arg_val<uint32_t>(1);
    const uint32_t weights_addr = get_arg_val<uint32_t>(2);
    constexpr uint32_t cb_weights = get_named_compile_time_arg_val("cb_weights");
    constexpr uint32_t cb_placement = get_named_compile_time_arg_val("cb_placement");
    constexpr uint32_t cb_rows = get_named_compile_time_arg_val("cb_rows");
    constexpr uint32_t cb_slots = get_named_compile_time_arg_val("cb_slots");
    constexpr uint32_t T = get_named_compile_time_arg_val("tokens");
    constexpr uint32_t k = get_named_compile_time_arg_val("experts");
    constexpr uint32_t R = get_named_compile_time_arg_val("slot_rows");
    constexpr uint32_t row_bytes = get_named_compile_time_arg_val("row_bytes");
    constexpr uint32_t num_cores = get_named_compile_time_arg_val("num_cores");
    constexpr uint32_t Ht = get_named_compile_time_arg_val("Ht");
    constexpr bool fp32_weights = get_named_compile_time_arg_val("fp32_weights") != 0;
    constexpr auto slots_args = TensorAccessorArgs<0>();
    constexpr auto weights_args = TensorAccessorArgs<slots_args.next_compile_time_args_offset()>();
    constexpr uint32_t bf16_tile = 2048;
    constexpr uint32_t f32_tile = 4096;
    const auto slots = TensorAccessor(slots_args, slots_addr, row_bytes);
    const auto weights = TensorAccessor(weights_args, weights_addr, fp32_weights ? f32_tile : bf16_tile);

    // The placement.
    cb_reserve_back(cb_weights, 1);
    cb_reserve_back(cb_placement, 1);
    const uint32_t w = get_write_ptr(cb_weights);
    const uint32_t p = get_write_ptr(cb_placement);
    noc_async_read(weights.get_noc_addr(0), w, fp32_weights ? f32_tile : bf16_tile);
    zero(p, f32_tile);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* placement = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(p);
    for (uint32_t i = 0; i < T; ++i) {
        for (uint32_t j = 0; j < k; ++j) {
            uint32_t bits;
            if constexpr (fp32_weights) {
                bits = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(w)[offset_of(i, j, 4) / 4];
            } else {
                bits = static_cast<uint32_t>(reinterpret_cast<volatile tt_l1_ptr uint16_t*>(w)[offset_of(i, j, 2) / 2])
                       << 16;
            }
            placement[offset_of(i, j * T + i, 4) / 4] = bits;
        }
    }
    cb_push_back(cb_placement, 1);

    // The slot tiles. Rows from R on stay zero: P's columns there are zero,
    // but zero times a stale NaN would not be.
    const uint32_t rows = get_write_ptr(cb_rows);
    for (uint32_t c = core; c < Ht; c += num_cores) {
        cb_reserve_back(cb_slots, 1);
        const uint32_t s = get_write_ptr(cb_slots);
        if constexpr (R < 32) {
            zero(s, bf16_tile);
        }
        for (uint32_t r = 0; r < R; ++r) {
            noc_async_read(slots.get_noc_addr(r, c * 64), rows + r * 64, 64);
        }
        noc_async_read_barrier();
        for (uint32_t r = 0; r < R; ++r) {
            const uint32_t face = s + (r / 16) * 2 * 512 + (r % 16) * 32;
            noc_async_read(get_noc_addr(rows + r * 64), face, 32);
            noc_async_read(get_noc_addr(rows + r * 64 + 32), face + 512, 32);
        }
        noc_async_read_barrier();
        cb_push_back(cb_slots, 1);
    }
}
