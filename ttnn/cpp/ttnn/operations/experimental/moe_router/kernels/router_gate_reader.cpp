// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// The reads of a gate core of moe_router_topk (see router_gate.cpp): its
// gate_kt tiles of the input's single tile row from k_start on, and the
// weight's tiles of the same K rows in column n_tile.
void kernel_main() {
    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t weight_addr = get_arg_val<uint32_t>(1);
    const uint32_t k_start = get_arg_val<uint32_t>(2);
    const uint32_t n_tile = get_arg_val<uint32_t>(3);

    constexpr uint32_t gate_kt = get_named_compile_time_arg_val("gate_kt");
    constexpr uint32_t Wt = get_named_compile_time_arg_val("Wt");
    constexpr uint32_t cb_x = get_named_compile_time_arg_val("cb_x");
    constexpr uint32_t cb_w = get_named_compile_time_arg_val("cb_w");

    constexpr auto input_args = TensorAccessorArgs<0>();
    constexpr auto weight_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    const auto input = TensorAccessor(input_args, input_addr);
    const auto weight = TensorAccessor(weight_args, weight_addr);

    cb_reserve_back(cb_w, gate_kt);
    cb_reserve_back(cb_x, gate_kt);
    const uint32_t w_ptr = get_write_ptr(cb_w);
    const uint32_t x_ptr = get_write_ptr(cb_x);
    const uint32_t w_bytes = get_tile_size(cb_w);
    const uint32_t x_bytes = get_tile_size(cb_x);
    for (uint32_t k = 0; k < gate_kt; ++k) {
        noc_async_read_page((k_start + k) * Wt + n_tile, weight, w_ptr + k * w_bytes);
    }
    for (uint32_t k = 0; k < gate_kt; ++k) {
        noc_async_read_page(k_start + k, input, x_ptr + k * x_bytes);
    }
    noc_async_read_barrier();
    cb_push_back(cb_x, gate_kt);
    cb_push_back(cb_w, gate_kt);
}
