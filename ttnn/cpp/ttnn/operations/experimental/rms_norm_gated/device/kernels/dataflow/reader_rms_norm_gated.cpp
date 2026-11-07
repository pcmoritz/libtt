// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// For each of this core's output tiles (row-major units): its group's input
// tiles, its gate tile and its weight tile.
void kernel_main() {
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);
    const uint32_t input_addr = get_arg_val<uint32_t>(2);
    const uint32_t gate_addr = get_arg_val<uint32_t>(3);
    const uint32_t weight_addr = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_input = get_named_compile_time_arg_val("cb_input");
    constexpr uint32_t cb_gate = get_named_compile_time_arg_val("cb_gate");
    constexpr uint32_t cb_weight = get_named_compile_time_arg_val("cb_weight");
    constexpr uint32_t Nt = get_named_compile_time_arg_val("Nt");
    constexpr uint32_t Gt = get_named_compile_time_arg_val("Gt");
    constexpr uint32_t gate_Nt = get_named_compile_time_arg_val("gate_Nt");
    constexpr uint32_t gate_offset_tiles = get_named_compile_time_arg_val("gate_offset_tiles");
    constexpr auto input_args = TensorAccessorArgs<0>();
    constexpr auto gate_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto weight_args = TensorAccessorArgs<gate_args.next_compile_time_args_offset()>();
    const uint32_t input_tile = get_tile_size(cb_input);
    const uint32_t gate_tile = get_tile_size(cb_gate);
    const uint32_t weight_tile = get_tile_size(cb_weight);
    const auto input = TensorAccessor(input_args, input_addr, input_tile);
    const auto gate = TensorAccessor(gate_args, gate_addr, gate_tile);
    const auto weight = TensorAccessor(weight_args, weight_addr, weight_tile);

    for (uint32_t unit = start; unit < start + count; ++unit) {
        const uint32_t row = unit / Nt;
        const uint32_t column = unit % Nt;
        const uint32_t group_start = column / Gt * Gt;
        cb_reserve_back(cb_input, Gt);
        cb_reserve_back(cb_gate, 1);
        cb_reserve_back(cb_weight, 1);
        for (uint32_t k = 0; k < Gt; ++k) {
            noc_async_read(
                input.get_noc_addr(row * Nt + group_start + k), get_write_ptr(cb_input) + k * input_tile, input_tile);
        }
        noc_async_read(
            gate.get_noc_addr(row * gate_Nt + gate_offset_tiles + column), get_write_ptr(cb_gate), gate_tile);
        noc_async_read(weight.get_noc_addr(column % Gt), get_write_ptr(cb_weight), weight_tile);
        noc_async_read_barrier();
        cb_push_back(cb_input, Gt);
        cb_push_back(cb_gate, 1);
        cb_push_back(cb_weight, 1);
    }
}
