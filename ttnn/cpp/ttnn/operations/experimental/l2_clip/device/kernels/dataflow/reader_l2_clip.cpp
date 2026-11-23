// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// For each of this core's output tiles (row-major units): its group's input
// tiles and their scale tiles, and its row of the row scale.
void kernel_main() {
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);
    const uint32_t input_addr = get_arg_val<uint32_t>(2);
    const uint32_t scale_addr = get_arg_val<uint32_t>(3);
    const uint32_t row_scale_addr = get_arg_val<uint32_t>(4);
    const uint32_t gate_addr = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_input = get_named_compile_time_arg_val("cb_input");
    constexpr uint32_t cb_scale = get_named_compile_time_arg_val("cb_scale");
    constexpr uint32_t cb_row_scale = get_named_compile_time_arg_val("cb_row_scale");
    constexpr uint32_t Nt = get_named_compile_time_arg_val("Nt");
    constexpr uint32_t in_Nt = get_named_compile_time_arg_val("in_Nt");
    constexpr uint32_t offset_tiles = get_named_compile_time_arg_val("offset_tiles");
    constexpr uint32_t Gt = get_named_compile_time_arg_val("Gt");
    constexpr bool has_scale = get_named_compile_time_arg_val("has_scale") != 0;
    constexpr bool has_row_scale = get_named_compile_time_arg_val("has_row_scale") != 0;
    constexpr uint32_t cb_gate = get_named_compile_time_arg_val("cb_gate");
    constexpr uint32_t gate_Nt = get_named_compile_time_arg_val("gate_Nt");
    constexpr uint32_t gate_offset_tiles = get_named_compile_time_arg_val("gate_offset_tiles");
    constexpr bool has_gate = get_named_compile_time_arg_val("has_gate") != 0;
    constexpr auto input_args = TensorAccessorArgs<0>();
    constexpr auto scale_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto row_scale_args = TensorAccessorArgs<scale_args.next_compile_time_args_offset()>();
    constexpr auto gate_args = TensorAccessorArgs<row_scale_args.next_compile_time_args_offset()>();
    const uint32_t input_tile = get_tile_size(cb_input);
    const uint32_t scale_tile = get_tile_size(cb_scale);
    const uint32_t row_scale_tile = get_tile_size(cb_row_scale);
    const auto input = TensorAccessor(input_args, input_addr, input_tile);
    const auto scale = TensorAccessor(scale_args, scale_addr, scale_tile);
    const auto row_scale = TensorAccessor(row_scale_args, row_scale_addr, row_scale_tile);
    const uint32_t gate_tile = get_tile_size(cb_gate);
    const auto gate = TensorAccessor(gate_args, gate_addr, gate_tile);

    for (uint32_t unit = start; unit < start + count; ++unit) {
        const uint32_t row = unit / Nt;
        const uint32_t group_start = unit % Nt / Gt * Gt;
        cb_reserve_back(cb_input, Gt);
        for (uint32_t k = 0; k < Gt; ++k) {
            noc_async_read(
                input.get_noc_addr(row * in_Nt + offset_tiles + group_start + k),
                get_write_ptr(cb_input) + k * input_tile,
                input_tile);
        }
        if constexpr (has_scale) {
            cb_reserve_back(cb_scale, Gt);
            for (uint32_t k = 0; k < Gt; ++k) {
                noc_async_read(
                    scale.get_noc_addr(group_start + k), get_write_ptr(cb_scale) + k * scale_tile, scale_tile);
            }
        }
        if constexpr (has_row_scale) {
            cb_reserve_back(cb_row_scale, 1);
            noc_async_read(row_scale.get_noc_addr(row), get_write_ptr(cb_row_scale), row_scale_tile);
        }
        if constexpr (has_gate) {
            cb_reserve_back(cb_gate, 1);
            noc_async_read(
                gate.get_noc_addr(row * gate_Nt + gate_offset_tiles + unit % Nt), get_write_ptr(cb_gate), gate_tile);
        }
        noc_async_read_barrier();
        cb_push_back(cb_input, Gt);
        if constexpr (has_scale) {
            cb_push_back(cb_scale, Gt);
        }
        if constexpr (has_row_scale) {
            cb_push_back(cb_row_scale, 1);
        }
        if constexpr (has_gate) {
            cb_push_back(cb_gate, 1);
        }
    }
}
