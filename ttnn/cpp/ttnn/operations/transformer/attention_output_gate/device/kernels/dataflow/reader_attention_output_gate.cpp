// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// The byte offset of the left half of row `row` of a BF16 tile (16 elements of
// face 0 or 2); its right half follows 512 bytes later, in face 1 or 3.
constexpr uint32_t row_offset(uint32_t row) { return (row / 16) * 1024 + (row % 16) * 32; }

// For each of this core's output tiles (head h, its columns j * 32 on): the
// gate tile holding gate head h's columns, every token's row, and the
// attention rows of head h, assembled as row t of a tile from token t's
// attention tile.
void kernel_main() {
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);
    const uint32_t attention_addr = get_arg_val<uint32_t>(2);
    const uint32_t gate_addr = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_gate = get_named_compile_time_arg_val("cb_gate");
    constexpr uint32_t cb_values = get_named_compile_time_arg_val("cb_values");
    constexpr uint32_t cb_attention = get_named_compile_time_arg_val("cb_attention");
    constexpr uint32_t tokens = get_named_compile_time_arg_val("tokens");
    constexpr uint32_t Dt = get_named_compile_time_arg_val("Dt");
    constexpr uint32_t attention_Ht = get_named_compile_time_arg_val("attention_Ht");
    constexpr uint32_t attention_Wt = get_named_compile_time_arg_val("attention_Wt");
    constexpr uint32_t value_offset_tiles = get_named_compile_time_arg_val("value_offset_tiles");
    constexpr uint32_t gate_offset_tiles = get_named_compile_time_arg_val("gate_offset_tiles");
    constexpr uint32_t gate_stride_tiles = get_named_compile_time_arg_val("gate_stride_tiles");
    constexpr auto attention_args = TensorAccessorArgs<0>();
    constexpr auto gate_args = TensorAccessorArgs<attention_args.next_compile_time_args_offset()>();
    const uint32_t tile = get_tile_size(cb_gate);
    const auto attention = TensorAccessor(attention_args, attention_addr, tile);
    const auto gate = TensorAccessor(gate_args, gate_addr, tile);

    // The tokens' attention tiles land here; only their head's rows move on.
    cb_reserve_back(cb_attention, tokens);
    const uint32_t staged = get_write_ptr(cb_attention);
    for (uint32_t unit = start; unit < start + count; ++unit) {
        const uint32_t head = unit / Dt;
        const uint32_t column = unit % Dt;
        cb_reserve_back(cb_gate, 1);
        cb_reserve_back(cb_values, 1);
        noc_async_read(
            gate.get_noc_addr(head * gate_stride_tiles + gate_offset_tiles + column), get_write_ptr(cb_gate), tile);
        for (uint32_t t = 0; t < tokens; ++t) {
            const uint32_t index = (t * attention_Ht + head / 32) * attention_Wt + value_offset_tiles + column;
            noc_async_read(attention.get_noc_addr(index), staged + t * tile, tile);
        }
        noc_async_read_barrier();
        const uint32_t values = get_write_ptr(cb_values);
        for (uint32_t t = 0; t < tokens; ++t) {
            const uint32_t from = staged + t * tile + row_offset(head % 32);
            const uint32_t to = values + row_offset(t);
            noc_async_read(get_noc_addr(from), to, 32);
            noc_async_read(get_noc_addr(from + 512), to + 512, 32);
        }
        noc_async_read_barrier();
        cb_push_back(cb_gate, 1);
        cb_push_back(cb_values, 1);
    }
}
