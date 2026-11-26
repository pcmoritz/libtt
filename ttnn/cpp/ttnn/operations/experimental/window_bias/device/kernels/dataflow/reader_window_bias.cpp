// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// For each token, key tile `key_tile` of its window: the matmul operands A
// (rel_r's heads as rows, their 16 values in columns 0..15, BF16) and B (the
// tile's 32 table rows in columns 0..15, FP32), and on the sink core each
// token's sink tile (sink[t, h] in row h, column 0).
//
// Tile layout: faces 0..3 are rows 0..15 / 16..31 by columns 0..15 / 16..31,
// each row-major; an FP32 face row (16 values) is 64 bytes, a BF16 one 32.

namespace {

constexpr uint32_t f32_face = 1024;
constexpr uint32_t f32_row = 64;
constexpr uint32_t bf16_face = 512;
constexpr uint32_t bf16_row = 32;

// The byte offset of (row, col) in a tile of elem-byte elements.
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
    const uint32_t key_tile = get_arg_val<uint32_t>(0);
    const bool sinks = get_arg_val<uint32_t>(1) != 0;
    const uint32_t rel_r_addr = get_arg_val<uint32_t>(2);
    const uint32_t table_addr = get_arg_val<uint32_t>(3);
    const uint32_t positions_addr = get_arg_val<uint32_t>(4);
    const uint32_t sink_addr = get_arg_val<uint32_t>(5);
    constexpr uint32_t cb_positions = get_named_compile_time_arg_val("cb_positions");
    constexpr uint32_t cb_rel_r = get_named_compile_time_arg_val("cb_rel_r");
    constexpr uint32_t cb_a = get_named_compile_time_arg_val("cb_a");
    constexpr uint32_t cb_b = get_named_compile_time_arg_val("cb_b");
    constexpr uint32_t cb_sink_in = get_named_compile_time_arg_val("cb_sink_in");
    constexpr uint32_t cb_sink = get_named_compile_time_arg_val("cb_sink");
    constexpr uint32_t tokens = get_named_compile_time_arg_val("tokens");
    constexpr uint32_t heads = get_named_compile_time_arg_val("heads");
    constexpr uint32_t rel_tiles = get_named_compile_time_arg_val("rel_tiles");
    constexpr uint32_t window_tiles = get_named_compile_time_arg_val("window_tiles");
    constexpr bool has_sink = get_named_compile_time_arg_val("has_sink") != 0;
    constexpr uint32_t W = 32 * window_tiles;
    constexpr auto rel_r_args = TensorAccessorArgs<0>();
    constexpr auto table_args = TensorAccessorArgs<rel_r_args.next_compile_time_args_offset()>();
    constexpr auto positions_args = TensorAccessorArgs<table_args.next_compile_time_args_offset()>();
    constexpr auto sink_args = TensorAccessorArgs<positions_args.next_compile_time_args_offset()>();
    constexpr uint32_t bf16_tile = 2048;
    constexpr uint32_t f32_tile = 4096;
    const auto rel_r = TensorAccessor(rel_r_args, rel_r_addr, bf16_tile);
    const auto table = TensorAccessor(table_args, table_addr, f32_tile);
    const auto positions = TensorAccessor(positions_args, positions_addr, f32_tile);

    // The positions (row 0 of tile 0) and rel_r's tile row; the operand
    // tiles' unused rows and columns zero, once (each token rewrites the same
    // rows and the CBs' two pages alternate).
    const uint32_t positions_l1 = get_write_ptr(cb_positions);
    const uint32_t rel_r_l1 = get_write_ptr(cb_rel_r);
    noc_async_read(positions.get_noc_addr(0), positions_l1, f32_tile);
    for (uint32_t c = 0; c < rel_tiles; ++c) {
        noc_async_read(rel_r.get_noc_addr(c), rel_r_l1 + c * bf16_tile, bf16_tile);
    }
    cb_reserve_back(cb_a, 2);
    cb_reserve_back(cb_b, 2);
    zero(get_write_ptr(cb_a), 2 * bf16_tile);
    zero(get_write_ptr(cb_b), 2 * f32_tile);
    uint32_t sink_l1 = 0;
    if (has_sink && sinks) {
        const auto sink = TensorAccessor(sink_args, sink_addr, bf16_tile);
        sink_l1 = get_write_ptr(cb_sink_in);
        noc_async_read(sink.get_noc_addr(0), sink_l1, bf16_tile);
        cb_reserve_back(cb_sink, 2);
        zero(get_write_ptr(cb_sink), 2 * bf16_tile);
    }
    noc_async_read_barrier();

    const volatile tt_l1_ptr int32_t* position = reinterpret_cast<volatile tt_l1_ptr int32_t*>(positions_l1);
    for (uint32_t t = 0; t < tokens; ++t) {
        // A: head h's 16 values (columns 16 h ..) to row h, columns 0..15.
        cb_reserve_back(cb_a, 1);
        const uint32_t a = get_write_ptr(cb_a);
        for (uint32_t h = 0; h < heads; ++h) {
            const uint32_t col = h * 16;
            noc_async_read(
                get_noc_addr(rel_r_l1 + (col / 32) * bf16_tile + offset_of(t, col % 32, 2)),
                a + offset_of(h, 0, 2),
                bf16_row);
        }

        // B: table rows m0 .. m0 + 31, m0 = W - 1 - d + 32 * key_tile for the
        // distance d = p - first of the window's first column; runs of rows
        // contiguous in both the table's and B's faces.
        const int32_t p = position[offset_of(0, t, 4) / 4] < 0 ? 0 : position[offset_of(0, t, 4) / 4];
        const uint32_t pos = static_cast<uint32_t>(p);
        const uint32_t first = pos / 32 + 1 > window_tiles ? pos / 32 + 1 - window_tiles : 0;
        const uint32_t m0 = W - 1 - (pos - 32 * first) + 32 * key_tile;
        cb_reserve_back(cb_b, 1);
        const uint32_t b = get_write_ptr(cb_b);
        for (uint32_t r = 0; r < 32;) {
            const uint32_t m = m0 + r;
            const uint32_t run = 16 - (m % 16) < 16 - (r % 16) ? 16 - (m % 16) : 16 - (r % 16);
            noc_async_read(
                table.get_noc_addr(m / 32, ((m % 32) / 16) * 2 * f32_face + (m % 16) * f32_row),
                b + (r / 16) * 2 * f32_face + (r % 16) * f32_row,
                run * f32_row);
            r += run;
        }
        noc_async_read_barrier();
        cb_push_back(cb_a, 1);
        cb_push_back(cb_b, 1);

        if (has_sink && sinks) {
            // sink[t, h] to row h, column 0.
            cb_reserve_back(cb_sink, 1);
            const volatile tt_l1_ptr uint16_t* from = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(sink_l1);
            volatile tt_l1_ptr uint16_t* to = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_sink));
            for (uint32_t h = 0; h < heads; ++h) {
                to[offset_of(h, 0, 2) / 2] = from[offset_of(t, h, 2) / 2];
            }
            cb_push_back(cb_sink, 1);
        }
    }
}
