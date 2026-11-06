// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// Writes each of this core's units, rows 0.. of the compute kernel's output
// tiles (of the reader's for v heads), to their rows of the q output or of the
// token's tile row of the kv output, then the rows and tiles of that tile row
// that only zeros fill: the padding rows below the last head and q's padding
// columns.
void kernel_main() {
    constexpr uint32_t cb_v = get_named_compile_time_arg_val("cb_v");
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t cb_zero = get_named_compile_time_arg_val("cb_zero");
    constexpr uint32_t head_tiles = get_named_compile_time_arg_val("head_tiles");
    constexpr uint32_t group = get_named_compile_time_arg_val("group");
    constexpr uint32_t row_bytes = get_named_compile_time_arg_val("row_bytes");
    constexpr auto qkv_args = TensorAccessorArgs<0>();
    constexpr auto q_weight_args = TensorAccessorArgs<qkv_args.next_compile_time_args_offset()>();
    constexpr auto k_weight_args = TensorAccessorArgs<q_weight_args.next_compile_time_args_offset()>();
    constexpr auto cos_args = TensorAccessorArgs<k_weight_args.next_compile_time_args_offset()>();
    constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();
    constexpr auto q_args = TensorAccessorArgs<sin_args.next_compile_time_args_offset()>();
    constexpr auto kv_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr uint32_t kQ = 0;
    constexpr uint32_t kV = 2;
    constexpr uint32_t kEnd = 0xFFFFFFFF;

    const uint32_t tile_bytes = get_tile_size(cb_out);
    const uint32_t face_bytes = tile_bytes / 4;
    cb_wait_front(cb_zero, 1);
    const uint32_t zeros = get_read_ptr(cb_zero);

    uint32_t arg = 5;  // past the reader's tensors
    for (;;) {
        const uint32_t kind = get_arg_val<uint32_t>(arg++);
        if (kind == kEnd) {
            break;
        }
        arg += 4;  // the reader's
        const uint32_t output_addr = get_arg_val<uint32_t>(arg++);
        const uint32_t first_tile = get_arg_val<uint32_t>(arg++);
        const uint32_t first_row = get_arg_val<uint32_t>(arg++);
        const uint32_t zero_rows_from = get_arg_val<uint32_t>(arg++);
        const uint32_t zero_tiles = get_arg_val<uint32_t>(arg++);

        const uint32_t cb = kind == kV ? cb_v : cb_out;
        cb_wait_front(cb, head_tiles);
        const uint32_t tiles = get_read_ptr(cb);
        auto write = [&](const auto& output) {
            // The group's rows lie within one face of each column of faces.
            const uint32_t row_offset = (first_row % 16) * row_bytes + (first_row / 16) * 2 * face_bytes;
            for (uint32_t c = 0; c < head_tiles; ++c) {
                const uint64_t tile = output.get_noc_addr(first_tile + c);
                for (uint32_t half = 0; half < 2; ++half) {
                    noc_async_write(
                        tiles + c * tile_bytes + half * face_bytes,
                        tile + row_offset + half * face_bytes,
                        group * row_bytes);
                }
                for (uint32_t face = 0; face < 4; ++face) {
                    const uint32_t face_row = face / 2 * 16;
                    const uint32_t from = zero_rows_from > face_row ? zero_rows_from : face_row;
                    if (from < face_row + 16) {
                        noc_async_write(
                            zeros,
                            tile + face * face_bytes + (from - face_row) * row_bytes,
                            (face_row + 16 - from) * row_bytes);
                    }
                }
            }
            for (uint32_t t = 0; t < zero_tiles; ++t) {
                noc_async_write(zeros, output.get_noc_addr(first_tile + head_tiles + t), tile_bytes);
            }
        };
        if (kind == kQ) {
            write(TensorAccessor(q_args, output_addr, tile_bytes));
        } else {
            write(TensorAccessor(kv_args, output_addr, tile_bytes));
        }
        noc_async_write_barrier();
        cb_pop_front(cb, head_tiles);
    }
}
