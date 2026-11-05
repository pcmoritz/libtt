// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"

// Reads each of this core's units: up to `group` heads of one token, which go
// to rows 0.. of head tiles (zero elsewhere), with the norm weight, cos and
// sin for q and k heads. A head's row in a QKV tile is a face row (16 values)
// and the same row of the next face, 512 bytes on; both come in one aligned
// window and are moved into place by the NOC (copies by the RISC cost several
// times more). First it builds the constant tiles of rms_norm and
// rotary_embedding: the reduce scaler, epsilon and -1.
void kernel_main() {
    constexpr uint32_t cb_x = get_named_compile_time_arg_val("cb_x");
    constexpr uint32_t cb_weight = get_named_compile_time_arg_val("cb_weight");
    constexpr uint32_t cb_cos = get_named_compile_time_arg_val("cb_cos");
    constexpr uint32_t cb_sin = get_named_compile_time_arg_val("cb_sin");
    constexpr uint32_t cb_scaler = get_named_compile_time_arg_val("cb_scaler");
    constexpr uint32_t cb_eps = get_named_compile_time_arg_val("cb_eps");
    constexpr uint32_t cb_neg = get_named_compile_time_arg_val("cb_neg");
    constexpr uint32_t cb_v = get_named_compile_time_arg_val("cb_v");
    constexpr uint32_t cb_zero = get_named_compile_time_arg_val("cb_zero");
    constexpr uint32_t cb_scratch = get_named_compile_time_arg_val("cb_scratch");
    constexpr uint32_t head_tiles = get_named_compile_time_arg_val("head_tiles");
    constexpr uint32_t row_bytes = get_named_compile_time_arg_val("row_bytes");
    constexpr uint32_t alignment = get_named_compile_time_arg_val("alignment");
    constexpr uint32_t window = get_named_compile_time_arg_val("window");
    constexpr uint32_t cos_per_token = get_named_compile_time_arg_val("cos_per_token");
    constexpr uint32_t eps_bf16 = get_named_compile_time_arg_val("eps_bf16");
    constexpr uint32_t neg_one_bf16 = get_named_compile_time_arg_val("neg_one_bf16");
    constexpr auto qkv_args = TensorAccessorArgs<0>();
    constexpr auto q_weight_args = TensorAccessorArgs<qkv_args.next_compile_time_args_offset()>();
    constexpr auto k_weight_args = TensorAccessorArgs<q_weight_args.next_compile_time_args_offset()>();
    constexpr auto cos_args = TensorAccessorArgs<k_weight_args.next_compile_time_args_offset()>();
    constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();
    constexpr uint32_t kV = 2;
    constexpr uint32_t kEnd = 0xFFFFFFFF;

    uint32_t arg = 0;
    const uint32_t qkv_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t q_weight_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t k_weight_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t cos_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t sin_addr = get_arg_val<uint32_t>(arg++);

    const uint32_t tile_bytes = get_tile_size(cb_x);
    const uint32_t face_bytes = tile_bytes / 4;
    const auto qkv = TensorAccessor(qkv_args, qkv_addr, tile_bytes);
    const auto q_weight = TensorAccessor(q_weight_args, q_weight_addr, tile_bytes);
    const auto k_weight = TensorAccessor(k_weight_args, k_weight_addr, tile_bytes);
    const auto cos = TensorAccessor(cos_args, cos_addr, tile_bytes);
    const auto sin = TensorAccessor(sin_args, sin_addr, tile_bytes);

    // The constant tiles, zero but for: the scaler's ones in row 0 of each
    // face (prepare_reduce_scaler's), epsilon down column 0
    // (generate_bcast_col_scalar's) and -1 in element 0.
    Noc noc;
    for (uint32_t cb : {cb_zero, cb_scaler, cb_eps, cb_neg}) {
        CircularBuffer buffer(cb);
        buffer.reserve_back(1);
        noc.async_write_zeros(buffer, tile_bytes);
    }
    noc.write_zeros_l1_barrier();
    const uint32_t zeros = get_write_ptr(cb_zero);
    auto* scaler = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_scaler));
    for (uint32_t face = 0; face < 4; ++face) {
        for (uint32_t column = 0; column < 16; ++column) {
            scaler[face * 256 + column] = 0x3F80;
        }
    }
    auto* eps = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_eps));
    for (uint32_t face = 0; face < 4; face += 2) {
        for (uint32_t i = 0; i < 256; i += 16) {
            eps[face * 256 + i] = eps_bf16;
        }
    }
    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_neg))[0] = neg_one_bf16;
    for (uint32_t cb : {cb_zero, cb_scaler, cb_eps, cb_neg}) {
        cb_push_back(cb, 1);
    }

    const uint32_t scratch = (get_write_ptr(cb_scratch) + alignment - 1) & ~(alignment - 1);
    auto read_tiles = [&](const auto& tensor, uint32_t cb, uint32_t first) {
        cb_reserve_back(cb, head_tiles);
        for (uint32_t c = 0; c < head_tiles; ++c) {
            noc_async_read(tensor.get_noc_addr(first + c), get_write_ptr(cb) + c * tile_bytes, tile_bytes);
        }
    };
    for (;;) {
        const uint32_t kind = get_arg_val<uint32_t>(arg++);
        if (kind == kEnd) {
            break;
        }
        const uint32_t token = get_arg_val<uint32_t>(arg++);
        const uint32_t source_head = get_arg_val<uint32_t>(arg++);
        const uint32_t heads = get_arg_val<uint32_t>(arg++);
        arg += 5;  // the writer's

        const uint32_t cb = kind == kV ? cb_v : cb_x;
        cb_reserve_back(cb, head_tiles);
        const uint32_t tiles = get_write_ptr(cb);
        for (uint32_t c = 0; c < head_tiles; ++c) {
            noc_async_read(get_noc_addr(zeros), tiles + c * tile_bytes, tile_bytes);
        }
        // The token's row: in faces 0 and 1 for rows 0-15, 2 and 3 for 16-31.
        const uint32_t row_offset = (token % 16) * row_bytes + (token / 16) * 2 * face_bytes;
        const uint32_t skew = row_offset & (alignment - 1);
        for (uint32_t i = 0; i < heads; ++i) {
            for (uint32_t c = 0; c < head_tiles; ++c) {
                noc_async_read(
                    qkv.get_noc_addr((source_head + i) * head_tiles + c) + row_offset - skew,
                    scratch + (i * head_tiles + c) * window,
                    window);
            }
        }
        if (kind != kV) {
            if (kind == 0) {
                read_tiles(q_weight, cb_weight, 0);
            } else {
                read_tiles(k_weight, cb_weight, 0);
            }
            const uint32_t table_tile = (cos_per_token ? token : 0) * head_tiles;
            read_tiles(cos, cb_cos, table_tile);
            read_tiles(sin, cb_sin, table_tile);
        }
        noc_async_read_barrier();

        // Head i's row goes to row i of the tiles.
        for (uint32_t i = 0; i < heads; ++i) {
            for (uint32_t c = 0; c < head_tiles; ++c) {
                for (uint32_t half = 0; half < 2; ++half) {
                    noc_async_read(
                        get_noc_addr(scratch + (i * head_tiles + c) * window + skew + half * face_bytes),
                        tiles + c * tile_bytes + half * face_bytes + i * row_bytes,
                        row_bytes);
                }
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb, head_tiles);
        if (kind != kV) {
            cb_push_back(cb_weight, head_tiles);
            cb_push_back(cb_cos, head_tiles);
            cb_push_back(cb_sin, head_tiles);
        }
    }
}
