// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"

// Runs on both data movement RISCs of a core, which handles one user, a row of
// the fused QKV projection. Head h of the q heads goes to row h of the head
// tiles and head h of the k heads to row num_q_heads + h, so that one pass of
// the compute kernel normalizes and rotates both; head h of the v heads goes
// to row h of the v output, as nlp_create_qkv_heads_decode places them.
// The heads alternate between the two RISCs. A face row of a head (16
// elements) may be narrower than the DRAM read alignment, so each is read as
// the aligned chunk around it into scratch and then moved into place by the
// NOC; copies by the RISC would cost several times more.
//
// The reader (role 0) also reads the per-row norm weights and pushes the head
// tiles once the writer has placed its heads. The writer (role 1) reads cos
// and sin, builds the reduce scaler, and at the end moves the k rows of the
// compute kernel's output into the k output.
void kernel_main() {
    const uint32_t qkv_addr = get_arg_val<uint32_t>(0);
    const uint32_t weight_addr = get_arg_val<uint32_t>(1);
    const uint32_t cos_addr = get_arg_val<uint32_t>(2);
    const uint32_t sin_addr = get_arg_val<uint32_t>(3);
    // Byte offset of the user's row within face 0 or 2 of a tile.
    const uint32_t row_offset = get_arg_val<uint32_t>(4);

    constexpr uint32_t role = get_named_compile_time_arg_val("role");
    constexpr uint32_t cb_heads = get_named_compile_time_arg_val("cb_heads");
    constexpr uint32_t cb_weight = get_named_compile_time_arg_val("cb_weight");
    constexpr uint32_t cb_cos = get_named_compile_time_arg_val("cb_cos");
    constexpr uint32_t cb_sin = get_named_compile_time_arg_val("cb_sin");
    constexpr uint32_t cb_scaler = get_named_compile_time_arg_val("cb_scaler");
    constexpr uint32_t cb_scratch = get_named_compile_time_arg_val("cb_scratch");
    constexpr uint32_t cb_placed = get_named_compile_time_arg_val("cb_placed");
    constexpr uint32_t cb_k_stage = get_named_compile_time_arg_val("cb_k_stage");
    constexpr uint32_t cb_q_out = get_named_compile_time_arg_val("cb_q_out");
    constexpr uint32_t cb_k_out = get_named_compile_time_arg_val("cb_k_out");
    constexpr uint32_t cb_v_out = get_named_compile_time_arg_val("cb_v_out");
    constexpr uint32_t num_q_heads = get_named_compile_time_arg_val("num_q_heads");
    constexpr uint32_t num_kv_heads = get_named_compile_time_arg_val("num_kv_heads");
    constexpr uint32_t head_tiles = get_named_compile_time_arg_val("head_tiles");
    constexpr uint32_t row_tiles = get_named_compile_time_arg_val("row_tiles");
    constexpr uint32_t q_row_tiles = get_named_compile_time_arg_val("q_row_tiles");
    constexpr uint32_t row_bytes = get_named_compile_time_arg_val("row_bytes");
    constexpr uint32_t alignment = get_named_compile_time_arg_val("alignment");
    constexpr auto qkv_args = TensorAccessorArgs<0>();
    constexpr auto weight_args = TensorAccessorArgs<qkv_args.next_compile_time_args_offset()>();
    constexpr auto cos_args = TensorAccessorArgs<weight_args.next_compile_time_args_offset()>();
    constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();

    constexpr uint32_t num_heads = num_q_heads + 2 * num_kv_heads;
    const uint32_t tile_bytes = get_tile_size(cb_heads);
    const uint32_t face_bytes = tile_bytes / 4;
    // Byte offset of row r of a tile row: faces 0 and 1 hold rows 0-15, 2
    // and 3 rows 16-31.
    auto row_in_tile = [&](uint32_t r) { return (r % 16) * row_bytes + (r / 16) * 2 * face_bytes; };
    const auto qkv = TensorAccessor(qkv_args, qkv_addr, tile_bytes);

    auto read_tiles = [&](const auto& args, uint32_t addr, uint32_t cb, uint32_t count) {
        const auto tensor = TensorAccessor(args, addr, get_tile_size(cb));
        cb_reserve_back(cb, count);
        for (uint32_t i = 0; i < count; ++i) {
            noc_async_read(tensor.get_noc_addr(i), get_write_ptr(cb) + i * get_tile_size(cb), get_tile_size(cb));
        }
    };

    // Read this RISC's heads, every face row as the aligned chunk around it.
    // Only the RISC that pushes a buffer reserves it: a reserve after the
    // other RISC's push would wait for space forever. The head tiles and the
    // v output start at their buffer's base either way.
    if constexpr (role == 0) {
        cb_reserve_back(cb_heads, row_tiles * head_tiles);
    } else {
        cb_reserve_back(cb_v_out, head_tiles);
    }
    const uint32_t scratch = (get_write_ptr(cb_scratch) + alignment - 1) & ~(alignment - 1);
    const uint32_t skew = row_offset & (alignment - 1);
    for (uint32_t head = role; head < num_heads; head += 2) {
        for (uint32_t i = 0; i < head_tiles; ++i) {
            const uint32_t tile = head * head_tiles + i;
            for (uint32_t half = 0; half < 2; ++half) {
                noc_async_read(
                    qkv.get_noc_addr(tile) + row_offset + half * face_bytes - skew,
                    scratch + (2 * tile + half) * alignment,
                    alignment);
            }
        }
    }
    if constexpr (role == 0) {
        read_tiles(weight_args, weight_addr, cb_weight, row_tiles * head_tiles);
    } else {
        read_tiles(cos_args, cos_addr, cb_cos, head_tiles);
        read_tiles(sin_args, sin_addr, cb_sin, head_tiles);
        // The reduce scaler: FP32 ones in row 0 of each face.
        Noc noc;
        CircularBuffer scaler(cb_scaler);
        scaler.reserve_back(1);
        noc.async_write_zeros(scaler, 4096);
        noc.write_zeros_l1_barrier();
        volatile tt_l1_ptr uint32_t* tile = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scaler.get_write_ptr());
        for (uint32_t face = 0; face < 4; ++face) {
            for (uint32_t column = 0; column < 16; ++column) {
                tile[face * 256 + column] = 0x3F800000u;
            }
        }
        scaler.push_back(1);
    }
    noc_async_read_barrier();

    // Move each face row from scratch into place.
    for (uint32_t head = role; head < num_heads; head += 2) {
        // q and k heads are stacked rows of the head tiles; v heads rows of v.
        const bool is_v = head >= num_q_heads + num_kv_heads;
        const uint32_t row = is_v ? head - num_q_heads - num_kv_heads : head;
        const uint32_t dest = (is_v ? get_write_ptr(cb_v_out) : get_write_ptr(cb_heads) +
                                                                  (row / 32) * head_tiles * tile_bytes) +
                              row_in_tile(row % 32);
        for (uint32_t i = 0; i < head_tiles; ++i) {
            const uint32_t tile = head * head_tiles + i;
            for (uint32_t half = 0; half < 2; ++half) {
                noc_async_read(
                    get_noc_addr(scratch + (2 * tile + half) * alignment + skew),
                    dest + i * tile_bytes + half * face_bytes,
                    row_bytes);
            }
        }
    }
    noc_async_read_barrier();

    if constexpr (role == 0) {
        // The head tiles are complete once the writer has placed its heads.
        cb_wait_front(cb_placed, 1);
        cb_pop_front(cb_placed, 1);
        cb_push_back(cb_heads, row_tiles * head_tiles);
        cb_push_back(cb_weight, row_tiles * head_tiles);
        return;
    }
    cb_reserve_back(cb_placed, 1);
    cb_push_back(cb_placed, 1);
    cb_push_back(cb_cos, head_tiles);
    cb_push_back(cb_sin, head_tiles);
    cb_push_back(cb_v_out, head_tiles);

    // Move the k rows of the output, rows num_q_heads.. of the stacked tiles,
    // to the k output. Row tiles below q_row_tiles went to the q output (its
    // padding rows), the rest to the k stage.
    cb_wait_front(cb_q_out, q_row_tiles * head_tiles);
    if constexpr (row_tiles > q_row_tiles) {
        cb_wait_front(cb_k_stage, (row_tiles - q_row_tiles) * head_tiles);
    }
    cb_reserve_back(cb_k_out, ((num_kv_heads + 31) / 32) * head_tiles);
    for (uint32_t h = 0; h < num_kv_heads; ++h) {
        const uint32_t row = num_q_heads + h;
        const uint32_t source = (row / 32 < q_row_tiles
                                     ? get_read_ptr(cb_q_out) + (row / 32) * head_tiles * tile_bytes
                                     : get_read_ptr(cb_k_stage) + (row / 32 - q_row_tiles) * head_tiles * tile_bytes) +
                                row_in_tile(row % 32);
        const uint32_t dest = get_write_ptr(cb_k_out) + (h / 32) * head_tiles * tile_bytes + row_in_tile(h % 32);
        for (uint32_t i = 0; i < head_tiles; ++i) {
            for (uint32_t half = 0; half < 2; ++half) {
                const uint32_t offset = i * tile_bytes + half * face_bytes;
                noc_async_read(get_noc_addr(source + offset), dest + offset, row_bytes);
            }
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_k_out, ((num_kv_heads + 31) / 32) * head_tiles);
}
