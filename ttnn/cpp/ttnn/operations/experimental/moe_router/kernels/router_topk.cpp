// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"

// The data-movement half of moe_router_topk. Each core selects the top k of
// its rows of the single tile row: values compared as unsigned keys that
// order like the floats (the sign bit set on non-negative values, every bit
// flipped on negative ones), insertion keeping the k best, the lower index
// first among equal values. It stores them as FP32 into its values tile,
// padded with -1e30 so that the softmax gives them zero weight, and the
// indices into its indices tile, and copies its rows into the first core's
// tiles. The first core masks the indices, hands the values to its compute
// kernel and writes the indices and the weights.
//
// A tile holds four 16 x 16 faces, row-major: element (r, c) is element
// (r % 16) * 16 + c % 16 of face (r / 16) * 2 + c / 16.
namespace {

template <uint32_t value_bytes>
FORCE_INLINE uint32_t element(uint32_t tiles, uint32_t row, uint32_t col) {
    constexpr uint32_t tile_bytes = 32 * 32 * value_bytes;
    const uint32_t c = col % 32;
    const uint32_t offset = ((row / 16) * 2 + c / 16) * 256 + (row % 16) * 16 + c % 16;
    const uint32_t base = tiles + (col / 32) * tile_bytes;
    if constexpr (value_bytes == 4) {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base)[offset];
    } else {
        return reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base)[offset];
    }
}

FORCE_INLINE void store32(uint32_t tile, uint32_t row, uint32_t col, uint32_t value) {
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile)[((row / 16) * 2 + col / 16) * 256 + (row % 16) * 16 + col % 16] =
        value;
}

// The byte offset in a tile of face row (row % 16) of face (row / 16) * 2 + half: 16 elements.
template <uint32_t bytes>
constexpr uint32_t face_row_offset(uint32_t row, uint32_t half) {
    return (((row / 16) * 2 + half) * 256 + (row % 16) * 16) * bytes;
}

}  // namespace

void kernel_main() {
    const uint32_t logits_addr = get_arg_val<uint32_t>(0);
    const uint32_t weights_addr = get_arg_val<uint32_t>(1);
    const uint32_t indices_addr = get_arg_val<uint32_t>(2);
    const uint32_t mask_addr = get_arg_val<uint32_t>(3);
    const uint32_t fill_addr = get_arg_val<uint32_t>(4);
    const uint32_t first_row = get_arg_val<uint32_t>(5);
    const uint32_t group = get_arg_val<uint32_t>(6);
    const uint32_t leader_x = get_arg_val<uint32_t>(7);
    const uint32_t leader_y = get_arg_val<uint32_t>(8);

    constexpr uint32_t Wt = get_named_compile_time_arg_val("Wt");
    constexpr uint32_t k = get_named_compile_time_arg_val("k");
    constexpr uint32_t width = get_named_compile_time_arg_val("width");
    constexpr uint32_t rows = get_named_compile_time_arg_val("rows");
    constexpr uint32_t value_bytes = get_named_compile_time_arg_val("value_bytes");
    constexpr uint32_t weight_bytes = get_named_compile_time_arg_val("weight_bytes");
    constexpr bool has_mask = get_named_compile_time_arg_val("has_mask") != 0;
    constexpr uint32_t mask_rows = get_named_compile_time_arg_val("mask_rows");
    constexpr uint32_t mask_bytes = get_named_compile_time_arg_val("mask_bytes");
    constexpr bool mask_float = get_named_compile_time_arg_val("mask_float") != 0;
    constexpr uint32_t cb_in = get_named_compile_time_arg_val("cb_in");
    constexpr uint32_t cb_mask = get_named_compile_time_arg_val("cb_mask");
    constexpr uint32_t cb_fill = get_named_compile_time_arg_val("cb_fill");
    constexpr uint32_t cb_scaler = get_named_compile_time_arg_val("cb_scaler");
    constexpr uint32_t cb_values = get_named_compile_time_arg_val("cb_values");
    constexpr uint32_t cb_indices = get_named_compile_time_arg_val("cb_indices");
    constexpr uint32_t cb_weights = get_named_compile_time_arg_val("cb_weights");
    constexpr uint32_t sem_id = get_named_compile_time_arg_val("sem");
    static_assert(k >= 1 && k <= 32 && rows <= 32, "one tile of output");
    constexpr uint32_t padding = 0xF149F2CAu;  // -1e30

    constexpr auto logits_args = TensorAccessorArgs<0>();
    constexpr auto weights_args = TensorAccessorArgs<logits_args.next_compile_time_args_offset()>();
    constexpr auto indices_args = TensorAccessorArgs<weights_args.next_compile_time_args_offset()>();
    const auto logits = TensorAccessor(logits_args, logits_addr);
    const auto weights = TensorAccessor(weights_args, weights_addr);
    const auto indices = TensorAccessor(indices_args, indices_addr);

    const uint32_t in_tiles = get_write_ptr(cb_in);
    const uint32_t values_tile = get_write_ptr(cb_values);
    const uint32_t indices_tile = get_write_ptr(cb_indices);
    const uint32_t in_tile_bytes = get_tile_size(cb_in);
    const bool leader = first_row == 0;
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));

    // Read only this core's rows. DRAM reads stay 64-byte aligned; BF16
    // therefore also fetches the adjacent row.
    for (uint32_t row = first_row; row < rows; row += group) {
        for (uint32_t j = 0; j < Wt; ++j) {
            for (uint32_t half = 0; half < 2; ++half) {
                const uint32_t offset = face_row_offset<value_bytes>(row, half) & ~63u;
                noc_async_read(logits.get_noc_addr(j, offset), in_tiles + j * in_tile_bytes + offset, 64);
            }
        }
    }
    if (leader) {
        if constexpr (has_mask) {
            constexpr auto mask_args = TensorAccessorArgs<indices_args.next_compile_time_args_offset()>();
            constexpr auto fill_args = TensorAccessorArgs<mask_args.next_compile_time_args_offset()>();
            const auto mask = TensorAccessor(mask_args, mask_addr);
            const auto fill = TensorAccessor(fill_args, fill_addr);
            noc_async_read_page(0, mask, get_write_ptr(cb_mask));
            noc_async_read_page(0, fill, get_write_ptr(cb_fill));
        }
        // Zero the rows past the logical ones of the values and indices tiles
        // (no other core writes them) and the scaler tile, with the NOC: a
        // store per element costs more than the rest of the op.
        cb_reserve_back(cb_scaler, 1);
        {
            Noc noc;
            CircularBuffer values(cb_values), indices(cb_indices), scaler(cb_scaler);
            // 32-bit tiles: a face row is 64 bytes, a face 1 KB.
            auto zero_rows = [&](const CircularBuffer& cb) {
                if constexpr (rows < 16) {
                    for (uint32_t face = 0; face < 2; ++face) {
                        noc.async_write_zeros(cb, (16 - rows) * 64, {.offset_bytes = face * 1024 + rows * 64});
                    }
                    noc.async_write_zeros(cb, 2048, {.offset_bytes = 2048});
                } else if constexpr (rows < 32) {
                    for (uint32_t face = 2; face < 4; ++face) {
                        noc.async_write_zeros(cb, (32 - rows) * 64, {.offset_bytes = face * 1024 + (rows - 16) * 64});
                    }
                }
            };
            zero_rows(values);
            zero_rows(indices);
            noc.async_write_zeros(scaler, 4096);
            noc.write_zeros_l1_barrier();
        }
        // A reduce sums a tile scaled by row zero of each face of the scaler
        // tile: FP32 ones.
        volatile tt_l1_ptr uint32_t* scaler = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_scaler));
        for (uint32_t face = 0; face < 4; ++face) {
            for (uint32_t col = 0; col < 16; ++col) {
                scaler[face * 256 + col] = 0x3F800000u;
            }
        }
        cb_push_back(cb_scaler, 1);
    }
    noc_async_read_barrier();

    for (uint32_t row = first_row; row < rows; row += group) {
        uint32_t best_key[k];
        uint32_t best_value[k];
        uint32_t best_index[k];
        uint32_t found = 0;
        for (uint32_t col = 0; col < width; ++col) {
            const uint32_t bits = element<value_bytes>(in_tiles, row, col);
            const uint32_t wide = value_bytes == 4 ? bits : bits << 16;
            const uint32_t key = (wide & 0x80000000u) ? ~wide : (wide | 0x80000000u);
            if (found == k && key <= best_key[k - 1]) {
                continue;
            }
            uint32_t pos = found < k ? found++ : k - 1;
            while (pos > 0 && best_key[pos - 1] < key) {
                best_key[pos] = best_key[pos - 1];
                best_value[pos] = best_value[pos - 1];
                best_index[pos] = best_index[pos - 1];
                --pos;
            }
            best_key[pos] = key;
            best_value[pos] = wide;
            best_index[pos] = col;
        }
        for (uint32_t col = 0; col < 32; ++col) {
            store32(values_tile, row, col, col < found ? best_value[col] : padding);
            store32(indices_tile, row, col, col < found ? best_index[col] : 0);
        }
        if (!leader) {
            for (uint32_t half = 0; half < 2; ++half) {
                const uint32_t v = values_tile + face_row_offset<4>(row, half);
                const uint32_t i = indices_tile + face_row_offset<4>(row, half);
                noc_async_write(v, get_noc_addr(leader_x, leader_y, v), 64);
                noc_async_write(i, get_noc_addr(leader_x, leader_y, i), 64);
            }
        }
    }
    if (!leader) {
        // Every row lands before the leader learns of it.
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(leader_x, leader_y, get_semaphore(sem_id)), 1);
        noc_async_atomic_barrier();
        return;
    }
    if (group > 1) {
        noc_semaphore_wait_min(sem, group - 1);
        noc_semaphore_set(sem, 0);
    }
    cb_reserve_back(cb_values, 1);
    cb_push_back(cb_values, 1);

    if constexpr (has_mask) {
        // A row whose mask is zero gets the fill as every index.
        const uint32_t fill = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_fill));
        const uint32_t mask = get_write_ptr(cb_mask);
        for (uint32_t row = 0; row < rows; ++row) {
            uint32_t bits = element<mask_bytes>(mask, mask_rows == 1 ? 0 : row, 0);
            if constexpr (mask_float) {
                // Either zero.
                bits &= mask_bytes == 4 ? 0x7FFFFFFFu : 0x7FFFu;
            }
            if (bits == 0) {
                for (uint32_t col = 0; col < k; ++col) {
                    store32(indices_tile, row, col, fill);
                }
            }
        }
    }

    // Only the logical rows and first face hold results when they are few.
    auto write = [&](const auto& tensor, uint32_t tile, uint32_t bytes) {
        if (rows <= 4 && k <= 16 && bytes == 4) {
            for (uint32_t row = 0; row < rows; ++row) {
                const uint32_t offset = face_row_offset<4>(row, 0);
                noc_async_write(tile + offset, tensor.get_noc_addr(0, offset), 64);
            }
        } else {
            noc_async_write_page(0, tensor, tile);
        }
    };
    write(indices, indices_tile, 4);
    cb_wait_front(cb_weights, 1);
    write(weights, get_read_ptr(cb_weights), weight_bytes);
    noc_async_write_barrier();
    cb_pop_front(cb_weights, 1);
}
