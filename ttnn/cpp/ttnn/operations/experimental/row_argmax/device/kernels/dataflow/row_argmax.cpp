// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

// Argmax of rows r < R over this core's tiles of a tiled [..., R, N] tensor:
// the rows' face pieces of a batch of tiles come in behind one barrier and
// are scanned in index order, comparing order-preserving unsigned keys of the
// values with a strict >, so the first maximum stays. Every core sends its
// rows' (key, index) to the first core, which merges them, ties going to the
// smallest index, and writes the indices.

#include <stdint.h>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"

namespace {

// Negative floats reversed below the positive ones: unsigned order is float
// order. -0.0 takes +0.0's key, one above its own, and every NaN the largest,
// above +inf, as torch's and JAX's argmax take the first NaN. Must match
// key_of in row_argmax.cpp, which keys the floor.
FORCE_INLINE uint32_t key_of(uint32_t bits) {
    uint32_t key = (bits & 0x80000000u) != 0 ? ~bits : bits | 0x80000000u;
    key += bits == 0x80000000u;
    return key | (0u - static_cast<uint32_t>((bits & 0x7FFFFFFFu) > 0x7F800000u));
}

}  // namespace

void kernel_main() {
    constexpr uint32_t rows = get_compile_time_arg_val(0);
    constexpr uint32_t element_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t batch_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t num_cores = get_compile_time_arg_val(3);
    // The floor's key, or 0, the lowest: with index 0 and a strict >, the first
    // maximum wins, and index 0 where no element exceeds the floor.
    constexpr uint32_t floor_key = get_compile_time_arg_val(4);
    constexpr auto input_args = TensorAccessorArgs<5>();
    constexpr auto output_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr uint32_t face_bytes = 256 * element_bytes;
    constexpr uint32_t row_bytes = 16 * element_bytes;
    constexpr uint32_t tile_bytes = 4 * face_bytes;
    constexpr uint32_t top_rows = rows < 16 ? rows : 16;
    constexpr uint32_t bottom_rows = rows - top_rows;
    // DRAM reads move whole 64-byte pieces: a face's first rows, rounded up.
    constexpr auto piece_bytes = [](uint32_t face_rows) { return (face_rows * row_bytes + 63) / 64 * 64; };

    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);
    const uint32_t core = get_arg_val<uint32_t>(2);
    const uint32_t leader_x = get_arg_val<uint32_t>(3);
    const uint32_t leader_y = get_arg_val<uint32_t>(4);
    const auto input = TensorAccessor(input_args, get_arg_val<uint32_t>(5), tile_bytes);
    const auto output = TensorAccessor(output_args, get_arg_val<uint32_t>(6), rows * sizeof(uint32_t));

    CircularBuffer scratch(tt::CBIndex::c_0);
    CircularBuffer slots(tt::CBIndex::c_1);
    Noc noc;
    scratch.reserve_back(batch_tiles);
    slots.reserve_back(1);
    const uint32_t base = scratch.get_write_ptr();

    uint32_t best_key[rows];
    uint32_t best_index[rows];
    for (uint32_t row = 0; row < rows; ++row) {
        best_key[row] = floor_key;
        best_index[row] = 0;
    }

    for (uint32_t first = start; first < start + count; first += batch_tiles) {
        const uint32_t tiles = start + count - first < batch_tiles ? start + count - first : batch_tiles;
        for (uint32_t i = 0; i < tiles; ++i) {
            for (uint32_t face = 0; face < 4; ++face) {
                const uint32_t face_rows = face < 2 ? top_rows : bottom_rows;
                if (face_rows != 0) {
                    noc.async_read(
                        input, scratch, piece_bytes(face_rows),
                        {.page_id = first + i, .offset_bytes = face * face_bytes},
                        {.offset_bytes = i * tile_bytes + face * face_bytes});
                }
            }
        }
        noc.async_read_barrier();
        for (uint32_t i = 0; i < tiles; ++i) {
            for (uint32_t row = 0; row < rows; ++row) {
                uint32_t key = best_key[row];
                uint32_t index = best_index[row];
                const uint32_t column0 = (first + i) * 32;
                // Row r of a tile is row r % 16 of its faces 2 (r / 16) and 2 (r / 16) + 1.
                for (uint32_t half = 0; half < 2; ++half) {
                    const uint32_t address =
                        base + i * tile_bytes + ((row / 16) * 2 + half) * face_bytes + (row % 16) * row_bytes;
                    if constexpr (element_bytes == 4) {
                        const volatile tt_l1_ptr uint32_t* values =
                            reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(address);
#pragma GCC unroll 16
                        for (uint32_t column = 0; column < 16; ++column) {
                            const uint32_t candidate = key_of(values[column]);
                            if (candidate > key) {
                                key = candidate;
                                index = column0 + half * 16 + column;
                            }
                        }
                    } else {
                        const volatile tt_l1_ptr uint16_t* values =
                            reinterpret_cast<const volatile tt_l1_ptr uint16_t*>(address);
#pragma GCC unroll 16
                        for (uint32_t column = 0; column < 16; ++column) {
                            const uint32_t candidate = key_of(static_cast<uint32_t>(values[column]) << 16);
                            if (candidate > key) {
                                key = candidate;
                                index = column0 + half * 16 + column;
                            }
                        }
                    }
                }
                best_key[row] = key;
                best_index[row] = index;
            }
        }
    }

    // This core's (key, index) pairs into its slot on the first core.
    volatile tt_l1_ptr uint32_t* all = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slots.get_write_ptr());
    const uint32_t semaphore = get_semaphore(0);
    if (core != 0) {
        // Stage the pairs in this core's own (unused) slot of the same buffer.
        for (uint32_t row = 0; row < rows; ++row) {
            all[core * rows * 2 + 2 * row] = best_key[row];
            all[core * rows * 2 + 2 * row + 1] = best_index[row];
        }
        const uint32_t offset = core * rows * 2 * sizeof(uint32_t);
        noc_async_write(
            slots.get_write_ptr() + offset, get_noc_addr(leader_x, leader_y, slots.get_write_ptr() + offset),
            rows * 2 * sizeof(uint32_t));
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(leader_x, leader_y, semaphore), 1);
        noc_async_atomic_barrier();
        return;
    }
    volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore);
    noc_semaphore_wait(arrived, num_cores - 1);
    noc_semaphore_set(arrived, 0);
    for (uint32_t row = 0; row < rows; ++row) {
        uint32_t key = best_key[row];
        uint32_t index = best_index[row];
        for (uint32_t other = 1; other < num_cores; ++other) {
            const uint32_t other_key = all[other * rows * 2 + 2 * row];
            const uint32_t other_index = all[other * rows * 2 + 2 * row + 1];
            if (other_key > key || (other_key == key && other_index < index)) {
                key = other_key;
                index = other_index;
            }
        }
        // The indices, in this core's first slot words, out as the row-major output.
        all[row] = index;
    }
    noc_async_write(slots.get_write_ptr(), output.get_noc_addr(0), rows * sizeof(uint32_t));
    noc_async_write_barrier();
}
