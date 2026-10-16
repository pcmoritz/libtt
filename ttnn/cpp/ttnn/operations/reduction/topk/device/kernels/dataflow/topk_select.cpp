// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"

// TopK's selection path (see topk_select.hpp). A tile row's cores split its
// width into segments and its rows into groups. Each core reads the face rows
// of its rows of its segment and keeps the best k of each row. With one
// segment it formats the row itself; otherwise it sends the k candidates to
// the group's first-segment core, which merges a row's segments. Either way the
// finished row goes straight into the tile row's first core's output tiles,
// which that core writes out once every group is in.
//
// Values compare as unsigned keys in IEEE total order (+NaN above +inf, -0.0
// below +0.0, as lax.top_k orders them): the sign bit set on non-negative
// values, every bit flipped on negative ones, and all of them flipped for the
// smallest k. Among equal keys the lower index comes first.
//
// A tile holds four 16 x 16 faces, row-major: element (r, c) is element
// (r % 16) * 16 + c % 16 of face (r / 16) * 2 + c / 16.
namespace {

constexpr uint32_t k = get_named_compile_time_arg_val("k");
constexpr bool largest = get_named_compile_time_arg_val("largest") != 0;
constexpr uint32_t value_bytes = get_named_compile_time_arg_val("value_bytes");
constexpr uint32_t index_bytes = get_named_compile_time_arg_val("index_bytes");
constexpr uint32_t value_tile_bytes = get_named_compile_time_arg_val("value_tile_bytes");
constexpr uint32_t index_tile_bytes = get_named_compile_time_arg_val("index_tile_bytes");
constexpr uint32_t k_tiles = get_named_compile_time_arg_val("k_tiles");
constexpr uint32_t height = get_named_compile_time_arg_val("height");
constexpr uint32_t width = get_named_compile_time_arg_val("width");
constexpr uint32_t Ht = get_named_compile_time_arg_val("Ht");
constexpr uint32_t Wt = get_named_compile_time_arg_val("Wt");
constexpr uint32_t segments = get_named_compile_time_arg_val("segments");
constexpr uint32_t segment_tiles = get_named_compile_time_arg_val("segment_tiles");
constexpr uint32_t row_groups = get_named_compile_time_arg_val("row_groups");
constexpr uint32_t cb_in = get_named_compile_time_arg_val("cb_in");
constexpr uint32_t cb_candidates = get_named_compile_time_arg_val("cb_candidates");
constexpr uint32_t cb_rows = get_named_compile_time_arg_val("cb_rows");
constexpr uint32_t cb_values = get_named_compile_time_arg_val("cb_values");
constexpr uint32_t cb_indices = get_named_compile_time_arg_val("cb_indices");
constexpr uint32_t merge_sem_id = get_named_compile_time_arg_val("merge_sem");
constexpr uint32_t owner_sem_id = get_named_compile_time_arg_val("owner_sem");

constexpr uint32_t read_bytes = 64;
// Lists are written core to core, which needs 16-byte alignment: an even
// number of 8-byte candidates.
constexpr uint32_t k_stride = (k + 1) & ~1u;
// A row's output: two face-row chunks of 16 elements per tile, values then
// indices, each chunk staged in 64 bytes.
constexpr uint32_t chunk_bytes = 64;
constexpr uint32_t row_bytes = 2 * k_tiles * 2 * chunk_bytes;
// A list padded past the elements its segment has: below every element in
// the merge, since a real element with key 0 has the lower index.
constexpr uint32_t no_index = 0xFFFFFFFFu;

struct Candidate {
    uint32_t key;
    uint32_t index;
};

FORCE_INLINE uint32_t key_of(uint32_t bits) {
    const uint32_t key = bits ^ (static_cast<uint32_t>(static_cast<int32_t>(bits) >> 31) | 0x80000000u);
    return largest ? key : ~key;
}

FORCE_INLINE uint32_t bits_of(uint32_t key) {
    if constexpr (!largest) {
        key = ~key;
    }
    return key ^ ((key & 0x80000000u) ? 0x80000000u : 0xFFFFFFFFu);
}

FORCE_INLINE bool better(uint32_t key, uint32_t index, const Candidate& other) {
    return key > other.key || (key == other.key && index < other.index);
}

// Inserts at or above position pos of a best-first list, dropping its entry.
FORCE_INLINE void insert(Candidate* list, uint32_t pos, uint32_t key, uint32_t index) {
    while (pos > 0 && better(key, index, list[pos - 1])) {
        list[pos] = list[pos - 1];
        --pos;
    }
    list[pos] = {key, index};
}

template <uint32_t bytes>
FORCE_INLINE void store(uint32_t addr, uint32_t i, uint32_t value) {
    if constexpr (bytes == 4) {
        reinterpret_cast<tt_l1_ptr uint32_t*>(addr)[i] = value;
    } else {
        reinterpret_cast<tt_l1_ptr uint16_t*>(addr)[i] = value;
    }
}

// Stages row `row`'s best k as face-row chunks and writes them into the first
// core's output tiles. Chunks past k hold zeros.
FORCE_INLINE void write_row(const Candidate* best, uint32_t row, uint32_t stage, uint64_t values_tiles,
                            uint64_t indices_tiles) {
    for (uint32_t tile = 0; tile < k_tiles; ++tile) {
        for (uint32_t half = 0; half < 2; ++half) {
            const uint32_t chunk = tile * 2 + half;
            const uint32_t values_chunk = stage + chunk * chunk_bytes;
            const uint32_t indices_chunk = stage + (k_tiles * 2 + chunk) * chunk_bytes;
            for (uint32_t c = 0; c < 16; ++c) {
                const uint32_t i = tile * 32 + half * 16 + c;
                const bool found = i < k;
                store<value_bytes>(values_chunk, c, found ? (value_bytes == 4 ? bits_of(best[i].key) : bits_of(best[i].key) >> 16) : 0);
                store<index_bytes>(indices_chunk, c, found ? best[i].index : 0);
            }
            const uint32_t face_row = ((row / 16) * 2 + half) * 256 + (row % 16) * 16;
            noc_async_write(values_chunk, values_tiles + tile * value_tile_bytes + face_row * value_bytes, 16 * value_bytes);
            noc_async_write(
                indices_chunk, indices_tiles + tile * index_tile_bytes + face_row * index_bytes, 16 * index_bytes);
        }
    }
}

// Zeroes the face rows of rows [rows, 32) of `tiles` tiles of `bytes`-wide elements.
template <uint32_t bytes, uint32_t tile_bytes>
FORCE_INLINE void zero_padding_rows(const Noc& noc, uint32_t cb, uint32_t rows) {
    CircularBuffer buffer(cb);
    for (uint32_t tile = 0; tile < k_tiles; ++tile) {
        const uint32_t base = tile * tile_bytes;
        if (rows < 16) {
            for (uint32_t face = 0; face < 2; ++face) {
                noc.async_write_zeros(buffer, (16 - rows) * 16 * bytes, {.offset_bytes = base + (face * 256 + rows * 16) * bytes});
            }
            noc.async_write_zeros(buffer, 512 * bytes, {.offset_bytes = base + 512 * bytes});
        } else if (rows < 32) {
            for (uint32_t face = 2; face < 4; ++face) {
                noc.async_write_zeros(
                    buffer, (32 - rows) * 16 * bytes, {.offset_bytes = base + (face * 256 + (rows - 16) * 16) * bytes});
            }
        }
    }
}

}  // namespace

void kernel_main() {
    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t values_addr = get_arg_val<uint32_t>(1);
    const uint32_t indices_addr = get_arg_val<uint32_t>(2);
    const uint32_t tile_row = get_arg_val<uint32_t>(3);
    const uint32_t segment = get_arg_val<uint32_t>(4);
    const uint32_t group = get_arg_val<uint32_t>(5);
    const uint32_t owner_x = get_arg_val<uint32_t>(6);
    const uint32_t owner_y = get_arg_val<uint32_t>(7);
    const uint32_t merger_x = get_arg_val<uint32_t>(8);
    const uint32_t merger_y = get_arg_val<uint32_t>(9);
    const bool merger = segment == 0;
    const bool owner = merger && group == 0;

    constexpr auto input_args = TensorAccessorArgs<0>();
    constexpr auto values_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto indices_args = TensorAccessorArgs<values_args.next_compile_time_args_offset()>();
    const auto input = TensorAccessor(input_args, input_addr);

    const uint32_t row_base = (tile_row % Ht) * 32;
    const uint32_t rows = row_base + 32 <= height ? 32 : height - row_base;
    const uint32_t first_tile = segment * segment_tiles;
    const uint32_t tiles = first_tile + segment_tiles <= Wt ? segment_tiles : Wt - first_tile;
    const uint64_t values_tiles = get_noc_addr(owner_x, owner_y, get_write_ptr(cb_values));
    const uint64_t indices_tiles = get_noc_addr(owner_x, owner_y, get_write_ptr(cb_indices));

    // The first core zeroes the padding rows of its output tiles; no other
    // core writes those.
    if (owner) {
        Noc noc;
        zero_padding_rows<value_bytes, value_tile_bytes>(noc, cb_values, rows);
        zero_padding_rows<index_bytes, index_tile_bytes>(noc, cb_indices, rows);
        noc.write_zeros_l1_barrier();
    }

    // Read this core's face rows: rows group, group + row_groups, ... of the
    // segment, the two halves of each tile in turn.
    const uint32_t in = get_write_ptr(cb_in);
    uint32_t slot = 0;
    for (uint32_t row = group; row < rows; row += row_groups, ++slot) {
        for (uint32_t t = 0; t < tiles; ++t) {
            const uint32_t page = tile_row * Wt + first_tile + t;
            for (uint32_t half = 0; half < 2; ++half) {
                const uint32_t offset =
                    (((row / 16) * 2 + half) * 256 + (row % 16) * 16) * value_bytes & ~(read_bytes - 1);
                noc_async_read(
                    input.get_noc_addr(page, offset), in + ((slot * tiles + t) * 2 + half) * read_bytes, read_bytes);
            }
        }
    }
    noc_async_read_barrier();

    // Select each row's best k of the segment. The list being built stays on
    // the stack, in the core's local memory: an insertion shifts it, which
    // costs several times more in L1.
    const uint32_t candidates = get_write_ptr(cb_candidates);
    const uint32_t stage = get_write_ptr(cb_rows);
    Candidate best[k];
    slot = 0;
    for (uint32_t row = group; row < rows; row += row_groups, ++slot) {
        uint32_t found = 0;
        uint32_t threshold = 0;
        for (uint32_t t = 0; t < tiles; ++t) {
            for (uint32_t half = 0; half < 2; ++half) {
                const uint32_t col = (first_tile + t) * 32 + half * 16;
                if (col >= width) {
                    break;
                }
                const uint32_t count = col + 16 <= width ? 16 : width - col;
                const uint32_t chunk = in + ((slot * tiles + t) * 2 + half) * read_bytes;
                for (uint32_t e = 0; e < count; ++e) {
                    uint32_t bits;
                    if constexpr (value_bytes == 4) {
                        bits = reinterpret_cast<const tt_l1_ptr uint32_t*>(chunk)[e];
                    } else {
                        bits = static_cast<uint32_t>(
                                   reinterpret_cast<const tt_l1_ptr uint16_t*>(chunk)[(row % 2) * 16 + e])
                               << 16;
                    }
                    const uint32_t key = key_of(bits);
                    if (found < k) {
                        insert(best, found++, key, col + e);
                        if (found == k) {
                            threshold = best[k - 1].key;
                        }
                    } else if (key > threshold) {
                        insert(best, k - 1, key, col + e);
                        threshold = best[k - 1].key;
                    }
                }
            }
        }
        for (; found < k; ++found) {
            best[found] = {0, no_index};
        }
        if constexpr (segments == 1) {
            write_row(best, row, stage + slot * row_bytes, values_tiles, indices_tiles);
        } else {
            Candidate* list = reinterpret_cast<Candidate*>(stage + slot * row_bytes);
            for (uint32_t i = 0; i < k; ++i) {
                list[i] = best[i];
            }
            const uint32_t dst = candidates + (slot * segments + segment) * k_stride * sizeof(Candidate);
            noc_async_write(
                reinterpret_cast<uint32_t>(list), get_noc_addr(merger_x, merger_y, dst), k_stride * sizeof(Candidate));
        }
    }
    noc_async_write_barrier();

    if constexpr (segments > 1) {
        if (!merger) {
            noc_semaphore_inc(get_noc_addr(merger_x, merger_y, get_semaphore(merge_sem_id)), 1);
            noc_async_atomic_barrier();
            return;
        }
        volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(merge_sem_id));
        noc_semaphore_wait(sem, segments - 1);
        noc_semaphore_set(sem, 0);

        // Merge each row's segments: a segment's list is best-first, so its
        // first candidate that does not beat the k-th ends it.
        const Candidate* all = reinterpret_cast<const Candidate*>(candidates);
        slot = 0;
        for (uint32_t row = group; row < rows; row += row_groups, ++slot) {
            const Candidate* lists = all + slot * segments * k_stride;
            for (uint32_t i = 0; i < k; ++i) {
                best[i] = lists[i];
            }
            for (uint32_t s = 1; s < segments; ++s) {
                const Candidate* list = lists + s * k_stride;
                for (uint32_t i = 0; i < k && better(list[i].key, list[i].index, best[k - 1]); ++i) {
                    insert(best, k - 1, list[i].key, list[i].index);
                }
            }
            write_row(best, row, stage + slot * row_bytes, values_tiles, indices_tiles);
        }
        noc_async_write_barrier();
    }

    if (!owner) {
        noc_semaphore_inc(get_noc_addr(owner_x, owner_y, get_semaphore(owner_sem_id)), 1);
        noc_async_atomic_barrier();
        return;
    }
    if constexpr (row_groups > 1) {
        volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(owner_sem_id));
        noc_semaphore_wait(sem, row_groups - 1);
        noc_semaphore_set(sem, 0);
    }
    const auto values = TensorAccessor(values_args, values_addr);
    const auto indices = TensorAccessor(indices_args, indices_addr);
    for (uint32_t tile = 0; tile < k_tiles; ++tile) {
        noc_async_write_page(tile_row * k_tiles + tile, values, get_write_ptr(cb_values) + tile * value_tile_bytes);
        noc_async_write_page(tile_row * k_tiles + tile, indices, get_write_ptr(cb_indices) + tile * index_tile_bytes);
    }
    noc_async_write_barrier();
}
