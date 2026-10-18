// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
//
// rows_gather reader: for this core's transfers, decodes their positions'
// coordinates and reads the addressed table pages, chunk by chunk, into the rows
// DFB; the writer stores them in the output. Positions are numbered
// item = b * M + m over the index tensor's batch and position dimensions. Each
// item makes W * S * chunks_per_page transfers (window offset j, page s of the
// row, chunk c), or W transfers of one column window when the last component
// addresses columns; transfer q of the program is transfer q % that of item
// q / that, and a core's range may start or end inside an item. Chunks are
// pushed in groups of up to `group` entries so several reads are in flight at
// once.

#include <array>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/experimental/indexed_rows/device/kernels/indexed_rows_common.hpp"

void kernel_main() {
    using namespace indexed_rows;
    constexpr uint32_t K = get_arg(args::num_components);
    constexpr bool column = get_arg(args::column) != 0;
    constexpr uint32_t KR = column ? K - 1 : K;
    constexpr bool index_unsigned = get_arg(args::index_unsigned) != 0;
    constexpr bool fill_oob = get_arg(args::fill_oob) != 0;
    constexpr bool index_shared = get_arg(args::index_shared) != 0;
    constexpr uint32_t W = get_arg(args::window);
    constexpr uint32_t M = get_arg(args::positions);
    constexpr uint32_t N = get_arg(args::rows_per_batch);
    constexpr uint32_t S = get_arg(args::pages_per_row);
    constexpr uint32_t page_bytes = get_arg(args::page_bytes);
    constexpr uint32_t chunk_bytes = get_arg(args::chunk_bytes);
    constexpr uint32_t elem_bytes = get_arg(args::elem_bytes);
    constexpr uint32_t index_stride = get_arg(args::index_stride);  // words between components
    constexpr uint32_t group = get_arg(args::group);
    constexpr uint32_t index_batch = get_arg(args::index_batch);
    constexpr uint32_t C = column ? 1 : (page_bytes + chunk_bytes - 1) / chunk_bytes;
    constexpr uint32_t transfers_per_item = column ? W : W * S * C;
    constexpr std::array<uint32_t, K> sizes{ROWS_SIZES};
    constexpr std::array<uint32_t, K> windows{ROWS_WINDOWS};
    constexpr std::array<uint32_t, K> strides{ROWS_STRIDES};
    constexpr uint32_t column_bytes = column ? windows[K - 1] * elem_bytes : 0;

    const uint32_t first_transfer = get_arg(args::first_transfer);
    const uint32_t num_transfers = get_arg(args::num_transfers);
    const uint32_t first_item = first_transfer / transfers_per_item;
    const uint32_t end_item = num_transfers == 0 ? first_item : (first_transfer + num_transfers - 1) / transfers_per_item + 1;

    DataflowBuffer rows(dfb::rows);
    DataflowBuffer index(dfb::index);
    const auto table = TensorAccessor(tensor::table);
    const auto indices = TensorAccessor(tensor::indices);
    Noc noc;

    const uint32_t entry = rows.get_entry_size();
    const uint32_t total = num_transfers;

    // The index chunk loaded so far: items [chunk_begin, loaded_end), at
    // `chunk_addr` in L1.
    uint32_t loaded_end = first_item;
    uint32_t chunk_begin = first_item;
    uint32_t chunk_addr = 0;
    bool chunk_loaded = false;
    Coords<K> coords{};
    uint32_t coords_item = ~0u;

    // Column windows: where each entry's data lands and where the writer wants it.
    std::array<uint32_t, group> from{};
    std::array<uint32_t, group> to{};

    for (uint32_t q = 0; q < total;) {
        const uint32_t count = total - q < group ? total - q : group;
        rows.reserve_back(count);
        const uint32_t write_ptr = rows.get_write_ptr();
        for (uint32_t i = 0; i < count; ++i, ++q) {
            const uint32_t item = (first_transfer + q) / transfers_per_item;
            const uint32_t t = (first_transfer + q) % transfers_per_item;
            if (item != coords_item) {
                if (item >= loaded_end) {
                    // Stream the next chunk of positions of this item's batch
                    // (a shared index tensor has one batch).
                    if (chunk_loaded) {
                        index.pop_front(1);
                    }
                    const uint32_t m = item % M;
                    uint32_t n = end_item - item;
                    if (n > index_batch) {
                        n = index_batch;
                    }
                    if (n > M - m) {
                        n = M - m;
                    }
                    chunk_addr = stream_index(
                        noc, index, indices, K, index_stride * sizeof(uint32_t), index_shared ? 0 : item / M, m, n);
                    chunk_loaded = true;
                    chunk_begin = item;
                    loaded_end = item + n;
                }
                coords = decode<K, index_unsigned>(
                    reinterpret_cast<const uint32_t*>(chunk_addr) + (item - chunk_begin), index_stride, sizes, windows);
                coords_item = item;
            }
            const uint32_t b = item / M;
            uint32_t bytes, page, offset, l1_offset = 0;
            if constexpr (column) {
                page = b * N + window_row<KR, K>(coords, t, windows, strides);
                offset = static_cast<uint32_t>(coords.c[K - 1]) * elem_bytes;
                bytes = column_bytes;
                from[i] = offset % kGranule;
                to[i] = (((item % M) * W + t) * column_bytes) % kGranule;
                l1_offset = from[i];
            } else {
                const uint32_t j = t / (S * C);
                const uint32_t s = (t / C) % S;
                const uint32_t c = t % C;
                page = (b * N + window_row<KR, K>(coords, j, windows, strides)) * S + s;
                offset = c * chunk_bytes;
                bytes = page_bytes - offset < chunk_bytes ? page_bytes - offset : chunk_bytes;
            }
            if (fill_oob && coords.out_of_bounds) {
                if constexpr (column) {
                    from[i] = to[i];  // zeros need no realignment
                    l1_offset = to[i];
                }
                noc.async_write_zeros(rows, bytes, {.offset_bytes = i * entry + l1_offset});
                noc.write_zeros_l1_barrier();
                continue;
            }
            noc.async_read(
                table, rows, bytes, {.page_id = page, .offset_bytes = offset}, {.offset_bytes = i * entry + l1_offset});
        }
        noc.async_read_barrier();
        if constexpr (column) {
            for (uint32_t i = 0; i < count; ++i) {
                realign(write_ptr + i * entry, from[i], to[i], column_bytes);
            }
        }
        rows.push_back(count);
    }
    if (chunk_loaded) {
        index.pop_front(1);
    }
}
