// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
//
// rows_gather reader: for this core's positions, decodes their coordinates and
// reads the addressed table rows, chunk by chunk, into the rows DFB; the writer
// stores them in the output. Positions are numbered item = b * M + m over the
// index tensor's batch and position dimensions; item i produces output rows
// i * W + j for its W window offsets. Chunks are pushed in groups of up to
// `group` entries so several reads are in flight at once.

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
    constexpr bool index_unsigned = get_arg(args::index_unsigned) != 0;
    constexpr bool fill_oob = get_arg(args::fill_oob) != 0;
    constexpr bool index_shared = get_arg(args::index_shared) != 0;
    constexpr uint32_t W = get_arg(args::window);
    constexpr uint32_t M = get_arg(args::positions);
    constexpr uint32_t N = get_arg(args::rows_per_batch);
    constexpr uint32_t row_bytes = get_arg(args::row_bytes);
    constexpr uint32_t chunk_bytes = get_arg(args::chunk_bytes);
    constexpr uint32_t index_bytes = get_arg(args::index_bytes);
    constexpr uint32_t group = get_arg(args::group);
    constexpr uint32_t index_batch = get_arg(args::index_batch);
    constexpr uint32_t chunks_per_row = (row_bytes + chunk_bytes - 1) / chunk_bytes;
    constexpr std::array<uint32_t, K> sizes{ROWS_SIZES};
    constexpr std::array<uint32_t, K> windows{ROWS_WINDOWS};
    constexpr std::array<uint32_t, K> strides{ROWS_STRIDES};

    const uint32_t first_item = get_arg(args::first_item);
    const uint32_t num_items = get_arg(args::num_items);

    DataflowBuffer rows(dfb::rows);
    DataflowBuffer index(dfb::index);
    const auto table = TensorAccessor(tensor::table);
    const auto indices = TensorAccessor(tensor::indices);
    Noc noc;

    const uint32_t entry = rows.get_entry_size();
    const uint32_t index_entry = index.get_entry_size();
    const uint32_t total_chunks = num_items * W * chunks_per_row;

    // Index pages loaded so far: items [first_item, loaded_end), the current
    // batch of which sits at the front of the index DFB.
    uint32_t loaded_end = first_item;
    uint32_t batch_begin = first_item;
    uint32_t batch_count = 0;
    Coords<K> coords{};
    uint32_t coords_item = ~0u;

    for (uint32_t q = 0; q < total_chunks;) {
        const uint32_t count = total_chunks - q < group ? total_chunks - q : group;
        rows.reserve_back(count);
        uint32_t write_addr = rows.get_write_ptr();
        for (uint32_t i = 0; i < count; ++i, ++q, write_addr += entry) {
            const uint32_t item = first_item + q / (W * chunks_per_row);
            const uint32_t j = (q / chunks_per_row) % W;
            const uint32_t chunk = q % chunks_per_row;
            if (item != coords_item) {
                if (item >= loaded_end) {
                    // Load the next batch of index pages. A shared index tensor
                    // restarts its pages at every batch of the table.
                    if (batch_count != 0) {
                        index.pop_front(batch_count);
                    }
                    const uint32_t m = item % M;
                    uint32_t n = first_item + num_items - item;
                    if (n > index_batch) {
                        n = index_batch;
                    }
                    if (index_shared && n > M - m) {
                        n = M - m;
                    }
                    read_index_pages(noc, index, indices, index_shared ? m : item, n, index_bytes);
                    index.wait_front(n);
                    batch_begin = item;
                    batch_count = n;
                    loaded_end = item + n;
                }
                const uint32_t addr = index.get_read_ptr() + (item - batch_begin) * index_entry;
                coords = decode<K, index_unsigned>(
                    reinterpret_cast<volatile tt_l1_ptr const uint32_t*>(addr), sizes, windows);
                coords_item = item;
            }
            const uint32_t offset = chunk * chunk_bytes;
            const uint32_t bytes = row_bytes - offset < chunk_bytes ? row_bytes - offset : chunk_bytes;
            if (fill_oob && coords.out_of_bounds) {
                noc.async_write_zeros(rows, bytes, {.offset_bytes = i * entry});
                noc.write_zeros_l1_barrier();
                continue;
            }
            const uint32_t b = item / M;
            const uint32_t page = b * N + window_row<K>(coords, j, windows, strides);
            noc.async_read(table, rows, bytes, {.page_id = page, .offset_bytes = offset}, {.offset_bytes = i * entry});
        }
        noc.async_read_barrier();
        rows.push_back(count);
    }
    if (batch_count != 0) {
        index.pop_front(batch_count);
    }
}
