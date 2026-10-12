// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
//
// rows_gather writer: stores the rows the reader gathered. Output row
// item * W + j takes window offset j of position item; the chunk sequence and
// its grouping match the reader's.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t W = get_arg(args::window);
    constexpr uint32_t row_bytes = get_arg(args::row_bytes);
    constexpr uint32_t chunk_bytes = get_arg(args::chunk_bytes);
    constexpr uint32_t group = get_arg(args::group);
    constexpr uint32_t chunks_per_row = (row_bytes + chunk_bytes - 1) / chunk_bytes;

    const uint32_t first_item = get_arg(args::first_item);
    const uint32_t num_items = get_arg(args::num_items);

    DataflowBuffer rows(dfb::rows);
    const auto output = TensorAccessor(tensor::output);
    Noc noc;

    const uint32_t entry = rows.get_entry_size();
    const uint32_t total_chunks = num_items * W * chunks_per_row;
    for (uint32_t q = 0; q < total_chunks;) {
        const uint32_t count = total_chunks - q < group ? total_chunks - q : group;
        rows.wait_front(count);
        for (uint32_t i = 0; i < count; ++i, ++q) {
            const uint32_t out_row = first_item * W + q / chunks_per_row;
            const uint32_t chunk = q % chunks_per_row;
            const uint32_t offset = chunk * chunk_bytes;
            const uint32_t bytes = row_bytes - offset < chunk_bytes ? row_bytes - offset : chunk_bytes;
            noc.async_write(rows, output, bytes, {.offset_bytes = i * entry}, {.page_id = out_row, .offset_bytes = offset});
        }
        noc.async_writes_flushed();
        rows.pop_front(count);
    }
    noc.async_write_barrier();
}
