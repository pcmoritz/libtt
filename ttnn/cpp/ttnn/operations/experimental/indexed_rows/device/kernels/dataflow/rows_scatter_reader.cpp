// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
//
// rows_scatter reader: streams this core's table pages (pages
// [first_page, first_page + num_pages) of the [B, N, S, R] table) into the rows
// DFB, chunk by chunk, for the writer to copy into the output before it applies
// the updates that land in those pages.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t page_bytes = get_arg(args::page_bytes);
    constexpr uint32_t chunk_bytes = get_arg(args::chunk_bytes);
    constexpr uint32_t group = get_arg(args::group);
    constexpr uint32_t C = (page_bytes + chunk_bytes - 1) / chunk_bytes;

    const uint32_t first_page = get_arg(args::first_page);
    const uint32_t num_pages = get_arg(args::num_pages);

    DataflowBuffer rows(dfb::rows);
    const auto table = TensorAccessor(tensor::table);
    Noc noc;

    const uint32_t entry = rows.get_entry_size();
    const uint32_t total = num_pages * C;
    for (uint32_t q = 0; q < total;) {
        const uint32_t count = total - q < group ? total - q : group;
        rows.reserve_back(count);
        for (uint32_t i = 0; i < count; ++i, ++q) {
            const uint32_t page = first_page + q / C;
            const uint32_t offset = (q % C) * chunk_bytes;
            const uint32_t bytes = page_bytes - offset < chunk_bytes ? page_bytes - offset : chunk_bytes;
            noc.async_read(table, rows, bytes, {.page_id = page, .offset_bytes = offset}, {.offset_bytes = i * entry});
        }
        noc.async_read_barrier();
        rows.push_back(count);
    }
}
