// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
//
// rows_gather writer: stores the pages the reader gathered. Item i = b * M + m
// and window offset j write output row i * W + j (its S pages, chunk by chunk),
// or, with a column component, output page b at column (m * W + j) * w. The
// transfer range, sequence and grouping match the reader's.

#include <array>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t K = get_arg(args::num_components);
    constexpr bool column = get_arg(args::column) != 0;
    constexpr uint32_t W = get_arg(args::window);
    constexpr uint32_t M = get_arg(args::positions);
    constexpr uint32_t S = get_arg(args::pages_per_row);
    constexpr uint32_t page_bytes = get_arg(args::page_bytes);
    constexpr uint32_t chunk_bytes = get_arg(args::chunk_bytes);
    constexpr uint32_t elem_bytes = get_arg(args::elem_bytes);
    constexpr uint32_t group = get_arg(args::group);
    constexpr uint32_t C = column ? 1 : (page_bytes + chunk_bytes - 1) / chunk_bytes;
    constexpr uint32_t transfers_per_item = column ? W : W * S * C;
    constexpr std::array<uint32_t, K> windows{ROWS_WINDOWS};
    constexpr uint32_t column_bytes = column ? windows[K - 1] * elem_bytes : 0;

    const uint32_t first_transfer = get_arg(args::first_transfer);
    const uint32_t num_transfers = get_arg(args::num_transfers);

    DataflowBuffer rows(dfb::rows);
    const auto output = TensorAccessor(tensor::output);
    Noc noc;

    const uint32_t entry = rows.get_entry_size();
    const uint32_t total = num_transfers;
    for (uint32_t q = 0; q < total;) {
        const uint32_t count = total - q < group ? total - q : group;
        rows.wait_front(count);
        for (uint32_t i = 0; i < count; ++i, ++q) {
            const uint32_t item = (first_transfer + q) / transfers_per_item;
            const uint32_t t = (first_transfer + q) % transfers_per_item;
            uint32_t bytes, page, offset, l1_offset = 0;
            if constexpr (column) {
                page = item / M;
                offset = ((item % M) * W + t) * column_bytes;
                bytes = column_bytes;
                l1_offset = offset % 64;  // where the reader realigned it
            } else {
                const uint32_t j = t / (S * C);
                const uint32_t s = (t / C) % S;
                const uint32_t c = t % C;
                page = (item * W + j) * S + s;
                offset = c * chunk_bytes;
                bytes = page_bytes - offset < chunk_bytes ? page_bytes - offset : chunk_bytes;
            }
            noc.async_write(
                rows, output, bytes, {.offset_bytes = i * entry + l1_offset}, {.page_id = page, .offset_bytes = offset});
        }
        noc.async_writes_flushed();
        rows.pop_front(count);
    }
    noc.async_write_barrier();
}
