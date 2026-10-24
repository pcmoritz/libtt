// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
//
// Writes the tiles the reader assembled to this core's run of output pages.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t first_page = get_arg(args::first_page);
    const uint32_t num_pages = get_arg(args::num_pages);
    constexpr uint32_t batch_tiles = get_arg(args::batch_tiles);
    constexpr uint32_t row_bytes = get_arg(args::row_bytes);
    constexpr uint32_t tile_bytes = 64 * row_bytes;

    DataflowBuffer out_dfb(dfb::out);
    const auto output = TensorAccessor(tensor::output);
    Noc noc;

    for (uint32_t done = 0; done < num_pages;) {
        const uint32_t count = num_pages - done < batch_tiles ? num_pages - done : batch_tiles;
        out_dfb.wait_front(count);
        for (uint32_t i = 0; i < count; ++i) {
            noc.async_write(
                out_dfb, output, tile_bytes, {.offset_bytes = i * tile_bytes}, {.page_id = first_page + done + i});
        }
        // The slots may be refilled once their data has left L1.
        noc.async_writes_flushed();
        out_dfb.pop_front(count);
        done += count;
    }
    noc.async_write_barrier();
}
