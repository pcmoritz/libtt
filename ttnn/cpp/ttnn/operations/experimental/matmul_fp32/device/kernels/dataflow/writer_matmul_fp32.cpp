// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

// Writes this core's output tiles, a contiguous range in tile order.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t first_tile = get_arg(args::first_tile);
    const uint32_t num_tiles = get_arg(args::num_tiles);

    DataflowBuffer out_dfb(dfb::out);
    const uint32_t tile_bytes = out_dfb.get_entry_size();
    const auto c = TensorAccessor(tensor::output);
    Noc noc;

    for (uint32_t tile = first_tile; tile < first_tile + num_tiles; ++tile) {
        out_dfb.wait_front(1);
        noc.async_write(out_dfb, c, tile_bytes, {.offset_bytes = 0}, {.page_id = tile});
        noc.async_writes_flushed();
        out_dfb.pop_front(1);
    }
    noc.async_write_barrier();
}
