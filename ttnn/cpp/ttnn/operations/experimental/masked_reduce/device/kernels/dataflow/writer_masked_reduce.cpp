// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
//
// Writes this core's accumulator tile to its slot of the partials tensor.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t partial_tile = get_arg(args::partial_tile);
    DataflowBuffer out_dfb(dfb::out);
    const auto partials = TensorAccessor(tensor::partials);
    Noc noc;
    out_dfb.wait_front(1);
    noc.async_write(out_dfb, partials, out_dfb.get_tile_size(), {.offset_bytes = 0}, {.page_id = partial_tile});
    noc.async_write_barrier();
    out_dfb.pop_front(1);
}
