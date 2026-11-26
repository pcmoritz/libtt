// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// Writes this core's output tiles (row-major units).
void kernel_main() {
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);
    const uint32_t output_addr = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_output = get_named_compile_time_arg_val("cb_output");
    constexpr auto output_args = TensorAccessorArgs<0>();
    const uint32_t tile = get_tile_size(cb_output);
    const auto output = TensorAccessor(output_args, output_addr, tile);

    for (uint32_t unit = start; unit < start + count; ++unit) {
        cb_wait_front(cb_output, 1);
        noc_async_write(get_read_ptr(cb_output), output.get_noc_addr(unit), tile);
        noc_async_write_barrier();
        cb_pop_front(cb_output, 1);
    }
}
