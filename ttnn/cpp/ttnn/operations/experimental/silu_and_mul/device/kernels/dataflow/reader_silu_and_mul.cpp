// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// For each of this core's output tiles (row-major units of Nt per tile row):
// its gate tile and its up tile, Nt tiles later in the product's row.
void kernel_main() {
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);
    const uint32_t product_addr = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_gate = get_named_compile_time_arg_val("cb_gate");
    constexpr uint32_t cb_up = get_named_compile_time_arg_val("cb_up");
    constexpr uint32_t Nt = get_named_compile_time_arg_val("Nt");
    constexpr auto product_args = TensorAccessorArgs<0>();
    const uint32_t tile = get_tile_size(cb_gate);
    const auto product = TensorAccessor(product_args, product_addr, tile);

    for (uint32_t unit = start; unit < start + count; ++unit) {
        const uint32_t gate = unit / Nt * 2 * Nt + unit % Nt;
        cb_reserve_back(cb_gate, 1);
        cb_reserve_back(cb_up, 1);
        noc_async_read(product.get_noc_addr(gate), get_write_ptr(cb_gate), tile);
        noc_async_read(product.get_noc_addr(gate + Nt), get_write_ptr(cb_up), tile);
        noc_async_read_barrier();
        cb_push_back(cb_gate, 1);
        cb_push_back(cb_up, 1);
    }
}
