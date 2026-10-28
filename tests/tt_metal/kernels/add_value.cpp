// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

// Adds a value to every word of page 0 of a DRAM buffer (64 bytes), through L1.
void kernel_main() {
    const uint32_t addr = get_arg_val<uint32_t>(0);
    const uint32_t value = get_arg_val<uint32_t>(1);
    const uint32_t scratch = get_arg_val<uint32_t>(2);
    const InterleavedAddrGen<true> page{.bank_base_address = addr, .page_size = 64};
    noc_async_read(page.get_noc_addr(0), scratch, 64);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    for (uint32_t i = 0; i < 16; i++) {
        words[i] = words[i] + value;
    }
    noc_async_write(scratch, page.get_noc_addr(0), 64);
    noc_async_write_barrier();
}
