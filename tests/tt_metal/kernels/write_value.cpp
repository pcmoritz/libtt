// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

// Fills page 0 of a DRAM buffer (64 bytes) with a value, staged through L1.
void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t value = get_arg_val<uint32_t>(1);
    const uint32_t scratch = get_arg_val<uint32_t>(2);
    volatile tt_l1_ptr uint32_t* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    for (uint32_t i = 0; i < 16; i++) {
        words[i] = value;
    }
    const InterleavedAddrGen<true> dst{.bank_base_address = dst_addr, .page_size = 64};
    noc_async_write(scratch, dst.get_noc_addr(0), 64);
    noc_async_write_barrier();
}
