// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

// Holds the core, and so every command queued after this program, until the
// host writes 1 to the flag in this core's L1.
void kernel_main() {
    volatile tt_l1_ptr uint32_t* flag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(0));
    while (*flag != 1) {
        invalidate_l1_cache();
    }
}
