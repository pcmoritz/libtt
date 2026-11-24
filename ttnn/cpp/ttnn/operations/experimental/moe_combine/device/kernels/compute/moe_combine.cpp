// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
//
// out tile = P @ slot tile per tile column, accumulated in FP32.
#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"

void kernel_main() {
    constexpr uint32_t cb_placement = get_named_compile_time_arg_val("cb_placement");
    constexpr uint32_t cb_slots = get_named_compile_time_arg_val("cb_slots");
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t num_cores = get_named_compile_time_arg_val("num_cores");
    constexpr uint32_t Ht = get_named_compile_time_arg_val("Ht");
    const uint32_t core = get_arg_val<uint32_t>(0);

    // in0 (the placement) unpacks to SrcB and in1 (the slots) to SrcA.
    compute_kernel_hw_startup(cb_slots, cb_placement, cb_out);
    reconfig_data_format(cb_slots, cb_placement);
    matmul_init(cb_placement, cb_slots);
    pack_reconfig_data_format(cb_out);
    cb_wait_front(cb_placement, 1);
    for (uint32_t c = core; c < Ht; c += num_cores) {
        cb_wait_front(cb_slots, 1);
        tile_regs_acquire();
        matmul_tiles(cb_placement, cb_slots, 0, 0, 0);
        tile_regs_commit();
        cb_pop_front(cb_slots, 1);
        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
    }
    cb_pop_front(cb_placement, 1);
}
