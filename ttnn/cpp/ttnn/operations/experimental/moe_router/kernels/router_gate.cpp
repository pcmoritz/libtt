// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

// The gate of moe_router_topk with a weight (see router_gate_writer.cpp). Each
// core multiplies its K slice of the input by the weight's for one column of
// 32 experts, accumulating in FP32 DST. The first core of each column then
// sums the partials: its writer gathers each row's partials as the rows of one
// FP32 tile, which is copied to DST as is and summed over its rows by the
// SFPU, so the sum is FP32 throughout.
void kernel_main() {
    const bool owner = get_arg_val<uint32_t>(0) != 0;

    constexpr uint32_t gate_kt = get_named_compile_time_arg_val("gate_kt");
    constexpr uint32_t rows = get_named_compile_time_arg_val("rows");
    constexpr uint32_t cb_x = get_named_compile_time_arg_val("cb_x");
    constexpr uint32_t cb_w = get_named_compile_time_arg_val("cb_w");
    constexpr uint32_t cb_partial = get_named_compile_time_arg_val("cb_partial");
    constexpr uint32_t cb_gather = get_named_compile_time_arg_val("cb_gather");
    constexpr uint32_t cb_sums = get_named_compile_time_arg_val("cb_sums");

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_partial);
    matmul_init(cb_x, cb_w);
    cb_wait_front(cb_x, gate_kt);
    cb_wait_front(cb_w, gate_kt);
    tile_regs_acquire();
    for (uint32_t k = 0; k < gate_kt; ++k) {
        matmul_tiles(cb_x, cb_w, k, k, 0);
    }
    tile_regs_commit();
    cb_pop_front(cb_x, gate_kt);
    cb_pop_front(cb_w, gate_kt);
    cb_reserve_back(cb_partial, 1);
    tile_regs_wait();
    pack_tile(0, cb_partial);
    tile_regs_release();
    cb_push_back(cb_partial, 1);
    if (!owner) {
        return;
    }

    reconfig_data_format_srca(cb_gather);
    copy_init(cb_gather);
    sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
    for (uint32_t row = 0; row < rows; ++row) {
        cb_wait_front(cb_gather, 1);
        tile_regs_acquire();
        copy_tile(cb_gather, 0, 0);
        // The column sums land in row 0.
        sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_COL>(0);
        tile_regs_commit();
        cb_pop_front(cb_gather, 1);
        cb_reserve_back(cb_sums, 1);
        tile_regs_wait();
        pack_tile(0, cb_sums);
        tile_regs_release();
        cb_push_back(cb_sums, 1);
    }
}
