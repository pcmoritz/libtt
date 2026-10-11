// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reduce.h"

// The softmax of each row's top k values (see router_topk.cpp), which come
// sorted, largest first, and padded with -1e30:
//   e = exp(v - v[:, 0]), w = e / sum(e)
void kernel_main() {
    constexpr uint32_t cb_values = get_named_compile_time_arg_val("cb_values");
    constexpr uint32_t cb_scaler = get_named_compile_time_arg_val("cb_scaler");
    constexpr uint32_t cb_exp = get_named_compile_time_arg_val("cb_exp");
    constexpr uint32_t cb_sum = get_named_compile_time_arg_val("cb_sum");
    constexpr uint32_t cb_weights = get_named_compile_time_arg_val("cb_weights");

    auto pack = [](uint32_t cb) {
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb);
        pack_tile(0, cb);
        tile_regs_release();
    };

    compute_kernel_hw_startup(cb_values, cb_values, cb_exp);
    cb_wait_front(cb_values, 1);
    cb_reserve_back(cb_exp, 1);
    sub_bcast_cols_init(cb_values, cb_values);
    tile_regs_acquire();
    sub_tiles_bcast_cols(cb_values, cb_values, 0, 0, 0);
    exp_tile_init();
    exp_tile(0);
    pack(cb_exp);
    cb_push_back(cb_exp, 1);
    cb_pop_front(cb_values, 1);

    cb_wait_front(cb_exp, 1);
    cb_wait_front(cb_scaler, 1);
    cb_reserve_back(cb_sum, 1);
    // A reduce unpacks the scaler first; set the formats in that order.
    reconfig_data_format(cb_scaler, cb_exp);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_exp, cb_scaler, cb_sum);
    tile_regs_acquire();
    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_exp, cb_scaler, 0, 0, 0);
    reduce_uninit(cb_exp);
    recip_tile_init();
    recip_tile(0);
    pack(cb_sum);
    cb_push_back(cb_sum, 1);
    cb_pop_front(cb_scaler, 1);

    cb_wait_front(cb_sum, 1);
    cb_reserve_back(cb_weights, 1);
    reconfig_data_format(cb_exp, cb_sum);
    mul_bcast_cols_init(cb_exp, cb_sum);
    tile_regs_acquire();
    mul_tiles_bcast_cols(cb_exp, cb_sum, 0, 0, 0);
    pack(cb_weights);
    cb_push_back(cb_weights, 1);
    cb_pop_front(cb_exp, 1);
    cb_pop_front(cb_sum, 1);
}
