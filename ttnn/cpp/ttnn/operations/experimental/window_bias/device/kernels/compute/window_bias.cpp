// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
//
// Per token: bias tile = A @ B^T (A: heads x 16 BF16, B: 32 key columns x 16
// FP32, the matmul's in1 transposed as for scores against keys), packed BF16;
// on the sink core, the sink tile times sink_scale.
#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t cb_a = get_named_compile_time_arg_val("cb_a");
    constexpr uint32_t cb_b = get_named_compile_time_arg_val("cb_b");
    constexpr uint32_t cb_sink = get_named_compile_time_arg_val("cb_sink");
    constexpr uint32_t cb_bias = get_named_compile_time_arg_val("cb_bias");
    constexpr uint32_t cb_sink_out = get_named_compile_time_arg_val("cb_sink_out");
    constexpr uint32_t tokens = get_named_compile_time_arg_val("tokens");
    constexpr bool has_sink = get_named_compile_time_arg_val("has_sink") != 0;
    constexpr uint32_t sink_scale = get_named_compile_time_arg_val("sink_scale");
    const bool sinks = get_arg_val<uint32_t>(0) != 0;

    compute_kernel_hw_startup(cb_a, cb_b, cb_bias);
    for (uint32_t t = 0; t < tokens; ++t) {
        cb_wait_front(cb_a, 1);
        cb_wait_front(cb_b, 1);
        tile_regs_acquire();
        // in0 (A) unpacks to SrcB and in1 (B) to SrcA.
        reconfig_data_format(cb_b, cb_a);
        matmul_init(cb_a, cb_b, /*transpose=*/1);
        matmul_tiles(cb_a, cb_b, 0, 0, 0);
        tile_regs_commit();
        cb_pop_front(cb_a, 1);
        cb_pop_front(cb_b, 1);
        cb_reserve_back(cb_bias, 1);
        tile_regs_wait();
        pack_reconfig_data_format(cb_bias);
        pack_tile(0, cb_bias);
        tile_regs_release();
        cb_push_back(cb_bias, 1);

        if (has_sink && sinks) {
            cb_wait_front(cb_sink, 1);
            tile_regs_acquire();
            reconfig_data_format_srca(cb_sink);
            copy_init(cb_sink);
            copy_tile(cb_sink, 0, 0);
            binop_with_scalar_tile_init();
            mul_unary_tile(0, sink_scale);
            tile_regs_commit();
            cb_pop_front(cb_sink, 1);
            cb_reserve_back(cb_sink_out, 1);
            tile_regs_wait();
            pack_reconfig_data_format(cb_sink_out);
            pack_tile(0, cb_sink_out);
            tile_regs_release();
            cb_push_back(cb_sink_out, 1);
        }
    }
}
