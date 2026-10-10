// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

// FP32 matmul on the vector unit. Takes the same input stream as matmul's
// bmm_metal2.cpp: for each output tile, the Kt tile pairs A(m, k), B(k, n).

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul_fp32.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t Kt = get_arg(args::Kt);
    const uint32_t num_tiles = get_arg(args::num_tiles);

    DataflowBuffer in0_dfb(dfb::in0);
    DataflowBuffer in1_dfb(dfb::in1);
    DataflowBuffer out_dfb(dfb::out);

    // DST tiles: the accumulator in 0, A (transposed in place) and B in 1 and
    // 2. The finished C tile goes to 1, with 2 as scratch.
    constexpr uint32_t dst_acc = 0;
    constexpr uint32_t dst_a = 1;
    constexpr uint32_t dst_b = 2;

    compute_kernel_hw_startup(dfb::in0, dfb::out);
    for (uint32_t tile = 0; tile < num_tiles; ++tile) {
        tile_regs_acquire();
        for (uint32_t kt = 0; kt < Kt; ++kt) {
            in0_dfb.wait_front(1);
            in1_dfb.wait_front(1);
            copy_init(dfb::in0);
            copy_tile(dfb::in0, 0, dst_a);
            copy_init(dfb::in1);
            copy_tile(dfb::in1, 0, dst_b);
            matmul_fp32_transpose_a(dst_a);
            matmul_fp32_tile_init();
            if (kt == 0) {
                matmul_fp32_tile<false, dst_a, dst_b, dst_acc>();
            } else {
                matmul_fp32_tile<true, dst_a, dst_b, dst_acc>();
            }
            in0_dfb.pop_front(1);
            in1_dfb.pop_front(1);
        }
        matmul_fp32_tile_finish(dst_acc, dst_a, dst_b);
        tile_regs_commit();

        out_dfb.reserve_back(1);
        tile_regs_wait();
        pack_tile(dst_a, dfb::out);
        tile_regs_release();
        out_dfb.push_back(1);
    }
}
