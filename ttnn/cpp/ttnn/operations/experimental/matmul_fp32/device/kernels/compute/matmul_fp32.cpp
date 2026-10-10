// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

// FP32 matmul on the vector unit, in runs of output tiles that share their A
// row (matmul_fp32_runs.hpp): in0 holds the run's Kt A tiles (or, when the
// row does not fit in L1, streams them one per k), in1 streams the B tiles,
// one per k.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul_fp32.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/experimental/matmul_fp32/device/kernels/matmul_fp32_runs.hpp"

void kernel_main() {
    const uint32_t Kt = get_arg(args::Kt);
    const uint32_t Nt = get_arg(args::Nt);
    const uint32_t run_width = get_arg(args::run_width);
    const uint32_t a_resident = get_arg(args::a_resident);
    const uint32_t first_tile = get_arg(args::first_tile);
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
    copy_init(dfb::in0);  // in1 has the same format, and the transposes leave the copy state alone
    matmul_fp32_tile_init();
    const uint32_t end = first_tile + num_tiles;
    for (uint32_t tile = first_tile; tile < end;) {
        const uint32_t cols = matmul_fp32_run_cols(tile, end, Nt, run_width);
        if (a_resident) {
            in0_dfb.wait_front(Kt);
        }
        for (uint32_t j = 0; j < cols; ++j) {
            tile_regs_acquire();
            for (uint32_t k = 0; k < Kt; ++k) {
                if (!a_resident) {
                    in0_dfb.wait_front(1);
                }
                in1_dfb.wait_front(1);
                copy_tile(dfb::in0, a_resident ? k : 0, dst_a);
                copy_tile(dfb::in1, 0, dst_b);
                matmul_fp32_transpose_a(dst_a);
                if (k == 0) {
                    matmul_fp32_tile<false>(dst_a, dst_b, dst_acc);
                } else {
                    matmul_fp32_tile<true>(dst_a, dst_b, dst_acc);
                }
                in1_dfb.pop_front(1);
                if (!a_resident) {
                    in0_dfb.pop_front(1);
                }
            }
            matmul_fp32_tile_finish(dst_acc, dst_a, dst_b);
            tile_regs_commit();

            out_dfb.reserve_back(1);
            tile_regs_wait();
            pack_tile(dst_a, dfb::out);
            tile_regs_release();
            out_dfb.push_back(1);
        }
        if (a_resident) {
            in0_dfb.pop_front(Kt);
        }
        tile += cols;
    }
}
