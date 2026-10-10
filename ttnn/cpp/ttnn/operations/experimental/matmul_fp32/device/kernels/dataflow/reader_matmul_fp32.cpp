// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

// For each run (matmul_fp32_runs.hpp): the run's A row (Kt tiles) into in0
// once, then for each of its output tiles the B column, one tile per k, into
// in1. If the A row does not fit in L1 (a_resident == 0, runs of one tile), A
// streams one tile per k alongside B.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/experimental/matmul_fp32/device/kernels/matmul_fp32_runs.hpp"
#include "ttnn/operations/kernel_helper_functions/pad_tile.hpp"

void kernel_main() {
    const uint32_t Mt = get_arg(args::Mt);
    const uint32_t Kt = get_arg(args::Kt);
    const uint32_t Nt = get_arg(args::Nt);
    const uint32_t bcast_b = get_arg(args::bcast_b);
    const uint32_t run_width = get_arg(args::run_width);
    const uint32_t a_resident = get_arg(args::a_resident);
    const uint32_t first_tile = get_arg(args::first_tile);
    const uint32_t num_tiles = get_arg(args::num_tiles);
    // Columns of A's and rows of B's last k tile that lie past K. Both are
    // zeroed, since either could hold an infinity or NaN left by an earlier
    // operation, and 0 * inf is NaN.
    constexpr uint32_t last_ktile_k = get_arg(args::last_ktile_k);

    DataflowBuffer in0_dfb(dfb::in0);
    DataflowBuffer in1_dfb(dfb::in1);
    const uint32_t in0_tile_bytes = in0_dfb.get_tile_size();
    const uint32_t in1_tile_bytes = in1_dfb.get_tile_size();
    const auto a = TensorAccessor(tensor::in0);
    const auto b = TensorAccessor(tensor::in1);
    Noc noc;

    const uint32_t end = first_tile + num_tiles;
    for (uint32_t tile = first_tile; tile < end;) {
        const uint32_t cols = matmul_fp32_run_cols(tile, end, Nt, run_width);
        const uint32_t batch = tile / (Mt * Nt);
        const uint32_t a_row = (batch * Mt + tile % (Mt * Nt) / Nt) * Kt;
        const uint32_t b_base = (bcast_b ? 0 : batch * Kt * Nt) + tile % Nt;

        auto read_a = [&](uint32_t k, uint32_t offset_bytes) {
            noc.async_read(a, in0_dfb, in0_tile_bytes, {.page_id = a_row + k}, {.offset_bytes = offset_bytes});
        };
        auto pad_a = [&](uint32_t write_addr) {
            if constexpr (last_ktile_k > 0) {
                pad_last_ktile<get_dataformat(dfb::in0), last_ktile_k>(write_addr);
            }
        };
        if (a_resident) {
            in0_dfb.reserve_back(Kt);
            const uint32_t in0_write = in0_dfb.get_write_ptr();
            for (uint32_t k = 0; k < Kt; ++k) {
                read_a(k, k * in0_tile_bytes);
            }
            noc.async_read_barrier();
            pad_a(in0_write + (Kt - 1) * in0_tile_bytes);
            in0_dfb.push_back(Kt);
        }
        for (uint32_t j = 0; j < cols; ++j) {
            for (uint32_t k = 0; k < Kt; ++k) {
                if (!a_resident) {
                    in0_dfb.reserve_back(1);
                    read_a(k, 0);
                }
                in1_dfb.reserve_back(1);
                noc.async_read(b, in1_dfb, in1_tile_bytes, {.page_id = b_base + k * Nt + j}, {.offset_bytes = 0});
                noc.async_read_barrier();
                if (k == Kt - 1) {
                    if constexpr (last_ktile_k > 0) {
                        pad_last_transposed_ktile<get_dataformat(dfb::in1), last_ktile_k>(in1_dfb.get_write_ptr());
                    }
                    if (!a_resident) {
                        pad_a(in0_dfb.get_write_ptr());
                    }
                }
                if (!a_resident) {
                    in0_dfb.push_back(1);
                }
                in1_dfb.push_back(1);
            }
        }
        tile += cols;
    }
}
