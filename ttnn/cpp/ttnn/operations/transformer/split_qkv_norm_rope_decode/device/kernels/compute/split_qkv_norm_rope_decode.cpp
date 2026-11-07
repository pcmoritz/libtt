// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/operations/normalization/kernel_util/compute/numeric.h"

namespace numeric = norm::kernel_util::compute::numeric;

// For each q or k unit, a tile row of heads (one per row): rms_norm, then
// rotary_embedding of the leading rotary_tiles, each step as those ops'
// kernels take it (layernorm.cpp with RMSNORM and gamma; rotary_embedding.cpp
// in decode mode), in the same buffer formats, so that every rounding is
// theirs:
//   x^2 (FP32), its row sum times 1 / width (FP32), + epsilon and rsqrt
//   (FP32), x * rstd (FP32), * weight (BF16 or FP32, as given);
//   y of the other half * sin (BF16), y * cos (BF16), their difference or
//   sum (BF16); the tiles past the rotated ones as rms_norm left them.
void kernel_main() {
    constexpr uint32_t cb_x = get_named_compile_time_arg_val("cb_x");
    constexpr uint32_t cb_weight = get_named_compile_time_arg_val("cb_weight");
    constexpr uint32_t cb_cos = get_named_compile_time_arg_val("cb_cos");
    constexpr uint32_t cb_sin = get_named_compile_time_arg_val("cb_sin");
    constexpr uint32_t cb_scaler = get_named_compile_time_arg_val("cb_scaler");
    constexpr uint32_t cb_eps = get_named_compile_time_arg_val("cb_eps");
    constexpr uint32_t cb_sq = get_named_compile_time_arg_val("cb_sq");
    constexpr uint32_t cb_mean = get_named_compile_time_arg_val("cb_mean");
    constexpr uint32_t cb_rstd = get_named_compile_time_arg_val("cb_rstd");
    constexpr uint32_t cb_scaled = get_named_compile_time_arg_val("cb_scaled");
    constexpr uint32_t cb_normed = get_named_compile_time_arg_val("cb_normed");
    constexpr uint32_t cb_sin_part = get_named_compile_time_arg_val("cb_sin_part");
    constexpr uint32_t cb_cos_part = get_named_compile_time_arg_val("cb_cos_part");
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t head_tiles = get_named_compile_time_arg_val("head_tiles");
    constexpr uint32_t rotary_tiles = get_named_compile_time_arg_val("rotary_tiles");
    constexpr uint32_t head_dim = get_named_compile_time_arg_val("head_dim");
    constexpr uint32_t half = rotary_tiles / 2;

    // Packs DST tile 0 as the next tile of `cb`.
    auto pack_one = [](uint32_t cb) {
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb, 1);
        pack_tile(0, cb);
        tile_regs_release();
        cb_push_back(cb, 1);
    };

    compute_kernel_hw_startup(cb_x, cb_x, cb_sq);
    DataflowBuffer dfb_sq(cb_sq);
    DataflowBuffer dfb_scaler(cb_scaler);
    DataflowBuffer dfb_mean(cb_mean);
    cb_wait_front(cb_eps, 1);
    // The q and k units; the v units only move data.
    const uint32_t units = get_arg_val<uint32_t>(0);
    for (uint32_t unit = 0; unit < units; ++unit) {
        // rms_norm: x^2
        cb_wait_front(cb_x, head_tiles);
        reconfig_data_format(cb_x, cb_x);
        pack_reconfig_data_format(cb_sq);
        mul_init(cb_x, cb_x);
        for (uint32_t j = 0; j < head_tiles; ++j) {
            tile_regs_acquire();
            mul_tiles(cb_x, cb_x, j, j, 0);
            pack_one(cb_sq);
        }

        // The mean of x^2, by rms_norm's own helper: a row reduce against the
        // scaler's ones, times 1 / width.
        numeric::row_wise_mean<PoolType::SUM, ReduceDim::REDUCE_ROW, true, policies::FullBlockWithPopPolicy>(
            dfb_sq, dfb_scaler, dfb_mean, head_dim, head_tiles, head_tiles);

        // rstd = rsqrt(mean + epsilon), down column 0.
        reconfig_data_format(cb_mean, cb_eps);
        pack_reconfig_data_format(cb_rstd);
        add_init(cb_mean, cb_eps);
        tile_regs_acquire();
        add_tiles(cb_mean, cb_eps, 0, 0, 0);
        rsqrt_tile_init();
        rsqrt_tile<ckernel::RsqrtMode::Default>(0);
        pack_one(cb_rstd);
        cb_pop_front(cb_mean, 1);

        // x * rstd (column broadcast), then * weight (row broadcast).
        cb_wait_front(cb_rstd, 1);
        reconfig_data_format(cb_x, cb_rstd);
        pack_reconfig_data_format(cb_scaled);
        mul_bcast_cols_init(cb_x, cb_rstd);
        for (uint32_t j = 0; j < head_tiles; ++j) {
            tile_regs_acquire();
            mul_tiles_bcast_cols(cb_x, cb_rstd, j, 0, 0);
            pack_one(cb_scaled);
        }
        cb_pop_front(cb_rstd, 1);
        cb_pop_front(cb_x, head_tiles);
        cb_wait_front(cb_scaled, head_tiles);
        cb_wait_front(cb_weight, head_tiles);
        reconfig_data_format(cb_scaled, cb_weight);
        pack_reconfig_data_format(cb_normed);
        mul_bcast_rows_init(cb_scaled, cb_weight);
        for (uint32_t j = 0; j < head_tiles; ++j) {
            tile_regs_acquire();
            mul_tiles_bcast_rows(cb_scaled, cb_weight, j, j, 0);
            pack_one(cb_normed);
        }
        cb_pop_front(cb_scaled, head_tiles);
        cb_pop_front(cb_weight, head_tiles);

        // rotary_embedding: y * cos + rotate_half(y) * sin, rotate_half(y)
        // = [-y2, y1] by whole tiles, cos and sin row 0 for every row. Products
        // round symmetrically, so the first half subtracts y2 * sin.
        cb_wait_front(cb_normed, head_tiles);
        cb_wait_front(cb_cos, rotary_tiles);
        cb_wait_front(cb_sin, rotary_tiles);
        reconfig_data_format(cb_normed, cb_sin);
        pack_reconfig_data_format(cb_out);
        for (uint32_t j = 0; j < rotary_tiles; ++j) {
            mul_bcast_rows_init(cb_normed, cb_sin);
            tile_regs_acquire();
            mul_tiles_bcast_rows(cb_normed, cb_sin, j < half ? j + half : j - half, j, 0);
            pack_one(cb_sin_part);
            tile_regs_acquire();
            mul_tiles_bcast_rows(cb_normed, cb_cos, j, j, 0);
            pack_one(cb_cos_part);

            cb_wait_front(cb_cos_part, 1);
            cb_wait_front(cb_sin_part, 1);
            tile_regs_acquire();
            if (j < half) {
                sub_init(cb_cos_part, cb_sin_part);
                sub_tiles(cb_cos_part, cb_sin_part, 0, 0, 0);
            } else {
                add_init(cb_cos_part, cb_sin_part);
                add_tiles(cb_cos_part, cb_sin_part, 0, 0, 0);
            }
            pack_one(cb_out);
            cb_pop_front(cb_cos_part, 1);
            cb_pop_front(cb_sin_part, 1);
        }
        if constexpr (rotary_tiles < head_tiles) {
            copy_init(cb_normed);
            for (uint32_t j = rotary_tiles; j < head_tiles; ++j) {
                tile_regs_acquire();
                copy_tile(cb_normed, j, 0);
                pack_one(cb_out);
            }
        }
        cb_pop_front(cb_normed, head_tiles);
        cb_pop_front(cb_cos, rotary_tiles);
        cb_pop_front(cb_sin, rotary_tiles);
    }
}
