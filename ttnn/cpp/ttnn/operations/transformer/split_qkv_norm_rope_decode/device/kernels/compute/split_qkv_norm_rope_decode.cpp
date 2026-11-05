// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reduce.h"

// For each q or k unit, a tile row of heads (one per row): rms_norm, then
// rotary_embedding, each step as those ops' kernels take it (layernorm.cpp
// with RMSNORM and gamma; rotary_embedding.cpp in decode mode), in the same
// buffer formats, so that every rounding is theirs:
//   x^2 (FP32), its row sum times 1 / width (FP32), + epsilon and rsqrt
//   (FP32), x * rstd (FP32), * weight (BF16);
//   -y of the other half (BF16), * sin (BF16), y * cos (BF16), their sum
//   (BF16).
void kernel_main() {
    constexpr uint32_t cb_x = get_named_compile_time_arg_val("cb_x");
    constexpr uint32_t cb_weight = get_named_compile_time_arg_val("cb_weight");
    constexpr uint32_t cb_cos = get_named_compile_time_arg_val("cb_cos");
    constexpr uint32_t cb_sin = get_named_compile_time_arg_val("cb_sin");
    constexpr uint32_t cb_scaler = get_named_compile_time_arg_val("cb_scaler");
    constexpr uint32_t cb_eps = get_named_compile_time_arg_val("cb_eps");
    constexpr uint32_t cb_neg = get_named_compile_time_arg_val("cb_neg");
    constexpr uint32_t cb_sq = get_named_compile_time_arg_val("cb_sq");
    constexpr uint32_t cb_mean = get_named_compile_time_arg_val("cb_mean");
    constexpr uint32_t cb_rstd = get_named_compile_time_arg_val("cb_rstd");
    constexpr uint32_t cb_scaled = get_named_compile_time_arg_val("cb_scaled");
    constexpr uint32_t cb_normed = get_named_compile_time_arg_val("cb_normed");
    constexpr uint32_t cb_rotated = get_named_compile_time_arg_val("cb_rotated");
    constexpr uint32_t cb_sin_part = get_named_compile_time_arg_val("cb_sin_part");
    constexpr uint32_t cb_cos_part = get_named_compile_time_arg_val("cb_cos_part");
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t head_tiles = get_named_compile_time_arg_val("head_tiles");
    constexpr uint32_t inv_width = get_named_compile_time_arg_val("inv_width");
    constexpr uint32_t half = head_tiles / 2;
    constexpr uint32_t kV = 2;
    constexpr uint32_t kEnd = 0xFFFFFFFF;

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
    cb_wait_front(cb_scaler, 1);
    cb_wait_front(cb_eps, 1);
    cb_wait_front(cb_neg, 1);
    for (uint32_t arg = 0;; ++arg) {
        const uint32_t kind = get_arg_val<uint32_t>(arg);
        if (kind == kEnd) {
            break;
        }
        if (kind == kV) {
            continue;
        }

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

        // The row sum of x^2 by a reduce against the scaler's ones, times
        // 1 / width on the SFPU.
        reconfig_data_format(cb_x, cb_sq, cb_x, cb_scaler);
        reconfig_data_format(cb_sq, cb_scaler);
        tile_regs_acquire();
        reconfig_data_format(cb_scaler, cb_sq);
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_scaler, cb_mean);
        cb_wait_front(cb_sq, head_tiles);
        for (uint32_t j = 0; j < head_tiles; ++j) {
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_scaler, j, 0, 0);
        }
        cb_pop_front(cb_sq, head_tiles);
        reduce_uninit();
        reconfig_data_format(cb_sq, cb_scaler);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, inv_width);
        pack_reconfig_data_format(cb_mean);
        pack_one(cb_mean);

        // rstd = rsqrt(mean + epsilon), down column 0.
        cb_wait_front(cb_mean, 1);
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
        // = [-y2, y1] by whole tiles, cos and sin row 0 for every row.
        cb_wait_front(cb_normed, head_tiles);
        cb_wait_front(cb_cos, head_tiles);
        cb_wait_front(cb_sin, head_tiles);
        for (uint32_t j = 0; j < head_tiles; ++j) {
            if (j < half) {
                reconfig_data_format(cb_normed, cb_neg);
                pack_reconfig_data_format(cb_rotated);
                mul_bcast_scalar_init(cb_normed, cb_neg);
                tile_regs_acquire();
                mul_tiles_bcast_scalar(cb_normed, cb_neg, j + half, 0, 0);
                pack_one(cb_rotated);
                cb_wait_front(cb_rotated, 1);
                reconfig_data_format(cb_rotated, cb_sin);
                pack_reconfig_data_format(cb_sin_part);
                mul_bcast_rows_init(cb_rotated, cb_sin);
                tile_regs_acquire();
                mul_tiles_bcast_rows(cb_rotated, cb_sin, 0, j, 0);
                pack_one(cb_sin_part);
                cb_pop_front(cb_rotated, 1);
            } else {
                reconfig_data_format(cb_normed, cb_sin);
                pack_reconfig_data_format(cb_sin_part);
                mul_bcast_rows_init(cb_normed, cb_sin);
                tile_regs_acquire();
                mul_tiles_bcast_rows(cb_normed, cb_sin, j - half, j, 0);
                pack_one(cb_sin_part);
            }
            reconfig_data_format(cb_normed, cb_cos);
            pack_reconfig_data_format(cb_cos_part);
            mul_bcast_rows_init(cb_normed, cb_cos);
            tile_regs_acquire();
            mul_tiles_bcast_rows(cb_normed, cb_cos, j, j, 0);
            pack_one(cb_cos_part);

            cb_wait_front(cb_cos_part, 1);
            cb_wait_front(cb_sin_part, 1);
            reconfig_data_format(cb_cos_part, cb_sin_part);
            pack_reconfig_data_format(cb_out);
            add_init(cb_cos_part, cb_sin_part);
            tile_regs_acquire();
            add_tiles(cb_cos_part, cb_sin_part, 0, 0, 0);
            pack_one(cb_out);
            cb_pop_front(cb_cos_part, 1);
            cb_pop_front(cb_sin_part, 1);
        }
        cb_pop_front(cb_normed, head_tiles);
        cb_pop_front(cb_cos, head_tiles);
        cb_pop_front(cb_sin, head_tiles);
    }
}
