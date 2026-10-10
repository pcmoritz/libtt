// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#include "api/compute/transpose_dest.h"

#if defined(TRISC_MATH) && defined(ARCH_BLACKHOLE)
#include "llk_sfpu/ckernel_sfpu_matmul_fp32.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#endif

namespace ckernel {

#if defined(ARCH_BLACKHOLE)

// FP32 matmul on the vector unit. Unlike matmul_tiles, which reads FP32
// inputs as TF32 on the matrix unit, every product and sum is an FP32
// multiply-add with one rounding. DST must hold FP32 data (fp32_dest_acc_en =
// true, with the inputs unpacked to DST as FP32). For C = sum_k A_k * B_k:
//
//     matmul_fp32_transpose_a(a);                    // for each k:
//     matmul_fp32_tile_init();
//     matmul_fp32_tile<k != 0, a, b, acc>();
//     ...
//     matmul_fp32_tile_finish(acc, c, scratch);      // once
//
// acc holds the sums in an internal order; matmul_fp32_tile_finish writes C
// in the usual tile order. acc, c and scratch must be distinct, and scratch
// is overwritten.

// clang-format off
/**
 * Transposes the A tile at idst_a in DST, exactly, as matmul_fp32_tile expects.
 *
 * Return value: None
 */
// clang-format on
ALWI void matmul_fp32_transpose_a(uint32_t idst_a) {
    transpose_dest_init</*is_32bit=*/true, /*transpose_of_faces=*/true>();
    transpose_dest</*is_32bit=*/true, /*transpose_of_faces=*/true>(idst_a);
}

// clang-format off
/**
 * Initializes the vector unit for matmul_fp32_tile and matmul_fp32_tile_finish.
 * Call it again after other operations, including matmul_fp32_transpose_a.
 *
 * Return value: None
 */
// clang-format on
ALWI void matmul_fp32_tile_init() { MATH((SFPU_BINARY_INIT_FN_NO_ARGS(unused, sfpu::_init_matmul_fp32_))); }

// clang-format off
/**
 * Adds the product of two 32x32 tiles to the accumulator tile at idst_acc
 * (or sets it, if ACCUMULATE is false). idst_at holds A transposed (see
 * matmul_fp32_transpose_a), idst_b holds B. The DST indices are template
 * parameters so that the kernel's addresses are compile-time constants.
 *
 * The DST register buffer must be in acquired state via *tile_regs_acquire* call.
 *
 * Return value: None
 *
 * | Template argument | Description                           | Type     | Valid Range                                           | Required |
 * |-------------------|---------------------------------------|----------|-------------------------------------------------------|----------|
 * | ACCUMULATE        | Add to (true) or set (false) idst_acc | bool     |                                                       | True     |
 * | idst_at           | Index of the transposed A tile in DST | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst_b            | Index of the B tile in DST            | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst_acc          | Index of the accumulator tile in DST  | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <bool ACCUMULATE, uint32_t idst_at, uint32_t idst_b, uint32_t idst_acc>
ALWI void matmul_fp32_tile() {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        _calculate_matmul_fp32_,
        (ACCUMULATE, idst_at, idst_b, idst_acc),
        idst_at,
        idst_b,
        idst_acc,
        VectorMode::None)));
}

// clang-format off
/**
 * Writes the product accumulated at idst_acc to idst_c as an ordinary tile,
 * using idst_scratch as scratch space.
 *
 * Return value: None
 */
// clang-format on
ALWI void matmul_fp32_tile_finish(uint32_t idst_acc, uint32_t idst_c, uint32_t idst_scratch) {
    MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        _matmul_fp32_finish_,
        idst_acc,
        idst_c,
        idst_scratch,
        VectorMode::None)));
}

#endif  // ARCH_BLACKHOLE

}  // namespace ckernel
