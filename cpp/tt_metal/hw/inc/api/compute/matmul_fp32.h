// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"

#if defined(TRISC_MATH) && defined(ARCH_BLACKHOLE)
#include "llk_sfpu/ckernel_sfpu_matmul_fp32.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#endif

namespace ckernel {

#if defined(ARCH_BLACKHOLE)

// clang-format off
/**
 * Initializes the vector unit for matmul_fp32_tile. Call it once before the
 * first matmul_fp32_tile, and again after other SFPU operations.
 *
 * Return value: None
 */
// clang-format on
ALWI void matmul_fp32_tile_init() { MATH((SFPU_BINARY_INIT_FN_NO_ARGS(unused, sfpu::_init_matmul_fp32_))); }

// clang-format off
/**
 * Multiplies two 32x32 tiles in DST on the vector unit with FP32 precision:
 *
 *     DST[idst_c] = DST[idst_a] * DST[idst_b]                  (ACCUMULATE = false)
 *     DST[idst_c] = DST[idst_a] * DST[idst_b] + DST[idst_c]    (ACCUMULATE = true)
 *
 * Unlike matmul_tiles, which reads FP32 inputs as TF32 on the matrix unit,
 * every product and sum is an FP32 multiply-add with one rounding. DST must
 * hold FP32 data (fp32_dest_acc_en = true, with the inputs unpacked to DST as
 * FP32). idst_c must differ from idst_a and idst_b.
 *
 * The DST register buffer must be in acquired state via *tile_regs_acquire* call.
 *
 * Return value: None
 *
 * | Argument   | Description                     | Type     | Valid Range                                           | Required |
 * |------------|---------------------------------|----------|-------------------------------------------------------|----------|
 * | idst_a     | Index of the A tile in DST      | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst_b     | Index of the B tile in DST      | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst_c     | Index of the output tile in DST | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <bool ACCUMULATE = true>
ALWI void matmul_fp32_tile(uint32_t idst_a, uint32_t idst_b, uint32_t idst_c) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        _calculate_matmul_fp32_,
        (ACCUMULATE),
        idst_a,
        idst_b,
        idst_c,
        VectorMode::None)));
}

#endif  // ARCH_BLACKHOLE

}  // namespace ckernel
