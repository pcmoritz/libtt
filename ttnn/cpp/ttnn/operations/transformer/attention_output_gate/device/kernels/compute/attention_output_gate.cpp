// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"

#ifdef TRISC_MATH
#include "sfpi.h"

namespace attention_output_gate {
// Each value of DST tile `dst` rounded to BF16, to nearest with ties to even
// (as ttnn.typecast and ttnn.multiply's BF16 outputs round), kept as FP32.
inline void round_to_bf16(uint32_t dst) {
    _llk_math_eltwise_sfpu_start_(dst);
    for (uint32_t face = 0; face < 4; ++face) {
#pragma GCC unroll 8
        for (uint32_t d = 0; d < 8; ++d) {
            sfpi::vUInt bits = sfpi::as<sfpi::vUInt>(sfpi::vFloat(sfpi::dst_reg[d]));
            bits = bits + 0x7fffU + ((bits >> 16) & 1);
            sfpi::dst_reg[d] = sfpi::as<sfpi::vFloat>(bits & 0xffff0000U);
        }
        _llk_math_eltwise_sfpu_inc_dst_face_addr_();
    }
    _llk_math_eltwise_sfpu_done_();
}
}  // namespace attention_output_gate
#endif

// For each output tile, as PyTorch's attention_output * sigmoid(gate) rounds
// on TT: the BF16 gate in FP32 DST, ttnn.sigmoid's FP32 sigmoid, rounded to
// BF16 (the typecast back), times the BF16 attention values, exact in FP32,
// rounded to BF16 (ttnn.multiply's output).
void kernel_main() {
    const uint32_t count = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_gate = get_named_compile_time_arg_val("cb_gate");
    constexpr uint32_t cb_values = get_named_compile_time_arg_val("cb_values");
    constexpr uint32_t cb_output = get_named_compile_time_arg_val("cb_output");

    compute_kernel_hw_startup(cb_gate, cb_output);
    for (uint32_t i = 0; i < count; ++i) {
        cb_wait_front(cb_gate, 1);
        cb_wait_front(cb_values, 1);
        tile_regs_acquire();
        copy_tile_to_dst_init_short(cb_gate);
        copy_tile(cb_gate, 0, 0);
        sigmoid_tile_init<false>();
        sigmoid_tile<VectorMode::RC, false>(0);
        MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::unused, DST_ACCUM_MODE>()));
        MATH((attention_output_gate::round_to_bf16(0)));
        copy_tile_to_dst_init_short(cb_values);
        copy_tile(cb_values, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::unused, DST_ACCUM_MODE>()));
        MATH((attention_output_gate::round_to_bf16(0)));
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_output, 1);
        pack_tile(0, cb_output);
        cb_push_back(cb_output, 1);
        tile_regs_release();
        cb_pop_front(cb_gate, 1);
        cb_pop_front(cb_values, 1);
    }
}
