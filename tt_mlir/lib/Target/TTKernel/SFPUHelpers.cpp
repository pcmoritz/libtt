// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

// The C++ that TTKernelToCpp emits into D2M compute kernels for libtt's
// kernel-level rewrites (see ttmlir/D2MFusion.h).

#include "ttmlir/D2MFusion.h"

namespace {

// Helpers for fused SFPU chains (D2MToTTKernel's fuseSFPUChains), math
// thread only. rsqrt: the SQRT_23-bits estimate with a reciprocal refinement
// (as libtt's l2_clip kernel); recip: two Newton steps on approx_recip.
static const char *const kLibttSfpuHelpers = R"libtt(#ifdef TRISC_MATH
#include "sfpi.h"
#include "ckernel_sfpu_exp.h"
namespace libtt_sfpu {
sfpi_inline sfpi::vFloat rsqrt_positive(sfpi::vFloat x) {
  sfpi::vInt i = sfpi::as<sfpi::vInt>(sfpi::as<sfpi::vUInt>(x) >> 1);
  sfpi::vFloat y = sfpi::as<sfpi::vFloat>(sfpi::vInt(0x5f1110a0) - i);
  sfpi::vFloat xy = x * y;
  sfpi::vFloat c = -y * xy;
  y = y * (2.2825186f + c * (2.2533049f + c));
  xy = x * y;
  sfpi::vFloat one_minus_xyy = 1.0f - y * xy;
  return one_minus_xyy * sfpi::addexp(y, -1) + y;
}
sfpi_inline sfpi::vFloat rsqrt(sfpi::vFloat x) {
  sfpi::vFloat y = rsqrt_positive(x);
  v_if(x == 0.0f) { y = sfpi::as<sfpi::vFloat>(sfpi::vInt(0x7f800000)); }
  v_endif;
  return y;
}
sfpi_inline sfpi::vFloat sqrt(sfpi::vFloat x) {
  sfpi::vFloat y = x * rsqrt_positive(x);
  v_if(x == 0.0f) { y = 0.0f; }
  v_endif;
  return y;
}
sfpi_inline sfpi::vFloat recip(sfpi::vFloat x) {
  sfpi::vFloat y = sfpi::approx_recip(x);
  y = y * (2.0f - x * y);
  y = y * (2.0f - x * y);
  return y;
}
// Accurate FP32 exp (tt-llk's), sigmoid without overflow: t = exp(-|x|) is in
// (0, 1]; sigmoid(x) = 1 / (1 + t) for x >= 0, t / (1 + t) otherwise.
sfpi_inline sfpi::vFloat exp(sfpi::vFloat x) {
  return ckernel::sfpu::_sfpu_exp_accurate_<true>(x);
}
sfpi_inline sfpi::vFloat sigmoid(sfpi::vFloat x) {
  sfpi::vFloat t = exp(-sfpi::abs(x));
  sfpi::vFloat d = recip(1.0f + t);
  v_if(x < 0.0f) { d = t * d; }
  v_endif;
  return d;
}
sfpi_inline sfpi::vFloat silu(sfpi::vFloat x) { return x * sigmoid(x); }
sfpi_inline sfpi::vFloat bits(uint32_t b) {
  return sfpi::vFloat(__builtin_bit_cast(float, b));
}
// In-DST tile helpers. FP32 DST lane layout: vector d (0..7) of face f holds,
// in lane l, the face's element at row 4 * (d / 2) + l / 8, column
// 2 * (l % 8) + d % 2; dst_reg[t * 32 + f * 8 + d] from sfpu start 0.
sfpi_inline sfpi::vInt rotate8(sfpi::vInt v) {
  return sfpi::vInt(sfpi::subvec_shflror1(v));
}
// Every lane of each 8-lane subvector gets the OR of the subvector's lanes:
// with one nonzero lane, its bits (exact for any 32-bit type).
sfpi_inline sfpi::vInt or_lanes8(sfpi::vInt v) {
  v |= rotate8(v);
  sfpi::vInt r = rotate8(v);
  v |= rotate8(r);
  r = rotate8(v);
  r = rotate8(r);
  r = rotate8(r);
  v |= rotate8(r);
  return v;
}
// dst = src (slices as template arguments: compile-time DST addresses).
template <uint32_t src, uint32_t dst>
inline void copy_dst() {
  llk_math_eltwise_unary_sfpu_init<SfpuType::unused, DST_ACCUM_MODE>();
  _llk_math_eltwise_sfpu_start_(0);
#pragma GCC unroll 32
  for (uint32_t i = 0; i < 32; ++i) {
    sfpi::vInt v = sfpi::dst_reg[src * 32 + i];
    sfpi::dst_reg[dst * 32 + i] = v;
  }
  _llk_math_eltwise_sfpu_done_();
}
// dst = column 0 of src spread across each row.
template <uint32_t src, uint32_t dst>
inline void bcast_col0() {
  llk_math_eltwise_unary_sfpu_init<SfpuType::unused, DST_ACCUM_MODE>();
  _llk_math_eltwise_sfpu_start_(0);
#pragma GCC unroll 2
  for (uint32_t half = 0; half < 2; ++half) {
#pragma GCC unroll 4
    for (uint32_t band = 0; band < 4; ++band) {
      // Integer loads and stores: raw bits (float loads flush denormals,
      // which int32 data looks like).
      sfpi::vInt bits = sfpi::dst_reg[src * 32 + 2 * half * 8 + 2 * band];
      v_if((sfpi::vInt(sfpi::vConstTileId) & 14) != 0) { bits = 0; }
      v_endif;
      sfpi::vInt v = or_lanes8(bits);
      sfpi::dst_reg[dst * 32 + (2 * half) * 8 + 2 * band] = v;
      sfpi::dst_reg[dst * 32 + (2 * half) * 8 + 2 * band + 1] = v;
      sfpi::dst_reg[dst * 32 + (2 * half + 1) * 8 + 2 * band] = v;
      sfpi::dst_reg[dst * 32 + (2 * half + 1) * 8 + 2 * band + 1] = v;
    }
  }
  _llk_math_eltwise_sfpu_done_();
}
// dst = row 0 of src spread down each column (in place allowed): a
// subvector transpose of four copies of a row-0 vector repeats its first
// subvector (row 0) in every subvector.
template <uint32_t src, uint32_t dst>
inline void bcast_row0() {
  llk_math_eltwise_unary_sfpu_init<SfpuType::unused, DST_ACCUM_MODE>();
  _llk_math_eltwise_sfpu_start_(0);
#pragma GCC unroll 2
  for (uint32_t face = 0; face < 2; ++face) {
#pragma GCC unroll 2
    for (uint32_t parity = 0; parity < 2; ++parity) {
      sfpi::vInt a = sfpi::dst_reg[src * 32 + face * 8 + parity];
      sfpi::vInt b = a, c = a, d = a, e = a, f = a, g = a, h = a;
      sfpi::subvec_transp(a, b, c, d, e, f, g, h);
#pragma GCC unroll 4
      for (uint32_t band = 0; band < 4; ++band) {
        sfpi::dst_reg[dst * 32 + face * 8 + 2 * band + parity] = a;
        sfpi::dst_reg[dst * 32 + (face + 2) * 8 + 2 * band + parity] = a;
      }
    }
  }
  _llk_math_eltwise_sfpu_done_();
}
sfpi_inline sfpi::vFloat rotate8f(sfpi::vFloat v) {
  return sfpi::vFloat(sfpi::subvec_shflror1(v));
}
// Every lane of each 8-lane subvector gets the subvector's sum.
sfpi_inline sfpi::vFloat sum_lanes8f(sfpi::vFloat v) {
  v += rotate8f(v);
  sfpi::vFloat r = rotate8f(v);
  v += rotate8f(r);
  r = rotate8f(v);
  r = rotate8f(r);
  r = rotate8f(r);
  v += rotate8f(r);
  return v;
}
// Each row's 16-column halves (faces 0/2 and 1/3) summed and broadcast in
// place: a face row is the even vector (columns 0, 2, .., 14) and the odd
// vector (1, 3, .., 15) of one 8-lane subvector.
inline void group_sum16(uint32_t idst) {
  llk_math_eltwise_unary_sfpu_init<SfpuType::unused, DST_ACCUM_MODE>();
  _llk_math_eltwise_sfpu_start_(0);
#pragma GCC unroll 0
  for (int face = 0; face < 4; face++) {
#pragma GCC unroll 0
    for (int band = 0; band < 4; band++) {
      sfpi::vFloat even = sfpi::dst_reg[idst * 32];
      sfpi::vFloat odd = sfpi::dst_reg[idst * 32 + 1];
      sfpi::vFloat sum = sum_lanes8f(even + odd);
      sfpi::dst_reg[idst * 32] = sum;
      sfpi::dst_reg[idst * 32 + 1] = sum;
      sfpi::dst_reg += 2;
    }
    _llk_math_eltwise_sfpu_inc_dst_face_addr_();
  }
  _llk_math_eltwise_sfpu_done_();
}
// Element (i, j) = element (0, i): row 0 made the columns, in place. Per
// 4-row band of the result: row 0's even and odd vectors repeated in every
// subvector (subvector transpose), subvector g takes the parity of its row's
// source column, keeps that column's lane, and ORs it across the subvector.
inline void row_to_column(uint32_t idst) {
  llk_math_eltwise_unary_sfpu_init<SfpuType::unused, DST_ACCUM_MODE>();
  _llk_math_eltwise_sfpu_start_(0);
  // Row 0's even and odd vectors repeated in every subvector, stored back
  // over themselves (rows 1-3 of a row tile are padding). The transpose uses
  // all eight registers, so nothing else is live across it.
#pragma GCC unroll 4
  for (uint32_t k = 0; k < 4; ++k) {
    const uint32_t at = idst * 32 + (k / 2) * 8 + k % 2;
    sfpi::vInt a = sfpi::dst_reg[at];
    sfpi::vInt b = a, c = a, d = a, e = a, f = a, g = a, h = a;
    sfpi::subvec_transp(a, b, c, d, e, f, g, h);
    sfpi::dst_reg[at] = a;
  }
  // vConstTileId is twice the lane: bits 1-3 the lane in its subvector,
  // bits 4-5 the subvector.
  sfpi::vInt id = sfpi::vConstTileId;
  sfpi::vInt sub = sfpi::as<sfpi::vInt>(sfpi::as<sfpi::vUInt>(id) >> 4) & 3;
  sfpi::vInt lane2 = id & 14;
  // Source face 1 (columns 16-31) first: its rows go to faces 2 and 3; face
  // 0's go to faces 0 and 1, band 0 (over the stored row vectors) last.
#pragma GCC unroll 2
  for (int face = 1; face >= 0; --face) {
#pragma GCC unroll 4
    for (int band = 3; band >= 0; --band) {
      sfpi::vInt v = sfpi::dst_reg[idst * 32 + face * 8];
      v_if((sub & 1) != 0) { v = sfpi::dst_reg[idst * 32 + face * 8 + 1]; }
      v_endif;
      // Column 4 * band + sub of the source face sits in lane
      // 2 * band + sub / 2 of its parity's vector.
      v_if(lane2 != (sub & 2) + 4 * band) { v = 0; }
      v_endif;
      v = or_lanes8(v);
#pragma GCC unroll 2
      for (uint32_t cf = 0; cf < 2; ++cf) {
        sfpi::dst_reg[idst * 32 + (2 * face + cf) * 8 + 2 * band] = v;
        sfpi::dst_reg[idst * 32 + (2 * face + cf) * 8 + 2 * band + 1] = v;
      }
    }
  }
  _llk_math_eltwise_sfpu_done_();
}
} // namespace libtt_sfpu
#endif)libtt";

// tt-metal (334f0f9) renamed these calls; D2M still emits the old
// names, defined here on the new API with the old semantics. (Matmul inits
// changed shape too; D2M matmuls are not on libtt's path.)
static const char *const kLibttBinaryCompat = R"libtt(
ALWI void binary_op_init_common(uint32_t icb0, uint32_t icb1, uint32_t ocb) {
  ckernel::compute_kernel_hw_startup(icb0, icb1, ocb);
}
template <ckernel::EltwiseBinaryType type,
          ckernel::EltwiseBinaryReuseDestType reuse>
ALWI void binary_dest_reuse_tiles_init(uint32_t icb) {
  ckernel::detail::binary_reuse_dest_init<type, reuse>(icb, __builtin_LINE());
}
template <ckernel::EltwiseBinaryType type,
          ckernel::EltwiseBinaryReuseDestType reuse>
ALWI void binary_dest_reuse_tiles(uint32_t cb, uint32_t tile, uint32_t dst) {
  ckernel::detail::binary_reuse_dest_tiles<type, reuse>(cb, tile, dst);
}
ALWI void add_tiles_init(uint32_t icb0, uint32_t icb1) {
  ckernel::binary_tiles_init<true, ckernel::EltwiseBinaryType::ELWADD>(
      icb0, icb1, false, __builtin_LINE());
}
ALWI void sub_tiles_init(uint32_t icb0, uint32_t icb1) {
  ckernel::binary_tiles_init<true, ckernel::EltwiseBinaryType::ELWSUB>(
      icb0, icb1, false, __builtin_LINE());
}
ALWI void mul_tiles_init(uint32_t icb0, uint32_t icb1) {
  ckernel::binary_tiles_init<true, ckernel::EltwiseBinaryType::ELWMUL>(
      icb0, icb1, false, __builtin_LINE());
}
ALWI void pack_tile_block(uint32_t ifrom_dst, uint32_t icb, uint32_t ntiles) {
  ckernel::pack_block(ifrom_dst, icb, ntiles);
})libtt";

// libtt tile ops called from kernel_main on every TRISC.
static const char *const kLibttTileOps = R"libtt(
ALWI void libtt_group_sum_tile_init() {}
ALWI void libtt_row_to_column_tile_init() {}
ALWI void libtt_row_to_column_tile(uint32_t idst) {
  MATH((libtt_sfpu::row_to_column(idst)));
}
ALWI void libtt_group_sum_tile(uint32_t idst) {
  MATH((libtt_sfpu::group_sum16(idst)));
})libtt";

} // namespace

namespace mlir::tt::d2m_fusion {

llvm::StringRef sfpuChainHelpers() { return kLibttSfpuHelpers; }
llvm::StringRef binaryCompatHelpers() { return kLibttBinaryCompat; }
llvm::StringRef tileOpHelpers() { return kLibttTileOps; }

} // namespace mlir::tt::d2m_fusion
