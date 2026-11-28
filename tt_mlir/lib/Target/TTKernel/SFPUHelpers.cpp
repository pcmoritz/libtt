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
// FP32 x / y as tt-metal's div_binary_tile (calculate_sfpu_binary_div with
// FP32 DST), bit for bit: the quotient of the mantissas refined once, the
// exponent restored in two factors, exceptional divisors by their inverted
// exponent. (Its reciprocal's 2.0 is vConstFloatPrgm0 there.)
sfpi_inline sfpi::vFloat div(sfpi::vFloat in0, sfpi::vFloat in1) {
  sfpi::vFloat ma = sfpi::setexp(in0, 127);
  sfpi::vFloat mb = sfpi::setexp(in1, 127);
  sfpi::vFloat r = sfpi::approx_recip(mb);
  sfpi::vFloat t = mb * r - 2.0f;
  r = r * -t - 0.0f;
  sfpi::vFloat q = ma * r;
  sfpi::vFloat residual = ma - q * mb;
  q = q + residual * r;
  sfpi::vInt ea = sfpi::exexp(in0, sfpi::ExponentMode::Biased);
  sfpi::vInt eb = sfpi::exexp(in1);
  sfpi::vInt exponent = ea - eb + sfpi::exexp(q, sfpi::ExponentMode::Biased);
  sfpi::vInt half =
      sfpi::as<sfpi::vInt>(sfpi::as<sfpi::vUInt>(exponent) >> 1);
  sfpi::vFloat result = sfpi::setexp(q, half) *
                        sfpi::setexp(sfpi::vFloat(1.0f), exponent - half);
  v_if(!(ea >= 1 && ea < 255 && eb >= -126 && eb < 128)) {
    sfpi::vFloat scale =
        sfpi::setman(sfpi::as<sfpi::vFloat>(~sfpi::as<sfpi::vInt>(in1)), 0);
    sfpi::vFloat nan_divisor = sfpi::as<sfpi::vFloat>(
        sfpi::mask(sfpi::setsgn(in1, 0) > __builtin_inff()));
    result = nan_divisor - in0 * scale;
  }
  v_endif;
  return result;
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
// The in-DST helpers' rows: only rows < rows of dst are written (libtt
// tile_rows kernels); vector i of a tile holds rows 16 (i / 16) +
// 4 (i % 8 / 2) .. + 3.
constexpr bool vec_has_rows(uint32_t i, uint32_t rows) {
  return 16 * (i / 16) + 4 * (i % 8 / 2) < rows;
}
template <uint32_t src, uint32_t dst, uint32_t rows = 32>
inline void copy_dst() {
  llk_math_eltwise_unary_sfpu_init<SfpuType::unused, DST_ACCUM_MODE>();
  _llk_math_eltwise_sfpu_start_(0);
#pragma GCC unroll 32
  for (uint32_t i = 0; i < 32; ++i) {
    if (!vec_has_rows(i, rows)) {
      continue;
    }
    sfpi::vInt v = sfpi::dst_reg[src * 32 + i];
    sfpi::dst_reg[dst * 32 + i] = v;
  }
  _llk_math_eltwise_sfpu_done_();
}
// dst = column 0 of src spread across each row.
template <uint32_t src, uint32_t dst, uint32_t rows = 32>
inline void bcast_col0() {
  llk_math_eltwise_unary_sfpu_init<SfpuType::unused, DST_ACCUM_MODE>();
  _llk_math_eltwise_sfpu_start_(0);
#pragma GCC unroll 2
  for (uint32_t half = 0; half < 2; ++half) {
#pragma GCC unroll 4
    for (uint32_t band = 0; band < 4; ++band) {
      if (16 * half + 4 * band >= rows) {
        continue;
      }
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
// Rows [from, 32) of DST tile t set to zero (raw bits): a kernel of fewer
// tile rows (libtt.tile_rows) leaves them as DST held them, and an output's
// padding rows are read by later ops (NaN there hung Inkling 2's decode).
template <uint32_t t, uint32_t from>
inline void zero_rows() {
  llk_math_eltwise_unary_sfpu_init<SfpuType::unused, DST_ACCUM_MODE>();
  _llk_math_eltwise_sfpu_start_(0);
#pragma GCC unroll 8
  for (uint32_t band = from / 4; band < 8; ++band) {
#pragma GCC unroll 2
    for (uint32_t h = 0; h < 2; ++h) {
#pragma GCC unroll 2
      for (uint32_t p = 0; p < 2; ++p) {
        const uint32_t a = t * 32 + (2 * (band / 4) + h) * 8 + 2 * (band % 4) + p;
        if (band * 4 >= from) {
          sfpi::dst_reg[a] = sfpi::vInt(0);
        } else {
          sfpi::vInt v = sfpi::dst_reg[a];
          v_if((sfpi::as<sfpi::vInt>(sfpi::as<sfpi::vUInt>(
                    sfpi::vInt(sfpi::vConstTileId)) >> 4) & 3) >=
               static_cast<int>(from - band * 4)) {
            v = 0;
          }
          v_endif;
          sfpi::dst_reg[a] = v;
        }
      }
    }
  }
  _llk_math_eltwise_sfpu_done_();
}
// dst = row 0 of src spread down each column (in place allowed): a
// subvector transpose of four copies of a row-0 vector repeats its first
// subvector (row 0) in every subvector.
template <uint32_t src, uint32_t dst, uint32_t rows = 32>
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
        if (4 * band < rows) {
          sfpi::dst_reg[dst * 32 + face * 8 + 2 * band + parity] = a;
        }
        if (16 + 4 * band < rows) {
          sfpi::dst_reg[dst * 32 + (face + 2) * 8 + 2 * band + parity] = a;
        }
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
// Row r of DST tile t: its vector of column half h (face 2 (r / 16) + h) and
// parity p; the row is subvector r % 4 of it.
sfpi_inline uint32_t row_vec(uint32_t t, uint32_t r, uint32_t h, uint32_t p) {
  return t * 32 + (2 * (r / 16) + h) * 8 + 2 * (r % 16 / 4) + p;
}
// v's subvector s in every subvector.
sfpi_inline sfpi::vInt replicate_subvec(sfpi::vInt v, uint32_t s) {
  sfpi::vInt a = v, b = v, c = v, d = v, e = v, f = v, g = v, h = v;
  sfpi::subvec_transp(a, b, c, d, e, f, g, h);
  return s == 0 ? a : s == 1 ? b : s == 2 ? c : d;
}
// Tile out = each column's sum (max) over tile in's rows [0, n) in every row:
// the bands' vectors combined elementwise (rows past n at the identity),
// then the four subvectors of the result (a subvector transpose, which
// needs all eight registers: nothing else is live across it). Only out's
// rows < rows are written (libtt.tile_rows kernels); with col0, only column
// 0 (the even vectors of faces 0 and 2).
sfpi_inline void column_reduce(uint32_t in, uint32_t out, bool max, uint32_t n,
                               uint32_t rows = 32, bool col0 = false) {
  llk_math_eltwise_unary_sfpu_init<SfpuType::unused, DST_ACCUM_MODE>();
  _llk_math_eltwise_sfpu_start_(0);
  const float identity = max ? -3.4028235e38f : 0.0f;
#pragma GCC unroll 1
  for (uint32_t h = 0; h < (col0 ? 1u : 2u); ++h) {
#pragma GCC unroll 1
    for (uint32_t p = 0; p < (col0 ? 1u : 2u); ++p) {
      sfpi::vFloat acc = identity;
      for (uint32_t b = 0; 4 * b < n; ++b) {
        sfpi::vFloat v = sfpi::dst_reg[in * 32 + (2 * (b / 4) + h) * 8 +
                                       2 * (b % 4) + p];
        if (4 * b + 4 > n) {
          sfpi::vInt sub = sfpi::as<sfpi::vInt>(sfpi::as<sfpi::vUInt>(
                               sfpi::vInt(sfpi::vConstTileId)) >> 4) &
                           3;
          v_if(sub >= static_cast<int>(n - 4 * b)) { v = identity; }
          v_endif;
        }
        if (max) {
          v_if(acc < v) { acc = v; }
          v_endif;
        } else {
          acc = acc + v;
        }
      }
      sfpi::vFloat a0 = acc, a1 = acc, a2 = acc, a3 = acc, a4 = acc, a5 = acc,
                   a6 = acc, a7 = acc;
      sfpi::subvec_transp(a0, a1, a2, a3, a4, a5, a6, a7);
      sfpi::vFloat r;
      if (max) {
        v_if(a0 < a1) { a0 = a1; }
        v_endif;
        v_if(a2 < a3) { a2 = a3; }
        v_endif;
        v_if(a0 < a2) { a0 = a2; }
        v_endif;
        r = a0;
      } else {
        r = (a0 + a1) + (a2 + a3);
      }
#pragma GCC unroll 8
      for (uint32_t band = 0; band < 8; ++band) {
        if (4 * band < rows) {
          sfpi::dst_reg[out * 32 + (2 * (band / 4) + h) * 8 +
                        2 * (band % 4) + p] = r;
        }
      }
    }
  }
  _llk_math_eltwise_sfpu_done_();
}
// out = dst with rows [to, to + count) = src's rows [from, from + count)
// (libtt_place_rows_tile); any of the three may be the same tile. Raw bits.
// Always inlined, with constant arguments: a partly constant clone
// (place_rows.constprop) crashed the SFPI compiler. Rows >= rows of out are
// left undefined (libtt.tile_rows kernels), and with col0 every column but 0
// (the even vectors of faces 0 and 2).
sfpi_inline void place_rows(uint32_t dst, uint32_t src, uint32_t out,
                            uint32_t to, uint32_t from, uint32_t count,
                            uint32_t rows = 32, bool col0 = false) {
  llk_math_eltwise_unary_sfpu_init<SfpuType::unused, DST_ACCUM_MODE>();
  _llk_math_eltwise_sfpu_start_(0);
  // Each lane's subvector (row in its band); recomputed where used: the
  // subvector transpose needs all eight registers.
  auto sub = []() __attribute__((always_inline)) {
    return sfpi::as<sfpi::vInt>(
               sfpi::as<sfpi::vUInt>(sfpi::vInt(sfpi::vConstTileId)) >> 4) &
           3;
  };
  // The vectors holding rows < rows: up to 16 rows, the first
  // 2 ceil(rows / 4) of faces 0 and 1; else all.
  const uint32_t faceVecs = rows > 16 ? 8 : 2 * ((rows + 3) / 4);
  const uint32_t numVecs = rows > 16 ? 32 : 16;
  if (out != src && out != dst) {
#pragma GCC unroll 8
    for (uint32_t i = 0; i < numVecs; ++i) {
      if (i % 8 >= faceVecs || (col0 && (i / 8 % 2 != 0 || i % 2 != 0))) {
        continue;
      }
      sfpi::vInt v = sfpi::dst_reg[dst * 32 + i];
      sfpi::dst_reg[out * 32 + i] = v;
    }
  }
  auto move = [&](uint32_t i) __attribute__((always_inline)) {
    const uint32_t r = to + i, q = from + i;
#pragma GCC unroll 2
    for (uint32_t h = 0; h < (col0 ? 1u : 2u); ++h) {
#pragma GCC unroll 2
      for (uint32_t p = 0; p < (col0 ? 1u : 2u); ++p) {
        const uint32_t sa = row_vec(src, q, h, p), oa = row_vec(out, r, h, p);
        sfpi::vInt v = sfpi::dst_reg[sa];
        if (sa != oa) {
          if (q % 4 != r % 4) {
            v = replicate_subvec(v, q % 4);
          }
          sfpi::vInt o = sfpi::dst_reg[oa];
          v_if(sub() == static_cast<int>(r % 4)) { o = v; }
          v_endif;
          sfpi::dst_reg[oa] = o;
        } else if (q != r) {
          // Both rows in one vector: rebuild it from its subvectors. (A
          // reload of the vector after the transpose is the load above,
          // which the compiler then keeps live across it: "too few lregs".)
          sfpi::vInt a = v, b = v, c = v, d = v, e = v, f = v, g = v, k = v;
          sfpi::subvec_transp(a, b, c, d, e, f, g, k);
          // Subvector j of the result: v's, q's for j = r.
          auto pick = [&](uint32_t j) __attribute__((always_inline)) {
            const uint32_t s = j == r % 4 ? q % 4 : j;
            return s == 0 ? a : s == 1 ? b : s == 2 ? c : d;
          };
          sfpi::vInt o = pick(0);
          v_if(sub() == 1) { o = pick(1); }
          v_endif;
          v_if(sub() == 2) { o = pick(2); }
          v_endif;
          v_if(sub() == 3) { o = pick(3); }
          v_endif;
          sfpi::dst_reg[oa] = o;
        }
      }
    }
  };
  // Within one tile, each row is read before it is written. Unrolled: a
  // move's code depends on its rows (v_if under a runtime branch does not
  // compile).
  if (out != src || to <= from) {
#pragma GCC unroll 32
    for (uint32_t i = 0; i < count; ++i) {
      move(i);
    }
  } else {
#pragma GCC unroll 32
    for (uint32_t i = count; i-- > 0;) {
      move(i);
    }
  }
  if (out == src && out != dst) {
    // dst's rows outside [to, to + count); vector i's first row is base.
    for (uint32_t i = 0; i < numVecs; ++i) {
      if (i % 8 >= faceVecs || (col0 && (i / 8 % 2 != 0 || i % 2 != 0))) {
        continue;
      }
      const int base = static_cast<int>(16 * (i / 16) + 4 * (i % 8 / 2));
      sfpi::vInt v = sfpi::dst_reg[out * 32 + i];
      const sfpi::vInt row = sub() + base;
      v_if(row < static_cast<int>(to) ||
           row >= static_cast<int>(to + count)) {
        v = sfpi::dst_reg[dst * 32 + i];
      }
      v_endif;
      sfpi::dst_reg[out * 32 + i] = v;
    }
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
// The chunk all-reduce's gather tiles: each core's partial row t in column
// t of its slot, gather_slot(rows) columns (libtt_gather_slot), 32 / slot
// slots per row.
constexpr uint32_t gather_slot(uint32_t rows) {
  uint32_t w = 1;
  while (w < rows) {
    w *= 2;
  }
  return w;
}
// dst += src (gather tiles).
sfpi_inline void add_gather(uint32_t dst, uint32_t src, uint32_t rows) {
  llk_math_eltwise_unary_sfpu_init<SfpuType::unused, DST_ACCUM_MODE>();
  _llk_math_eltwise_sfpu_start_(0);
#pragma GCC unroll 32
  for (uint32_t i = 0; i < 32; ++i) {
    sfpi::vFloat a = sfpi::dst_reg[dst * 32 + i];
    sfpi::vFloat b = sfpi::dst_reg[src * 32 + i];
    sfpi::dst_reg[dst * 32 + i] = a + b;
  }
  _llk_math_eltwise_sfpu_done_();
}
// A whole-tile chunk all-reduce's combine (libtt_dm_chunk_allreduce_tiles)
// on the sum of a partial tile's gather tiles in t: the cores' slots of slot
// rows folded, rows of equal index mod slot summed onto rows [0, slot)
// (vector adds: 16 rows apart = 16 vectors, 8 = 4, 4 = 2); slot 1: each
// column's sum over all 32 rows, in every row.
sfpi_inline void gather_fold(uint32_t t, uint32_t slot) {
  if (slot == 1) {
    column_reduce(t, t, false, 32);
    return;
  }
  llk_math_eltwise_unary_sfpu_init<SfpuType::unused, DST_ACCUM_MODE>();
  _llk_math_eltwise_sfpu_start_(0);
  auto fold = [&](uint32_t v, uint32_t dv) __attribute__((always_inline)) {
    sfpi::vFloat a = sfpi::dst_reg[t * 32 + v];
    sfpi::vFloat b = sfpi::dst_reg[t * 32 + v + dv];
    sfpi::dst_reg[t * 32 + v] = a + b;
  };
  if (slot <= 16) {
#pragma GCC unroll 16
    for (uint32_t v = 0; v < 16; ++v) {
      fold(v, 16);
    }
  }
  if (slot <= 8) {
#pragma GCC unroll 4
    for (uint32_t v = 0; v < 4; ++v) {
      fold(v, 4);
      fold(8 + v, 4);
    }
  }
  if (slot <= 4) {
#pragma GCC unroll 2
    for (uint32_t v = 0; v < 2; ++v) {
      fold(v, 2);
      fold(8 + v, 2);
    }
  }
  _llk_math_eltwise_sfpu_done_();
}
// The exchanged partial from the gather tiles' sum in t: partial row c (the
// sum of column c of every slot, over the 32 gather rows) in every column of
// row c, rows c < rows. Per column half (one with slots of <= 16 columns:
// the halves fold) and parity: the faces' band vectors summed, the slots
// within a face folded (lane rotations), the band's four rows summed
// (subvectors, a transpose); then row_to_column's spreading on the rows
// that exist.
sfpi_inline void gather_combine(uint32_t t, uint32_t rows) {
  llk_math_eltwise_unary_sfpu_init<SfpuType::unused, DST_ACCUM_MODE>();
  _llk_math_eltwise_sfpu_start_(0);
  const uint32_t slot = gather_slot(rows);
#pragma GCC unroll 2
  for (uint32_t h = 0; h < (slot > 16 ? 2u : 1u); ++h) {
#pragma GCC unroll 2
    for (uint32_t p = 0; p < 2; ++p) {
      sfpi::vFloat acc = sfpi::dst_reg[t * 32 + h * 8 + p];
#pragma GCC unroll 16
      for (uint32_t k = 1; k < (slot > 16 ? 8u : 16u); ++k) {
        // Faces h, h + 2 (and with slots <= 16 columns h + 1, h + 3).
        const uint32_t face = slot > 16 ? 2 * (k / 4) + h : k / 4;
        sfpi::vFloat x = sfpi::dst_reg[t * 32 + face * 8 + 2 * (k % 4) + p];
        acc += x;
      }
      // Slots of `slot` columns, slot / 2 lanes apart within a subvector.
#pragma GCC unroll 3
      for (uint32_t step = 8; step >= 2; step /= 2) {
        if (slot <= step) {
          sfpi::vFloat moved = acc;
#pragma GCC unroll 4
          for (uint32_t r = 0; r < step / 2; ++r) {
            moved = rotate8f(moved);
          }
          acc = acc + moved;
        }
      }
      sfpi::vFloat a0 = acc, a1 = acc, a2 = acc, a3 = acc, a4 = acc,
                   a5 = acc, a6 = acc, a7 = acc;
      sfpi::subvec_transp(a0, a1, a2, a3, a4, a5, a6, a7);
      sfpi::dst_reg[t * 32 + h * 8 + p] = (a0 + a1) + (a2 + a3);
    }
  }
  if (slot == 1) {
    // One column per core: the odd columns too.
    sfpi::vFloat e = sfpi::dst_reg[t * 32];
    sfpi::vFloat o = sfpi::dst_reg[t * 32 + 1];
    sfpi::dst_reg[t * 32] = e + o;
  }
  sfpi::vInt id = sfpi::vConstTileId;
  sfpi::vInt sub = sfpi::as<sfpi::vInt>(sfpi::as<sfpi::vUInt>(id) >> 4) & 3;
  sfpi::vInt lane2 = id & 14;
#pragma GCC unroll 2
  for (int face = 1; face >= 0; --face) {
#pragma GCC unroll 4
    for (int band = 3; band >= 0; --band) {
      if (16 * face + 4 * band < static_cast<int>(rows)) {
        sfpi::vInt v = sfpi::dst_reg[t * 32 + face * 8];
        v_if((sub & 1) != 0) { v = sfpi::dst_reg[t * 32 + face * 8 + 1]; }
        v_endif;
        v_if(lane2 != (sub & 2) + 4 * band) { v = 0; }
        v_endif;
        v = or_lanes8(v);
#pragma GCC unroll 2
        for (uint32_t cf = 0; cf < 2; ++cf) {
          sfpi::dst_reg[t * 32 + (2 * face + cf) * 8 + 2 * band] = v;
          sfpi::dst_reg[t * 32 + (2 * face + cf) * 8 + 2 * band + 1] = v;
        }
      }
    }
  }
  _llk_math_eltwise_sfpu_done_();
}
// sfpu_reduce<SUM, Float32, REDUCE_ROW> on rows < rows (<= 16): each row's
// sum over its 32 columns in column 0 (the even columns of face 0 hold it).
sfpi_inline void row_sum(uint32_t t, uint32_t rows) {
  llk_math_eltwise_unary_sfpu_init<SfpuType::unused, DST_ACCUM_MODE>();
  _llk_math_eltwise_sfpu_start_(0);
#pragma GCC unroll 4
  for (uint32_t b = 0; b < 4; ++b) {
    if (4 * b < rows) {
      sfpi::vFloat a = sfpi::dst_reg[t * 32 + 2 * b];
      sfpi::vFloat c = sfpi::dst_reg[t * 32 + 2 * b + 1];
      sfpi::vFloat d = sfpi::dst_reg[t * 32 + 8 + 2 * b];
      sfpi::vFloat e = sfpi::dst_reg[t * 32 + 8 + 2 * b + 1];
      sfpi::vFloat v = (a + c) + (d + e);
      sfpi::dst_reg[t * 32 + 2 * b] = sum_lanes8f(v);
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
#include "api/compute/compute_kernel_api.h"
ALWI void libtt_group_sum_tile_init() {}
ALWI void libtt_row_to_column_tile_init() {}
ALWI void libtt_row_to_column_tile(uint32_t idst) {
  MATH((libtt_sfpu::row_to_column(idst)));
}
ALWI void libtt_group_sum_tile(uint32_t idst) {
  MATH((libtt_sfpu::group_sum16(idst)));
}
// The chunk all-reduce (d2m_fusion::insertChunkAllReduce): the gather tile in
// DST (row j: core j's row sums, row t's in column t) summed over its rows,
// then row 0 made the columns: row t holds row t's total in every column.
// The end of a DST section that keeps DST for the next one: the packer's
// release with clear_dest off (math may write again once the packer has read
// DST). A kernel kept in DST across a chunk all-reduce's exchange.
ALWI void libtt_tile_regs_release_keep() {
  PACK((_llk_pack_dest_section_done_<DST_SYNC_MODE, DST_ACCUM_MODE, false>()));
}
// DST[out] = each column's sum (max) over DST[in]'s rows [0, n), in every row
// (ttir.column_reduce; libtt_sfpu::column_reduce).
ALWI void libtt_column_reduce_tile_init() {}
ALWI void libtt_column_reduce_tile(uint32_t in, uint32_t out, uint32_t max,
                                   uint32_t n) {
  MATH((libtt_sfpu::column_reduce(in, out, max != 0, n)));
}
ALWI void libtt_place_rows_tile_init() {}
ALWI void libtt_place_rows_tile(uint32_t dst, uint32_t src, uint32_t out,
                                uint32_t to, uint32_t from, uint32_t count) {
  MATH((libtt_sfpu::place_rows(dst, src, out, to, from, count)));
}
)libtt";

// libtt data-movement helpers (kernels that call libtt_*_dm functions).
static const char *const kLibttDataMovement = R"libtt(
// Gather row g's offset in the FP32 gather tiles (columns 0-15; 16-31 are the
// next face, 1024 bytes on).
inline uint32_t libtt_gather_row_offset(uint32_t g) {
  return (g / 32) * 4096 + (g % 32 / 16) * 2048 + (g % 16) * 64;
}
// Columns per core in a gather row: the partial's rows rounded up to a power
// of two (32 / slot cores share a row). Also libtt_sfpu::gather_slot and
// insertChunkAllReduce's tile count.
inline uint32_t libtt_gather_slot(uint32_t rows) {
  uint32_t w = 1;
  while (w < rows) {
    w *= 2;
  }
  return w;
}
// Each NoC-contiguous rectangle of the logical rectangle (x0, y0)-(x1, y1)
// (harvested columns and rows split it): f(multicast address with offset 0,
// number of its cores other than (mx, my)).
template <typename F>
inline void libtt_dm_for_rects(uint32_t x0, uint32_t y0, uint32_t x1,
                               uint32_t y1, uint32_t mx, uint32_t my,
                               uint32_t noc, F f) {
  for (uint32_t xa = x0; xa <= x1;) {
    uint32_t xb = xa;
    while (xb < x1 && worker_logical_col_to_virtual_col[xb + 1] ==
                          worker_logical_col_to_virtual_col[xb] + 1) {
      ++xb;
    }
    for (uint32_t ya = y0; ya <= y1;) {
      uint32_t yb = ya;
      while (yb < y1 && worker_logical_row_to_virtual_row[yb + 1] ==
                            worker_logical_row_to_virtual_row[yb] + 1) {
        ++yb;
      }
      const uint32_t ax = worker_logical_col_to_virtual_col[xa];
      const uint32_t bx = worker_logical_col_to_virtual_col[xb];
      const uint32_t ay = worker_logical_row_to_virtual_row[ya];
      const uint32_t by = worker_logical_row_to_virtual_row[yb];
      const bool self = mx >= xa && mx <= xb && my >= ya && my <= yb;
      const uint32_t n = (xb - xa + 1) * (yb - ya + 1) - (self ? 1 : 0);
      if (n > 0) {
        f(noc == 0 ? get_noc_multicast_addr(ax, ay, bx, by, 0, noc)
                   : get_noc_multicast_addr(bx, by, ax, ay, 0, noc),
          n);
      }
      ya = yb + 1;
    }
    xa = xb + 1;
  }
}

// Program entry. tt-metal pipelines programs across cores: a core done with
// the previous program starts this one while others still run theirs, and the
// exchanges write into other cores' CBs, which a previous program there may
// still use (an Inkling 2 decode step hung on it). So: every core but the
// root bumps the root's entry semaphore when it starts (a semaphore is set up
// before any core starts); before the first exchange's writes the root waits
// for all k - 1 and multicasts the go-ahead, which the others wait for.
inline void libtt_dm_entry_signal(uint32_t sem_id, uint32_t x0, uint32_t y0,
                                  uint32_t x1, uint32_t y1, uint32_t noc) {
  const uint32_t w = x1 - x0 + 1;
  const uint32_t me = ((uint32_t)get_absolute_logical_y() - y0) * w +
                      ((uint32_t)get_absolute_logical_x() - x0);
  if (me != 0) {
    noc_semaphore_inc(
        get_noc_addr(worker_logical_col_to_virtual_col[x0],
                     worker_logical_row_to_virtual_row[y0],
                     get_semaphore(sem_id), noc),
        1, noc);
  }
}
inline void libtt_dm_entry_wait(uint32_t sem_id, uint32_t x0, uint32_t y0,
                                uint32_t x1, uint32_t y1, uint32_t noc) {
  const uint32_t w = x1 - x0 + 1;
  const uint32_t k = w * (y1 - y0 + 1);
  const uint32_t mx = (uint32_t)get_absolute_logical_x();
  const uint32_t my = (uint32_t)get_absolute_logical_y();
  const uint32_t me = (my - y0) * w + (mx - x0);
  const uint32_t sem = get_semaphore(sem_id);
  volatile tt_l1_ptr uint32_t *s =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(sem);
  if (me == 0) {
    noc_semaphore_wait_min(s, k - 1);
    // The go-ahead's source: this semaphore (16-byte aligned, as the
    // destination), set to 1 for the multicast.
    noc_semaphore_set(s, 1);
    libtt_dm_for_rects(x0, y0, x1, y1, mx, my, noc,
                       [&](uint64_t mcast_base, uint32_t n) {
      noc_semaphore_set_multicast(sem, mcast_base | sem, n, false, noc);
    });
    noc_async_write_barrier(noc);
  } else {
    noc_semaphore_wait_min(s, 1);
  }
  noc_semaphore_set(s, 0);
}

// The exchanges' end: every core but the root (x0, y0) bumps the root's
// semaphore after its gather writes (the same NoC and VC, so the bump lands
// after them) and waits for the flag; the root, once all k - 1 arrived,
// multicasts its gather tiles to the others, each NoC-contiguous rectangle
// once, with the flag on the same path (linked; its source is send's first
// word, 16-byte aligned as the semaphore); then every core pushes recv.
inline void libtt_dm_gather_done(CircularBuffer &send, CircularBuffer &recv,
                                 uint32_t sem_id, uint32_t x0, uint32_t y0,
                                 uint32_t x1, uint32_t y1, uint32_t mx,
                                 uint32_t my, uint32_t me, uint32_t k,
                                 uint32_t src, uint32_t base,
                                 uint32_t recv_tiles, uint32_t send_tiles,
                                 uint32_t noc) {
  const uint32_t rx = worker_logical_col_to_virtual_col[x0];
  const uint32_t ry = worker_logical_row_to_virtual_row[y0];
  volatile tt_l1_ptr uint32_t *p =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(src);
  const uint32_t sem = get_semaphore(sem_id);
  volatile tt_l1_ptr uint32_t *s =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(sem);
  if (me != 0) {
    noc_async_writes_flushed(noc);
    noc_semaphore_inc(get_noc_addr(rx, ry, sem, noc), 1, noc);
    noc_async_write_barrier(noc);
    send.pop_front(send_tiles);
    noc_semaphore_wait_min(s, 1);
  } else {
    noc_async_write_barrier(noc);
    // The flag's source: P's first word, done with (16-byte aligned, as the
    // semaphore: the NoC needs equal source and destination alignment).
    p[0] = 1;
    noc_semaphore_wait_min(s, k - 1);
    noc_async_read_barrier(noc);
    libtt_dm_for_rects(x0, y0, x1, y1, mx, my, noc,
                       [&](uint64_t mcast_base, uint32_t n) {
      noc_async_write_multicast(base, mcast_base | base, recv_tiles * 4096, n,
                                true, noc);
      noc_semaphore_set_multicast(src, mcast_base | sem, n, false, noc);
    });
    noc_async_write_barrier(noc);
    send.pop_front(send_tiles);
  }
  noc_semaphore_set(s, 0);
  recv.push_back(recv_tiles);
}

// The chunk all-reduce's exchange (d2m_fusion::insertChunkAllReduce) on the k
// cores of the logical rectangle (x0, y0)-(x1, y1). Each core's row sums
// (column 0 of the FP32 tile in send_cb, rows 0..rows-1) go, as row t's in
// column t of its slot (libtt_gather_slot columns; slot me % (32 / slot) of
// gather row me / (32 / slot)), to recv_cb on the root core (x0, y0), with a
// semaphore bump after it (the same NoC and VC, so the bump lands after the
// row). Once all k - 1 bumps arrived, the root multicasts its gather tiles
// to the others, each NoC-contiguous rectangle once, with a flag on the same
// path (linked); then every core pushes recv_cb. (Every core writing every
// core, unicast, took 72 x 72 transactions; every core multicasting, 72
// multicasts to the same rectangles, serialized: 260 us.) The root zeroes
// the slots past k (NoC reads of the zero region).
inline void libtt_dm_chunk_allreduce(uint32_t send_cb, uint32_t recv_cb,
                                     uint32_t sem_id, uint32_t x0, uint32_t y0,
                                     uint32_t x1, uint32_t y1, uint32_t rows,
                                     uint32_t recv_tiles, uint32_t noc) {
  CircularBuffer send(send_cb), recv(recv_cb);
  const uint32_t w = x1 - x0 + 1;
  const uint32_t k = w * (y1 - y0 + 1);
  const uint32_t mx = (uint32_t)get_absolute_logical_x();
  const uint32_t my = (uint32_t)get_absolute_logical_y();
  const uint32_t me = (my - y0) * w + (mx - x0);
  const uint32_t slot = libtt_gather_slot(rows);
  const uint32_t per_row = 32 / slot;
  const uint32_t used_rows = (k + per_row - 1) / per_row;
  recv.reserve_back(recv_tiles);
  const uint32_t base = recv.get_write_ptr();
  if (me == 0) {
    const uint64_t zeros =
        get_noc_addr(my_x[noc], my_y[noc], MEM_ZEROS_BASE, noc);
    // Bytes [at, at + n) zeroed (the zeros' address congruent mod 16).
    auto zero = [&](uint32_t at, uint32_t n) {
      while (n > 0) {
        const uint32_t m = n < MEM_ZEROS_SIZE - 16 ? n : MEM_ZEROS_SIZE - 16;
        noc_async_read(zeros + (at & 15), at, m, noc);
        at += m;
        n -= m;
      }
    };
    // The free slots of the last used row, then the unused rows.
    for (uint32_t c = (k % per_row) * slot; k % per_row != 0 && c < 32;
         c = (c / 16 + 1) * 16) {
      zero(base + libtt_gather_row_offset(used_rows - 1) + c / 16 * 1024 +
               c % 16 * 4,
           (16 - c % 16) * 4);
    }
    for (uint32_t g0 = used_rows / 16 * 16; g0 < recv_tiles * 32; g0 += 16) {
      const uint32_t from = g0 < used_rows ? used_rows - g0 : 0;
      for (uint32_t h = 0; h < 2 && from < 16; ++h) {
        zero(base + libtt_gather_row_offset(g0 + from) + h * 1024,
             (16 - from) * 64);
      }
    }
  }
  send.wait_front(1);
  const uint32_t src = send.get_read_ptr();
  volatile tt_l1_ptr uint32_t *p =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(src);
  // Row t's sum (column 0) to column t of row 0 (faces 0 and 2 to 0 and 1),
  // at the slot's offset mod 16 bytes (the NoC needs equal alignment).
  // (A transpose in the compute phase instead measured the same: +0.1 us at
  // T = 1, -0.1 us at T = 32.)
  const uint32_t col = me % per_row * slot;
  const uint32_t off = col % 4;
  if (off != 0) {
    const uint32_t v1 = rows > 1 ? p[16] : 0;
    p[off] = p[0];
    p[off + 1] = v1;
  } else {
    for (uint32_t t = 1; t < rows && t < 16; ++t) {
      p[t] = p[t * 16];
    }
    for (uint32_t t = 16; t < rows; ++t) {
      p[256 + t - 16] = p[512 + (t - 16) * 16];
    }
  }
  const uint32_t dst = base + libtt_gather_row_offset(me / per_row) +
                       col / 16 * 1024 + col % 16 * 4;
  const uint32_t rx = worker_logical_col_to_virtual_col[x0];
  const uint32_t ry = worker_logical_row_to_virtual_row[y0];
  for (uint32_t h = 0; h < (slot > 16 ? 2u : 1u); ++h) {
    noc_async_write(src + off * 4 + h * 1024,
                    get_noc_addr(rx, ry, dst + h * 1024, noc),
                    (slot > 16 ? 16 : slot) * 4, noc);
  }
  libtt_dm_gather_done(send, recv, sem_id, x0, y0, x1, y1, mx, my, me, k, src,
                       base, recv_tiles, 1, noc);
}
// Whole tiles (libtt.full_tile): rows [0, rows) of each of the ntiles FP32
// tiles in send_cb (every column) go to recv_cb on the root, as rows
// me % per * slot.. of gather tile n * g + me / per (slot:
// libtt_full_slot(rows); per = 32 / slot cores per gather tile, g gather
// tiles per partial tile); the root zeroes each last gather tile's rows past
// the k-th slot; then as libtt_dm_chunk_allreduce. The compute adds a
// partial tile's g gather tiles and folds their rows (libtt_sfpu::gather_fold).
inline uint32_t libtt_full_slot(uint32_t rows) {
  return rows == 1 ? 1 : rows <= 4 ? 4 : libtt_gather_slot(rows);
}
inline void libtt_dm_chunk_allreduce_tiles(uint32_t send_cb, uint32_t recv_cb,
                                           uint32_t sem_id, uint32_t x0,
                                           uint32_t y0, uint32_t x1,
                                           uint32_t y1, uint32_t rows,
                                           uint32_t ntiles, uint32_t g,
                                           uint32_t noc) {
  CircularBuffer send(send_cb), recv(recv_cb);
  const uint32_t w = x1 - x0 + 1;
  const uint32_t k = w * (y1 - y0 + 1);
  const uint32_t mx = (uint32_t)get_absolute_logical_x();
  const uint32_t my = (uint32_t)get_absolute_logical_y();
  const uint32_t me = (my - y0) * w + (mx - x0);
  const uint32_t slot = libtt_full_slot(rows);
  const uint32_t per = 32 / slot;
  recv.reserve_back(ntiles * g);
  const uint32_t base = recv.get_write_ptr();
  if (me == 0) {
    const uint64_t zeros =
        get_noc_addr(my_x[noc], my_y[noc], MEM_ZEROS_BASE, noc);
    auto zero = [&](uint32_t at, uint32_t n) {
      while (n > 0) {
        const uint32_t m = n < MEM_ZEROS_SIZE - 16 ? n : MEM_ZEROS_SIZE - 16;
        noc_async_read(zeros + (at & 15), at, m, noc);
        at += m;
        n -= m;
      }
    };
    const uint32_t used = (k - (g - 1) * per) * slot;
    for (uint32_t n = 0; n < ntiles; ++n) {
      const uint32_t tile = base + (n * g + g - 1) * 4096;
      for (uint32_t r = used; r < 32; r = (r / 16 + 1) * 16) {
        for (uint32_t h = 0; h < 2; ++h) {
          zero(tile + r / 16 * 2048 + h * 1024 + r % 16 * 64,
               ((r / 16 + 1) * 16 - r) * 64);
        }
      }
    }
  }
  send.wait_front(ntiles);
  const uint32_t src = send.get_read_ptr();
  const uint32_t rx = worker_logical_col_to_virtual_col[x0];
  const uint32_t ry = worker_logical_row_to_virtual_row[y0];
  for (uint32_t n = 0; n < ntiles; ++n) {
    const uint32_t tile = base + (n * g + me / per) * 4096;
    // Rows [r, e) of one 16-row band (faces 0-1 or 2-3: two writes).
    for (uint32_t r = 0; r < rows;) {
      const uint32_t e = rows < (r / 16 + 1) * 16 ? rows : (r / 16 + 1) * 16;
      const uint32_t d = me % per * slot + r;
      for (uint32_t h = 0; h < 2; ++h) {
        noc_async_write(src + n * 4096 + r / 16 * 2048 + h * 1024 + r % 16 * 64,
                        get_noc_addr(rx, ry,
                                     tile + d / 16 * 2048 + h * 1024 +
                                         d % 16 * 64,
                                     noc),
                        (e - r) * 64, noc);
      }
      r = e;
    }
  }
  libtt_dm_gather_done(send, recv, sem_id, x0, y0, x1, y1, mx, my, me, k, src,
                       base, ntiles * g, ntiles, noc);
}

// The whole-tile exchange in two levels, for partials of several rows (as
// libtt_dm_chunk_allreduce_tiles): row r of partial tile n is reduced on
// core q = n * rows + r. Every core writes that row (two 64-byte face-row
// halves) to q's inbox, tile me / 32, row me % 32, and bumps q's semaphore
// a; q, once all k - 1 others did, pushes its inbox (the rows past k zeroed)
// and the compute sums it to red (every row: the column sums). Core q writes
// red's row 0 to row r of tile n of the root's recv, then libtt_dm_gather_done
// on semaphore b: the root multicasts only the summed tiles (recv rows past
// rows zeroed).
inline void libtt_dm_chunk_allreduce_rows(uint32_t send_cb, uint32_t recv_cb,
                                          uint32_t inbox_cb, uint32_t red_cb,
                                          uint32_t sem_a, uint32_t sem_b,
                                          uint32_t x0, uint32_t y0,
                                          uint32_t x1, uint32_t y1,
                                          uint32_t rows, uint32_t ntiles,
                                          uint32_t noc) {
  CircularBuffer send(send_cb), recv(recv_cb), inbox(inbox_cb), red(red_cb);
  const uint32_t w = x1 - x0 + 1;
  const uint32_t k = w * (y1 - y0 + 1);
  const uint32_t mx = (uint32_t)get_absolute_logical_x();
  const uint32_t my = (uint32_t)get_absolute_logical_y();
  const uint32_t me = (my - y0) * w + (mx - x0);
  const uint32_t g = (k + 31) / 32;
  const uint32_t reducers = rows * ntiles;
  inbox.reserve_back(g);
  recv.reserve_back(ntiles);
  const uint32_t ibase = inbox.get_write_ptr();
  const uint32_t base = recv.get_write_ptr();
  const uint64_t zeros =
      get_noc_addr(my_x[noc], my_y[noc], MEM_ZEROS_BASE, noc);
  auto zero = [&](uint32_t at, uint32_t n) {
    while (n > 0) {
      const uint32_t m = n < MEM_ZEROS_SIZE - 16 ? n : MEM_ZEROS_SIZE - 16;
      noc_async_read(zeros + (at & 15), at, m, noc);
      at += m;
      n -= m;
    }
  };
  // Rows [from, 32) of a tile (both 16-column halves of each 16-row band).
  auto zeroRows = [&](uint32_t tile, uint32_t from) {
    for (uint32_t r = from; r < 32; r = (r / 16 + 1) * 16) {
      for (uint32_t h = 0; h < 2; ++h) {
        zero(tile + r / 16 * 2048 + h * 1024 + r % 16 * 64,
             ((r / 16 + 1) * 16 - r) * 64);
      }
    }
  };
  if (me < reducers) {
    zeroRows(ibase + (g - 1) * 4096, k - (g - 1) * 32);
  }
  if (me == 0) {
    for (uint32_t n = 0; n < ntiles; ++n) {
      zeroRows(base + n * 4096, rows);
    }
  }
  auto row = [](uint32_t r, uint32_t h) {
    return r / 16 * 2048 + h * 1024 + r % 16 * 64;
  };
  send.wait_front(ntiles);
  const uint32_t src = send.get_read_ptr();
  const uint32_t sa = get_semaphore(sem_a);
  const uint32_t at = ibase + me / 32 * 4096;
  for (uint32_t q = 0; q < reducers; ++q) {
    const uint32_t qx = worker_logical_col_to_virtual_col[x0 + q % w];
    const uint32_t qy = worker_logical_row_to_virtual_row[y0 + q / w];
    const uint32_t n = q / rows, r = q % rows;
    for (uint32_t h = 0; h < 2; ++h) {
      noc_async_write(src + n * 4096 + row(r, h),
                      get_noc_addr(qx, qy, at + row(me % 32, h), noc), 64,
                      noc);
    }
  }
  noc_async_writes_flushed(noc);
  for (uint32_t q = 0; q < reducers; ++q) {
    if (q != me) {
      const uint32_t qx = worker_logical_col_to_virtual_col[x0 + q % w];
      const uint32_t qy = worker_logical_row_to_virtual_row[y0 + q / w];
      noc_semaphore_inc(get_noc_addr(qx, qy, sa, noc), 1, noc);
    }
  }
  noc_async_write_barrier(noc);
  send.pop_front(ntiles);
  if (me < reducers) {
    volatile tt_l1_ptr uint32_t *s =
        reinterpret_cast<volatile tt_l1_ptr uint32_t *>(sa);
    noc_semaphore_wait_min(s, k - 1);
    noc_semaphore_set(s, 0);
    noc_async_read_barrier(noc);
  }
  inbox.push_back(g);
  red.wait_front(1);
  const uint32_t sum = red.get_read_ptr();
  if (me < reducers) {
    const uint32_t rx = worker_logical_col_to_virtual_col[x0];
    const uint32_t ry = worker_logical_row_to_virtual_row[y0];
    const uint32_t n = me / rows, r = me % rows;
    for (uint32_t h = 0; h < 2; ++h) {
      noc_async_write(sum + row(0, h),
                      get_noc_addr(rx, ry, base + n * 4096 + row(r, h), noc),
                      64, noc);
    }
  }
  libtt_dm_gather_done(red, recv, sem_b, x0, y0, x1, y1, mx, my, me, k, sum,
                       base, ntiles, 1, noc);
}
)libtt";

} // namespace

namespace mlir::tt::d2m_fusion {

llvm::StringRef sfpuChainHelpers() { return kLibttSfpuHelpers; }
llvm::StringRef binaryCompatHelpers() { return kLibttBinaryCompat; }
llvm::StringRef tileOpHelpers() { return kLibttTileOps; }
llvm::StringRef dataMovementHelpers() { return kLibttDataMovement; }

} // namespace mlir::tt::d2m_fusion
