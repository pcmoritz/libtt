// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
//
// Per-row float8 quantization on the SFPU in FP32 DST, shared by
// quantize_rows and rms_norm_residual's float8 output.
//
// Row maxima: band g (rows 4g .. 4g + 3) of a tile holds rows
// (4g .. 4g + 3) % stride, stride 4, 8, 16 or 32, so band q = g % (stride / 4)
// of an accumulator tile keeps a running maximum of |x| over the bands.
// Maxima compare the bits of |x| as integers: that is the order of
// non-negative floats, with NaN above infinity, so a NaN in a row makes its
// maximum NaN as in PyTorch (for FP32 inputs: the unpacker widens a BF16 NaN
// to infinity). Each accumulator row's 8 lanes then take their maximum, and
// the tile goes to the other cores with row m's maximum in columns 0-15 of
// row m.
//
// The divisor: a gather buffer holds every core's rows, `stride / 4` bands
// per core, so band k of a gather tile is accumulator band k % (stride / 4).
// The scales, amax / fp8_max by ttnn.divide's FP32 divide (div_binary_tile's
// routine, on the stride / 4 distinct vectors), floored at min_scale, and NaN
// for a NaN maximum as in PyTorch, spread across their rows (all 32 columns)
// make the divisor tile: every row holds its scale in every column. Tiles are
// then divided by it and converted to float8 with ttnn.typecast's exact
// conversion, so that the codes equal ttnn.divide and ttnn.typecast's.
//
// SFPU lane layout (fp32 DST): vector d (0..7) of a face holds, in lane l
// (0..31), the face's element at row 4 * (d / 2) + l / 8, column
// 2 * (l % 8) + d % 2.
#pragma once

#include <cstdint>
#include <type_traits>
#include <utility>

#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/compute/float8_sfpu.h"

#ifdef TRISC_MATH
#include "sfpi.h"

namespace qr {

template <typename F, uint32_t... I>
sfpi_inline void static_for_impl(F&& f, std::integer_sequence<uint32_t, I...>) {
    (f(std::integral_constant<uint32_t, I>{}), ...);
}

// f(0) .. f(N - 1), with compile-time arguments.
template <uint32_t N, typename F>
sfpi_inline void static_for(F&& f) {
    static_for_impl(f, std::make_integer_sequence<uint32_t, N>{});
}

// SFPU vector `parity` (even or odd columns) of band b (rows 4b .. 4b + 3) of
// face f of DST tile t.
constexpr uint32_t vec(uint32_t t, uint32_t f, uint32_t b, uint32_t parity) { return t * 32 + f * 8 + b * 2 + parity; }

// Accumulator band q of tile t: the even columns of the left face's band.
constexpr uint32_t acc_vec(uint32_t t, uint32_t q) { return vec(t, q < 4 ? 0 : 2, q % 4, 0); }

sfpi_inline sfpi::vInt load_bits(uint32_t i) { return sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[i])); }
sfpi_inline void store_bits(uint32_t i, sfpi::vInt v) { sfpi::dst_reg[i] = sfpi::as<sfpi::vFloat>(v); }

// |x| as integer bits.
sfpi_inline sfpi::vInt abs_bits(uint32_t i) {
    return sfpi::as<sfpi::vInt>(sfpi::setsgn(sfpi::vFloat(sfpi::dst_reg[i]), 0));
}

// The larger of two non-negative (or NaN) float bit patterns.
sfpi_inline sfpi::vInt imax(sfpi::vInt a, sfpi::vInt b) {
    v_if(b > a) { a = b; }
    v_endif;
    return a;
}

sfpi_inline sfpi::vInt ror1(sfpi::vInt v) { return sfpi::vInt(sfpi::subvec_shflror1(v)); }

// Every lane of each 8-lane row gets the row's maximum.
sfpi_inline sfpi::vInt lanes8_max(sfpi::vInt v) {
    v = imax(v, ror1(v));
    v = imax(v, ror1(ror1(v)));
    return imax(v, ror1(ror1(ror1(ror1(v)))));
}

// Zeroes the accumulator bands of tile `acc`.
template <uint32_t groups, uint32_t acc>
inline void zero_acc() {
    _llk_math_eltwise_sfpu_start_(0);
    static_for<groups>([](auto q) { store_bits(acc_vec(acc, decltype(q)::value), sfpi::vInt(0)); });
    _llk_math_eltwise_sfpu_done_();
}

// The first `bands` bands of dense tile x into the accumulators of tile acc.
template <uint32_t groups, uint32_t x, uint32_t acc>
inline void fold_absmax(uint32_t bands) {
    _llk_math_eltwise_sfpu_start_(0);
    static_for<8>([&](auto gc) {
        constexpr uint32_t g = decltype(gc)::value;
        if (g < bands) {
            constexpr uint32_t f = g < 4 ? 0 : 2, b = g % 4;
            sfpi::vInt m = imax(abs_bits(vec(x, f, b, 0)), abs_bits(vec(x, f, b, 1)));
            m = imax(m, imax(abs_bits(vec(x, f + 1, b, 0)), abs_bits(vec(x, f + 1, b, 1))));
            constexpr uint32_t a = acc_vec(acc, g % groups);
            store_bits(a, imax(load_bits(a), m));
        }
    });
    _llk_math_eltwise_sfpu_done_();
}

// Row m's maximum into columns 0-15 of row m.
template <uint32_t groups, uint32_t acc>
inline void finish_partial() {
    _llk_math_eltwise_sfpu_start_(0);
    static_for<groups>([](auto qc) {
        constexpr uint32_t a = acc_vec(acc, decltype(qc)::value);
        const sfpi::vInt m = lanes8_max(load_bits(a));
        store_bits(a, m);
        store_bits(a + 1, m);
    });
    _llk_math_eltwise_sfpu_done_();
}

// The first `valid` bands of gather tile `src` into the accumulators of tile acc.
template <uint32_t groups, uint32_t src, uint32_t acc>
inline void fold_gather(uint32_t valid) {
    _llk_math_eltwise_sfpu_start_(0);
    static_for<16>([&](auto kc) {
        constexpr uint32_t k = decltype(kc)::value;
        if (k < valid) {
            constexpr uint32_t a = acc_vec(acc, k % groups);
            store_bits(a, imax(load_bits(a), load_bits(vec(src, k / 4, k % 4, 0))));
        }
    });
    _llk_math_eltwise_sfpu_done_();
}

// The row maxima to divide by fp8_max, as the first `groups` vectors of tile
// t (accumulator band q of tile acc as vector q), and fp8_max as those of tile
// c: the divide then runs on them alone, not on a whole tile.
template <uint32_t groups, uint32_t acc, uint32_t t, uint32_t c>
inline void stage_scale(uint32_t fp8_max_bits) {
    _llk_math_eltwise_sfpu_start_(0);
    static_for<groups>([&](auto qc) {
        constexpr uint32_t q = decltype(qc)::value;
        store_bits(t * 32 + q, load_bits(acc_vec(acc, q)));
        store_bits(c * 32 + q, sfpi::vInt(fp8_max_bits));
    });
    _llk_math_eltwise_sfpu_done_();
}

// Vector q of tile c = vector q of tile t / vector q of tile c, by
// ttnn.divide's FP32 divide (div_binary_tile's routine on `groups` vectors).
template <uint32_t groups, uint32_t t, uint32_t c>
inline void divide_staged() {
    _llk_math_eltwise_sfpu_start_(0);
    ckernel::sfpu::calculate_sfpu_binary_div<APPROX, ckernel::BinaryOp::DIV, groups, DST_ACCUM_MODE>(t, c, c);
    _llk_math_eltwise_sfpu_done_();
}

// The divisor tile d: every band g gets, in all four vectors, the scale of
// staged vector g % groups: the quotient amax / fp8_max (tile c) raised to at
// least the float with bits `lo`, or NaN where amax (tile t) is NaN, which the
// SFPU's divide makes infinite.
// All four vectors of band g of tile d set to v. (SFPU vectors cannot be
// spilled, so they pass by value, never captured by reference.)
template <uint32_t d, uint32_t g>
sfpi_inline void store_band(sfpi::vInt v) {
    constexpr uint32_t f = g < 4 ? 0 : 2, b = g % 4;
    store_bits(vec(d, f, b, 0), v);
    store_bits(vec(d, f, b, 1), v);
    store_bits(vec(d, f + 1, b, 0), v);
    store_bits(vec(d, f + 1, b, 1), v);
}

template <uint32_t groups, uint32_t q, uint32_t d, uint32_t... J>
sfpi_inline void store_bands(sfpi::vInt v, std::integer_sequence<uint32_t, J...>) {
    (store_band<d, J * groups + q>(v), ...);
}

template <uint32_t groups, uint32_t t, uint32_t c, uint32_t d>
inline void spread_scale(uint32_t lo_bits) {
    _llk_math_eltwise_sfpu_start_(0);
    static_for<groups>([&](auto qc) {
        constexpr uint32_t q = decltype(qc)::value;
        sfpi::vInt scale = imax(load_bits(c * 32 + q), sfpi::vInt(lo_bits));
        v_if(load_bits(t * 32 + q) > 0x7f800000) { scale = 0x7fc00000; }
        v_endif;
        store_bands<groups, q, d>(scale, std::make_integer_sequence<uint32_t, 8 / groups>{});
    });
    _llk_math_eltwise_sfpu_done_();
}

// Raw bits: a float load would flush the float8 codes, small integers, as
// subnormals.
sfpi_inline sfpi::vUInt load_raw(uint32_t i) { return sfpi::dst_reg[i].mode<sfpi::DataLayout::U32>(); }
sfpi_inline void store_raw(uint32_t i, sfpi::vUInt v) { sfpi::dst_reg[i].mode<sfpi::DataLayout::U32>() = v; }

// The first `real` rows of bands 0 .. bands - 1 of tile t, each row in one
// vector: a band's four vectors (the even and odd columns of its left and
// right faces) transposed by subvector, so that vector i holds row 4b + i in
// a column order the reverse transpose undoes. Row i of band b goes to vector
// b * real + i of the tile; each band's vectors are read before a later band
// writes over them (real <= 2, or one band).

// SFPI keeps only the eight-register subvec_transp for integers; transposing four registers pads the
// other four with zeros, as its own four-register vFloat form does.
inline void subvec_transp4(sfpi::vUInt& v0, sfpi::vUInt& v1, sfpi::vUInt& v2, sfpi::vUInt& v3) {
    sfpi::vUInt v4 = 0, v5 = 0, v6 = 0, v7 = 0;
    sfpi::subvec_transp(v0, v1, v2, v3, v4, v5, v6, v7);
}

template <uint32_t t, uint32_t bands, uint32_t real>
inline void gather_rows() {
    static_assert(real <= 2 || bands == 1, "the rows would overwrite bands not yet read");
    _llk_math_eltwise_sfpu_start_(0);
    static_for<bands>([](auto bc) {
        constexpr uint32_t b = decltype(bc)::value;
        constexpr uint32_t f = b < 4 ? 0 : 2;
        sfpi::vUInt v0 = load_raw(vec(t, f, b % 4, 0));
        sfpi::vUInt v1 = load_raw(vec(t, f, b % 4, 1));
        sfpi::vUInt v2 = load_raw(vec(t, f + 1, b % 4, 0));
        sfpi::vUInt v3 = load_raw(vec(t, f + 1, b % 4, 1));
        subvec_transp4(v0, v1, v2, v3);
        store_raw(t * 32 + b * real, v0);
        if constexpr (real > 1) {
            store_raw(t * 32 + b * real + 1, v1);
        }
        if constexpr (real > 2) {
            store_raw(t * 32 + b * real + 2, v2);
        }
    });
    _llk_math_eltwise_sfpu_done_();
}

// The reverse of gather_rows, last band first; the bands' other rows are left
// undefined.
template <uint32_t t, uint32_t bands, uint32_t real>
inline void scatter_rows() {
    _llk_math_eltwise_sfpu_start_(0);
    static_for<bands>([](auto bc) {
        constexpr uint32_t b = bands - 1 - decltype(bc)::value;
        constexpr uint32_t f = b < 4 ? 0 : 2;
        sfpi::vUInt v0 = load_raw(t * 32 + b * real);
        sfpi::vUInt v1 = v0;
        sfpi::vUInt v2 = v0;
        sfpi::vUInt v3 = v0;
        if constexpr (real > 1) {
            v1 = load_raw(t * 32 + b * real + 1);
        }
        if constexpr (real > 2) {
            v2 = load_raw(t * 32 + b * real + 2);
        }
        subvec_transp4(v0, v1, v2, v3);
        store_raw(vec(t, f, b % 4, 0), v0);
        store_raw(vec(t, f, b % 4, 1), v1);
        store_raw(vec(t, f + 1, b % 4, 0), v2);
        store_raw(vec(t, f + 1, b % 4, 1), v3);
    });
    _llk_math_eltwise_sfpu_done_();
}

// quantize_tiles' divide of vectors 0 .. n - 1 of tile x by those of tile d.
template <uint32_t d, uint32_t x, uint32_t n>
inline void divide_vectors() {
    _llk_math_eltwise_sfpu_start_(0);
    ckernel::sfpu::calculate_sfpu_binary_div<APPROX, ckernel::BinaryOp::DIV, n, DST_ACCUM_MODE>(x, d, x);
    _llk_math_eltwise_sfpu_done_();
}

// quantize_tiles' float8 conversion of vectors 0 .. n - 1 of tile x.
template <bool e5m2, uint32_t x, uint32_t n>
inline void convert_vectors() {
    _llk_math_eltwise_sfpu_start_(x);
    ckernel::sfpu::calculate_fp32_to_fp8<e5m2 ? 2 : 3, e5m2 ? 15 : 7, e5m2, n>();
    _llk_math_eltwise_sfpu_done_();
}

}  // namespace qr
#endif

namespace qr {

inline void sfpu_init() { MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::unused, DST_ACCUM_MODE>())); }

// The divisor tile d from gather tiles 0 .. gather_tiles - 1 of `gather`,
// whose first `total_bands` bands hold maxima; DST tiles t and c are scratch.
// The caller sets the unpacker's source format for `gather` first.
template <uint32_t groups, uint32_t d, uint32_t t, uint32_t c>
inline void make_divisor(
    uint32_t gather, uint32_t gather_tiles, uint32_t total_bands, uint32_t fp8_max_bits, uint32_t min_scale_bits) {
    copy_init(gather);
    sfpu_init();
    MATH((zero_acc<groups, d>()));
    for (uint32_t i = 0; i < gather_tiles; ++i) {
        copy_tile(gather, i, t);
        const uint32_t valid = total_bands - 16 * i < 16 ? total_bands - 16 * i : 16;
        MATH((fold_gather<groups, t, d>(valid)));
    }
    MATH((stage_scale<groups, d, t, c>(fp8_max_bits)));
    div_binary_tile_init();
    MATH((divide_staged<groups, t, c>()));
    sfpu_init();
    MATH((spread_scale<groups, t, c, d>(min_scale_bits)));
}

// Tiles first .. first + n - 1 of `cb` (at most three) into DST tiles x0 ..
// x0 + n - 1, divided by the divisor tile d and converted to float8 codes, in
// their first `rows` rows (4, 8, 16 or 32: two SFPU vectors per four rows of
// a face, in the top faces for up to 16 rows); div_binary_tile and
// fp8_typecast_tile's routines. With a tensor of fewer than four rows, only
// its rows of every four hold data: when gather_rows can gather them (one or
// two rows, or up to three in a 4-row tile), just those rows are divided and
// converted, in whole vectors, which cuts the SFPU work up to fourfold; the
// other rows' codes are undefined, and so are the divisor's other rows
// afterwards. The caller sets the unpacker's source format for `cb` first.
template <bool e5m2, uint32_t d, uint32_t x0, uint32_t rows, uint32_t tensor_rows>
inline void quantize_tiles(uint32_t cb, uint32_t first, uint32_t n) {
    copy_init(cb);
    for (uint32_t j = 0; j < n; ++j) {
        copy_tile(cb, first + j, x0 + j);
    }
    constexpr uint32_t bands = rows / 4;
    constexpr uint32_t real = tensor_rows < 4 ? tensor_rows : 4;
    if constexpr (real < 4 && (real <= 2 || bands == 1)) {
        constexpr uint32_t vectors = bands * real;
        sfpu_init();
        MATH((gather_rows<d, bands, real>()));
        MATH((static_for<3>([&](auto i) {
            if (i < n) {
                gather_rows<x0 + decltype(i)::value, bands, real>();
            }
        })));
        div_binary_tile_init();
        MATH((static_for<3>([&](auto i) {
            if (i < n) {
                divide_vectors<d, x0 + decltype(i)::value, vectors>();
            }
        })));
        MATH((SFPU_UNARY_INIT(typecast)));
        MATH((static_for<3>([&](auto i) {
            if (i < n) {
                convert_vectors<e5m2, x0 + decltype(i)::value, vectors>();
            }
        })));
        // The divisor's data rows back in place, for a caller that packs it.
        sfpu_init();
        MATH((scatter_rows<d, bands, real>()));
        MATH((static_for<3>([&](auto i) {
            if (i < n) {
                scatter_rows<x0 + decltype(i)::value, bands, real>();
            }
        })));
    } else {
        constexpr int iterations = rows <= 16 ? rows / 2 : 8;
        constexpr VectorMode mode = rows <= 16 ? VectorMode::R : VectorMode::RC;
        div_binary_tile_init();
        for (uint32_t j = 0; j < n; ++j) {
            MATH((SFPU_BINARY_CALL(
                DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sfpu_binary_div,
                (APPROX, ckernel::BinaryOp::DIV, iterations, DST_ACCUM_MODE), x0 + j, d, x0 + j, mode)));
        }
        MATH((SFPU_UNARY_INIT(typecast)));
        for (uint32_t j = 0; j < n; ++j) {
            MATH((SFPU_UNARY_CALL(
                DST_SYNC_MODE, DST_ACCUM_MODE, calculate_fp32_to_fp8,
                (e5m2 ? 2 : 3, e5m2 ? 15 : 7, e5m2, iterations), x0 + j, mode)));
        }
    }
}

}  // namespace qr
