// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
//
// Everything happens on the SFPU in FP32 DST.
//
// Phase 1, this core's row maxima: dense band g (rows 4g .. 4g + 3) of a dense
// tile holds rows (4g .. 4g + 3) % stride of one input tile, so band q =
// g % (stride / 4) of an accumulator tile keeps a running maximum of |x| over
// the bands. Maxima compare the bits of |x| as integers: that is the order of
// non-negative floats, with NaN above infinity, so a NaN in a row makes its
// maximum NaN as in PyTorch (for FP32 inputs: the unpacker widens a BF16 NaN
// to infinity). Each accumulator row's 8 lanes then take their
// maximum, and the tile goes to the reader with row m's maximum in columns
// 0-15 of row m.
//
// Phase 2: the gather buffer holds every core's rows, `stride / 4` bands per
// core, so band k of a gather tile is accumulator band k % (stride / 4). The
// scales, amax / fp8_max by ttnn.divide's FP32 divide (div_binary_tile's
// routine, on the stride / 4 distinct vectors), floored at min_scale, and NaN
// for a NaN maximum as in PyTorch, spread across their dense rows (all 32
// columns) make the divisor tile. The dense tiles are
// then divided by it and converted to float8 with ttnn.typecast's exact
// conversion, chunk_tiles at a time; DST holds the divisor and three dense
// tiles, and the divisor is made again for every chunk. The first core also
// packs the divisor tile as the scale: row m holds scale[m] in every column.
//
// SFPU lane layout (fp32 DST): vector d (0..7) of a face holds, in lane l
// (0..31), the face's element at row 4 * (d / 2) + l / 8, column
// 2 * (l % 8) + d % 2.
#include <cstdint>
#include <type_traits>
#include <utility>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
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

}  // namespace qr
#endif

namespace {
// Phase 1: the accumulator and a dense tile. Phase 2: the divisor, then a
// gather tile and the fp8_max tile while making it, then up to three dense
// tiles.
constexpr uint32_t dst_acc = 0, dst_x = 1;
constexpr uint32_t dst_d = 0, dst_t = 1, dst_c = 2, dst_x0 = 1;

inline void custom_sfpu_init() { MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::unused, DST_ACCUM_MODE>())); }
}  // namespace

void kernel_main() {
    const uint32_t num_dense = get_arg(args::num_dense);
    const uint32_t last_count = get_arg(args::last_count);
    const uint32_t writes_scale = get_arg(args::writes_scale);
    constexpr uint32_t stride = get_arg(args::stride);
    constexpr uint32_t pack = get_arg(args::pack);
    constexpr uint32_t cores = get_arg(args::cores);
    constexpr uint32_t slot_bands = get_arg(args::slot_bands);
    constexpr uint32_t gather_tiles = get_arg(args::gather_tiles);
    constexpr uint32_t chunk_tiles = get_arg(args::chunk_tiles);
    constexpr uint32_t e5m2 = get_arg(args::e5m2);
    constexpr uint32_t fp8_max_bits = get_arg(args::fp8_max_bits);
    constexpr uint32_t min_scale_bits = get_arg(args::min_scale_bits);
    constexpr uint32_t groups = stride / 4;
    static_assert(slot_bands == groups, "a core's slot holds its stride / 4 bands");
    static_assert(chunk_tiles <= 3, "DST holds the divisor and three dense tiles");
    constexpr uint32_t total_bands = cores * slot_bands;

    DataflowBuffer in_dfb(dfb::in);
    DataflowBuffer part_dfb(dfb::part);
    DataflowBuffer gather_dfb(dfb::gather);
    DataflowBuffer out_dfb(dfb::out);
    DataflowBuffer scale_dfb(dfb::scale);

    compute_kernel_hw_startup(dfb::in, dfb::in, dfb::part);
    copy_init(dfb::in);
    in_dfb.wait_front(num_dense);

    // Phase 1: this core's row maxima, for the reader to share.
    tile_regs_acquire();
    custom_sfpu_init();
    MATH((qr::zero_acc<groups, dst_acc>()));
    for (uint32_t i = 0; i < num_dense; ++i) {
        copy_tile(dfb::in, i, dst_x);
        const uint32_t bands = (i + 1 == num_dense ? last_count : pack) * stride / 4;
        MATH((qr::fold_absmax<groups, dst_x, dst_acc>(bands)));
    }
    MATH((qr::finish_partial<groups, dst_acc>()));
    tile_regs_commit();
    part_dfb.reserve_back(1);
    tile_regs_wait();
    pack_tile(dst_acc, dfb::part);
    tile_regs_release();
    part_dfb.push_back(1);

    // Phase 2: the divisor from every core's maxima, then the quotients.
    gather_dfb.wait_front(gather_tiles);
    for (uint32_t first = 0; first < num_dense; first += chunk_tiles) {
        const uint32_t n = num_dense - first < chunk_tiles ? num_dense - first : chunk_tiles;
        tile_regs_acquire();
        reconfig_data_format_srca(dfb::in, dfb::gather);
        copy_init(dfb::gather);
        custom_sfpu_init();
        MATH((qr::zero_acc<groups, dst_d>()));
        for (uint32_t t = 0; t < gather_tiles; ++t) {
            copy_tile(dfb::gather, t, dst_t);
            const uint32_t valid = total_bands - 16 * t < 16 ? total_bands - 16 * t : 16;
            MATH((qr::fold_gather<groups, dst_t, dst_d>(valid)));
        }
        MATH((qr::stage_scale<groups, dst_d, dst_t, dst_c>(fp8_max_bits)));
        div_binary_tile_init();
        MATH((qr::divide_staged<groups, dst_t, dst_c>()));
        custom_sfpu_init();
        MATH((qr::spread_scale<groups, dst_t, dst_c, dst_d>(min_scale_bits)));

        reconfig_data_format_srca(dfb::gather, dfb::in);
        copy_init(dfb::in);
        for (uint32_t j = 0; j < n; ++j) {
            copy_tile(dfb::in, first + j, dst_x0 + j);
        }
        div_binary_tile_init();
        for (uint32_t j = 0; j < n; ++j) {
            div_binary_tile(dst_x0 + j, dst_d, dst_x0 + j);
        }
        MATH((SFPU_UNARY_INIT(typecast)));
        for (uint32_t j = 0; j < n; ++j) {
            MATH((fp8_typecast_tile<false, e5m2 ? 2 : 3, e5m2 ? 15 : 7, e5m2 != 0>(dst_x0 + j)));
        }
        tile_regs_commit();

        tile_regs_wait();
        if (first == 0 && writes_scale) {
            scale_dfb.reserve_back(1);
            pack_reconfig_data_format(dfb::scale);
            pack_tile(dst_d, dfb::scale);
            scale_dfb.push_back(1);
        }
        out_dfb.reserve_back(n);
        pack_reconfig_data_format(dfb::out);
        for (uint32_t j = 0; j < n; ++j) {
            pack_tile(dst_x0 + j, dfb::out);
        }
        out_dfb.push_back(n);
        tile_regs_release();
    }
    gather_dfb.pop_front(gather_tiles);
    in_dfb.pop_front(num_dense);
}
