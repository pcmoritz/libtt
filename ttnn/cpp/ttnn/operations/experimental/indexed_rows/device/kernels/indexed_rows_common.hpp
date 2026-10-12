// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
//
// Shared by the indexed-rows kernels (rows_gather and rows_scatter): index
// decoding, row addressing and the element-wise reductions of a scatter.
//
// A table is a row-major tensor [B, N, S, R]: B batches of N rows, each row
// S consecutive pages of R elements (S is 1 unless the operand had trailing
// dimensions, which are kept as pages so the table never changes page size).
// An index tensor [Bi, M, K] holds, per batch and position, K int32
// coordinates. The first KR address a dimension of extent sizes[k] with a
// window of windows[k] consecutive positions; the dimensions are laid out
// row-major in the N rows (strides[k] rows apart), so a position names
// W = prod(row windows) rows: row(j) = sum_k (c_k + o_k(j)) * strides[k].
// With a column component (K == KR + 1), the last coordinate picks a window of
// windows[KR] elements inside each row's page instead of the whole row.

#pragma once

#include <array>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"

namespace indexed_rows {

// The K coordinates of one position, as the kernels read them from L1.
template <uint32_t K>
struct Coords {
    std::array<int32_t, K> c;
    bool out_of_bounds;
};

// Decodes the K coordinates at `ptr`. Each coordinate is clamped into
// [0, size - window], as StableHLO's gather does; out_of_bounds reports whether
// any coordinate needed clamping (a scatter drops such positions, a gather in
// fill mode writes zeros for them). Unsigned indices compare as unsigned, so
// values past INT32_MAX clamp high instead of reading as negative.
template <uint32_t K, bool kUnsigned>
FORCE_INLINE Coords<K> decode(
    volatile tt_l1_ptr const uint32_t* ptr, const std::array<uint32_t, K>& sizes, const std::array<uint32_t, K>& windows) {
    Coords<K> out;
    out.out_of_bounds = false;
    for (uint32_t k = 0; k < K; ++k) {
        const uint32_t raw = ptr[k];
        const int32_t hi = static_cast<int32_t>(sizes[k] - windows[k]);
        int32_t c;
        if constexpr (kUnsigned) {
            c = raw > static_cast<uint32_t>(hi) ? hi : static_cast<int32_t>(raw);
            out.out_of_bounds |= raw > static_cast<uint32_t>(hi);
        } else {
            const int32_t v = static_cast<int32_t>(raw);
            c = v < 0 ? 0 : (v > hi ? hi : v);
            out.out_of_bounds |= v != c;
        }
        out.c[k] = c;
    }
    return out;
}

// The table row of window offset j (row-major over the KR row windows).
template <uint32_t KR, uint32_t K>
FORCE_INLINE uint32_t window_row(
    const Coords<K>& coords, uint32_t j, const std::array<uint32_t, K>& windows, const std::array<uint32_t, K>& strides) {
    uint32_t row = 0;
    for (uint32_t k = KR; k-- > 0;) {
        const uint32_t o = j % windows[k];
        j /= windows[k];
        row += (static_cast<uint32_t>(coords.c[k]) + o) * strides[k];
    }
    return row;
}

// Element types of a reduction, chosen at compile time by the host.
enum ElemKind : uint32_t {
    kBytes = 0,  // overwrite only
    kF32 = 1,
    kBF16 = 2,
    kF16 = 3,
    kI32 = 4,
    kU32 = 5,
    kI16 = 6,
    kU16 = 7,
    kI8 = 8,
    kU8 = 9,
};

enum ReduceKind : uint32_t { kNone = 0, kAdd = 1, kMul = 2, kMax = 3, kMin = 4 };

FORCE_INLINE float bf16_to_f32(uint16_t bits) {
    const uint32_t u = static_cast<uint32_t>(bits) << 16;
    return __builtin_bit_cast(float, u);
}

// Round to nearest even. NaN keeps a quiet payload bit so it stays NaN.
FORCE_INLINE uint16_t f32_to_bf16(float value) {
    uint32_t u = __builtin_bit_cast(uint32_t, value);
    if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu) != 0) {
        return static_cast<uint16_t>((u >> 16) | 0x0040u);
    }
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return static_cast<uint16_t>(u >> 16);
}

FORCE_INLINE float f16_to_f32(uint16_t bits) {
    const uint32_t sign = (static_cast<uint32_t>(bits) & 0x8000u) << 16;
    uint32_t exp = (bits >> 10) & 0x1fu;
    uint32_t mant = bits & 0x3ffu;
    uint32_t u;
    if (exp == 0) {
        if (mant == 0) {
            u = sign;
        } else {
            // Subnormal: normalize.
            int shift = 0;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                ++shift;
            }
            mant &= 0x3ffu;
            u = sign | ((127 - 15 + 1 - shift) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        u = sign | 0x7f800000u | (mant << 13);
    } else {
        u = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    return __builtin_bit_cast(float, u);
}

// Round to nearest even, with subnormals, overflow to infinity and NaN kept.
FORCE_INLINE uint16_t f32_to_f16(float value) {
    const uint32_t u = __builtin_bit_cast(uint32_t, value);
    const uint16_t sign = static_cast<uint16_t>((u >> 16) & 0x8000u);
    const uint32_t exp = (u >> 23) & 0xffu;
    uint32_t mant = u & 0x7fffffu;
    if (exp == 0xffu) {
        return sign | 0x7c00u | (mant ? 0x0200u | (mant >> 13) : 0u);
    }
    const int32_t e = static_cast<int32_t>(exp) - 127 + 15;
    if (e >= 31) {
        return sign | 0x7c00u;
    }
    if (e <= 0) {
        if (e < -10) {
            return sign;
        }
        // Subnormal result: shift the implicit one in, then round.
        mant |= 0x800000u;
        const uint32_t shift = static_cast<uint32_t>(14 - e);
        const uint32_t half = 1u << (shift - 1);
        const uint32_t rem = mant & ((1u << shift) - 1);
        uint32_t r = mant >> shift;
        if (rem > half || (rem == half && (r & 1u))) {
            ++r;
        }
        return sign | static_cast<uint16_t>(r);
    }
    uint32_t r = (static_cast<uint32_t>(e) << 10) | (mant >> 13);
    const uint32_t rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1u))) {
        ++r;  // may carry into the exponent, which is the right rounding
    }
    return sign | static_cast<uint16_t>(r);
}

FORCE_INLINE bool is_nan(float v) { return v != v; }

// StableHLO maximum/minimum: NaN wins, and -0 orders below +0.
FORCE_INLINE float fmax_total(float a, float b) {
    if (is_nan(a) || is_nan(b)) {
        return __builtin_bit_cast(float, 0x7fc00000u);
    }
    if (a == b) {
        return __builtin_bit_cast(uint32_t, a) & 0x80000000u ? b : a;
    }
    return a > b ? a : b;
}

FORCE_INLINE float fmin_total(float a, float b) {
    if (is_nan(a) || is_nan(b)) {
        return __builtin_bit_cast(float, 0x7fc00000u);
    }
    if (a == b) {
        return __builtin_bit_cast(uint32_t, a) & 0x80000000u ? a : b;
    }
    return a < b ? a : b;
}

template <uint32_t kReduce>
FORCE_INLINE float reduce_f32(float cur, float upd) {
    if constexpr (kReduce == kAdd) {
        return cur + upd;
    } else if constexpr (kReduce == kMul) {
        return cur * upd;
    } else if constexpr (kReduce == kMax) {
        return fmax_total(cur, upd);
    } else {
        return fmin_total(cur, upd);
    }
}

// Integers add and multiply modulo 2^bits, like StableHLO.
template <typename T, typename U, uint32_t kReduce>
FORCE_INLINE T reduce_int(T cur, T upd) {
    if constexpr (kReduce == kAdd) {
        return static_cast<T>(static_cast<U>(static_cast<U>(cur) + static_cast<U>(upd)));
    } else if constexpr (kReduce == kMul) {
        return static_cast<T>(static_cast<U>(static_cast<U>(cur) * static_cast<U>(upd)));
    } else if constexpr (kReduce == kMax) {
        return cur > upd ? cur : upd;
    } else {
        return cur < upd ? cur : upd;
    }
}

// cur[i] = reduce(cur[i], upd[i]) over `count` elements in L1.
template <uint32_t kElem, uint32_t kReduce>
FORCE_INLINE void reduce_rows(uint32_t cur_addr, uint32_t upd_addr, uint32_t count) {
    if constexpr (kElem == kF32) {
        auto* cur = reinterpret_cast<volatile tt_l1_ptr float*>(cur_addr);
        auto* upd = reinterpret_cast<volatile tt_l1_ptr float*>(upd_addr);
        for (uint32_t i = 0; i < count; ++i) {
            cur[i] = reduce_f32<kReduce>(cur[i], upd[i]);
        }
    } else if constexpr (kElem == kBF16) {
        auto* cur = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cur_addr);
        auto* upd = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(upd_addr);
        for (uint32_t i = 0; i < count; ++i) {
            cur[i] = f32_to_bf16(reduce_f32<kReduce>(bf16_to_f32(cur[i]), bf16_to_f32(upd[i])));
        }
    } else if constexpr (kElem == kF16) {
        auto* cur = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cur_addr);
        auto* upd = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(upd_addr);
        for (uint32_t i = 0; i < count; ++i) {
            cur[i] = f32_to_f16(reduce_f32<kReduce>(f16_to_f32(cur[i]), f16_to_f32(upd[i])));
        }
    } else if constexpr (kElem == kI32) {
        auto* cur = reinterpret_cast<volatile tt_l1_ptr int32_t*>(cur_addr);
        auto* upd = reinterpret_cast<volatile tt_l1_ptr int32_t*>(upd_addr);
        for (uint32_t i = 0; i < count; ++i) {
            cur[i] = reduce_int<int32_t, uint32_t, kReduce>(cur[i], upd[i]);
        }
    } else if constexpr (kElem == kU32) {
        auto* cur = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cur_addr);
        auto* upd = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(upd_addr);
        for (uint32_t i = 0; i < count; ++i) {
            cur[i] = reduce_int<uint32_t, uint32_t, kReduce>(cur[i], upd[i]);
        }
    } else if constexpr (kElem == kI16) {
        auto* cur = reinterpret_cast<volatile tt_l1_ptr int16_t*>(cur_addr);
        auto* upd = reinterpret_cast<volatile tt_l1_ptr int16_t*>(upd_addr);
        for (uint32_t i = 0; i < count; ++i) {
            cur[i] = reduce_int<int16_t, uint16_t, kReduce>(cur[i], upd[i]);
        }
    } else if constexpr (kElem == kU16) {
        auto* cur = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cur_addr);
        auto* upd = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(upd_addr);
        for (uint32_t i = 0; i < count; ++i) {
            cur[i] = reduce_int<uint16_t, uint16_t, kReduce>(cur[i], upd[i]);
        }
    } else if constexpr (kElem == kI8) {
        auto* cur = reinterpret_cast<volatile tt_l1_ptr int8_t*>(cur_addr);
        auto* upd = reinterpret_cast<volatile tt_l1_ptr int8_t*>(upd_addr);
        for (uint32_t i = 0; i < count; ++i) {
            cur[i] = reduce_int<int8_t, uint8_t, kReduce>(cur[i], upd[i]);
        }
    } else if constexpr (kElem == kU8) {
        auto* cur = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(cur_addr);
        auto* upd = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(upd_addr);
        for (uint32_t i = 0; i < count; ++i) {
            cur[i] = reduce_int<uint8_t, uint8_t, kReduce>(cur[i], upd[i]);
        }
    }
}

// NoC transfers need the L1 address to share the DRAM address's offset within
// a 64-byte granule. A column window read from DRAM offset `src` is placed at
// `src % 64` inside its 64-byte-slack entry; this moves it to `dst % 64` so it
// can be written to DRAM offset `dst`. Overlap-safe.
constexpr uint32_t kGranule = 64;

FORCE_INLINE void realign(uint32_t base, uint32_t from, uint32_t to, uint32_t bytes) {
    if (from == to) {
        return;
    }
    auto* p = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(base);
    if (to < from) {
        for (uint32_t i = 0; i < bytes; ++i) {
            p[to + i] = p[from + i];
        }
    } else {
        for (uint32_t i = bytes; i-- > 0;) {
            p[to + i] = p[from + i];
        }
    }
}

// Reads `count` index pages, starting at page `first`, into the index DFB's
// next `count` entries (one page per entry) and publishes them. The DFB is a
// self-loop: the same kernel consumes them with wait_front/pop_front.
template <typename Accessor>
FORCE_INLINE void read_index_pages(
    Noc& noc, DataflowBuffer& dfb, const Accessor& indices, uint32_t first, uint32_t count, uint32_t bytes) {
    dfb.reserve_back(count);
    const uint32_t entry = dfb.get_entry_size();
    uint32_t addr = dfb.get_write_ptr();
    for (uint32_t i = 0; i < count; ++i, addr += entry) {
        noc.async_read(
            PrecomposedUnicastEndpoint{},
            CoreLocalMem<uint32_t>(addr),
            bytes,
            {.noc_addr = indices.get_noc_addr(first + i)},
            {.offset_bytes = 0});
    }
    noc.async_read_barrier();
    dfb.push_back(count);
}

}  // namespace indexed_rows
