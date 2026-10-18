// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

// ttnn::experimental::rows_gather and rows_scatter move table rows by index:
// any element type as bytes, clamped or dropped out-of-bounds coordinates,
// several index components with windows, batched or shared indices, rows
// larger than a chunk, and deterministic reductions.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "ttnn/operations/experimental/indexed_rows/indexed_rows.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

using ttnn::experimental::IndexedRowsSpec;
using ttnn::experimental::RowsReduce;

class IndexedRowsTest : public ::testing::Test {
protected:
    void SetUp() override { mesh_device_ = MeshDevice::create(MeshDeviceConfig(MeshShape{1, 1})); }

    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
    }

    template <typename T>
    ttnn::Tensor to_device(const std::vector<T>& values, const std::vector<uint32_t>& shape, DataType dtype) {
        const TensorSpec spec(
            ttnn::Shape(shape), TensorLayout(dtype, PageConfig(Layout::ROW_MAJOR), MemoryConfig{}));
        return ttnn::Tensor::from_vector(values, spec).to_device(mesh_device_.get());
    }

    // The reference keeps indices as [Bi, M, K]; the ops take [Bi, K, M].
    template <typename T, typename R>
    ttnn::Tensor index_tensor(const R& ref, const std::vector<T>& idx, DataType dtype) {
        std::vector<T> by_component(idx.size());
        for (uint32_t bi = 0; bi < ref.Bi; ++bi) {
            for (uint32_t m = 0; m < ref.M; ++m) {
                for (uint32_t k = 0; k < ref.K; ++k) {
                    by_component[(static_cast<size_t>(bi) * ref.K + k) * ref.M + m] =
                        idx[(static_cast<size_t>(bi) * ref.M + m) * ref.K + k];
                }
            }
        }
        return to_device(by_component, {ref.Bi, ref.K, ref.M}, dtype);
    }

    std::shared_ptr<MeshDevice> mesh_device_;
};

// Host reference of both ops over flat row-major [B, N, R] / [Bi, M, K] data
// (the device index tensor is [Bi, K, M], see index_tensor).
struct Ref {
    uint32_t B, N, R, Bi, M, K;  // R: elements per row (all S pages)
    std::vector<uint32_t> sizes, windows, strides;
    bool column = false;
    uint32_t W = 1;  // over the row components

    Ref(uint32_t B, uint32_t R, uint32_t Bi, uint32_t M, std::vector<uint32_t> sizes_, std::vector<uint32_t> windows_,
        bool column_ = false) :
        B(B), R(R), Bi(Bi), M(M), sizes(std::move(sizes_)), windows(std::move(windows_)), column(column_) {
        K = sizes.size();
        const uint32_t KR = column ? K - 1 : K;
        N = 1;
        strides.assign(K, 0);
        for (uint32_t k = KR; k-- > 0;) {
            strides[k] = N;
            N *= sizes[k];
        }
        for (uint32_t k = 0; k < KR; ++k) {
            W *= windows[k];
        }
    }
    uint32_t col_window() const { return column ? windows.back() : R; }
    uint32_t col(const std::vector<int64_t>& c) const { return column ? c.back() : 0; }

    // Clamped coordinates of position (bi, m); returns whether any clamped.
    bool coords(const std::vector<int32_t>& idx, uint32_t bi, uint32_t m, std::vector<int64_t>& c) const {
        bool oob = false;
        c.resize(K);
        for (uint32_t k = 0; k < K; ++k) {
            int64_t v = idx[(bi * M + m) * K + k];
            int64_t hi = static_cast<int64_t>(sizes[k]) - windows[k];
            int64_t cl = std::min<int64_t>(std::max<int64_t>(v, 0), hi);
            oob |= cl != v;
            c[k] = cl;
        }
        return oob;
    }

    uint32_t row(const std::vector<int64_t>& c, uint32_t j) const {
        uint32_t r = 0;
        for (uint32_t k = (column ? K - 1 : K); k-- > 0;) {
            uint32_t o = j % windows[k];
            j /= windows[k];
            r += (c[k] + o) * strides[k];
        }
        return r;
    }

    template <typename T>
    std::vector<T> gather(const std::vector<T>& table, const std::vector<int32_t>& idx, bool fill) const {
        const uint32_t w = col_window();
        std::vector<T> out(static_cast<size_t>(B) * M * W * w, T{});
        std::vector<int64_t> c;
        for (uint32_t b = 0; b < B; ++b) {
            for (uint32_t m = 0; m < M; ++m) {
                bool oob = coords(idx, Bi == 1 ? 0 : b, m, c);
                for (uint32_t j = 0; j < W; ++j) {
                    T* dst = out.data() + ((static_cast<size_t>(b) * M + m) * W + j) * w;
                    if (oob && fill) {
                        continue;
                    }
                    const T* src = table.data() + (static_cast<size_t>(b) * N + row(c, j)) * R + col(c);
                    std::copy(src, src + w, dst);
                }
            }
        }
        return out;
    }

    template <typename T, typename Combine>
    std::vector<T> scatter(
        const std::vector<T>& table, const std::vector<int32_t>& idx, const std::vector<T>& source, Combine combine) const {
        std::vector<T> out = table;
        std::vector<int64_t> c;
        for (uint32_t b = 0; b < B; ++b) {
            for (uint32_t m = 0; m < M; ++m) {
                if (coords(idx, Bi == 1 ? 0 : b, m, c)) {
                    continue;
                }
                const uint32_t w = col_window();
                for (uint32_t j = 0; j < W; ++j) {
                    T* dst = out.data() + (static_cast<size_t>(b) * N + row(c, j)) * R + col(c);
                    const T* src = source.data() + ((static_cast<size_t>(b) * M + m) * W + j) * w;
                    for (uint32_t r = 0; r < w; ++r) {
                        dst[r] = combine(dst[r], src[r]);
                    }
                }
            }
        }
        return out;
    }
};

std::vector<int32_t> random_indices(size_t count, int32_t lo, int32_t hi, std::mt19937& rng) {
    std::uniform_int_distribution<int32_t> dist(lo, hi);
    std::vector<int32_t> out(count);
    for (auto& v : out) {
        v = dist(rng);
    }
    return out;
}

template <typename T>
std::vector<T> iota_values(size_t count, float scale) {
    std::vector<T> out(count);
    for (size_t i = 0; i < count; ++i) {
        out[i] = static_cast<T>(static_cast<float>(i % 251) * scale);
    }
    return out;
}

TEST_F(IndexedRowsTest, GatherBf16ClampsIndices) {
    Ref ref(/*B=*/1, /*R=*/64, /*Bi=*/1, /*M=*/37, {100}, {1});
    std::mt19937 rng(1);
    auto table = iota_values<bfloat16>(ref.N * ref.R, 0.5f);
    auto idx = random_indices(ref.M, -5, 104, rng);
    auto out = ttnn::experimental::rows_gather(
        to_device(table, {1, ref.N, ref.R}, DataType::BFLOAT16), index_tensor(ref, idx, DataType::INT32),
        IndexedRowsSpec{{100}, {1}});
    EXPECT_EQ(out.logical_shape(), ttnn::Shape({1, ref.M, ref.R}));
    auto got = out.cpu().to_vector<bfloat16>();
    auto want = ref.gather(table, idx, false);
    ASSERT_EQ(got.size(), want.size());
    for (size_t i = 0; i < got.size(); ++i) {
        ASSERT_EQ(got[i].operator float(), want[i].operator float()) << "at " << i;
    }
}

TEST_F(IndexedRowsTest, GatherF32WindowsBatchedAndFill) {
    // Two components (6 x 8) with a window of 3 along the second; batched
    // indices; fill mode zeroes clamped positions.
    Ref ref(/*B=*/2, /*R=*/5, /*Bi=*/2, /*M=*/11, {6, 8}, {1, 3});
    std::mt19937 rng(2);
    auto table = iota_values<float>(ref.B * ref.N * ref.R, 1.0f);
    std::vector<int32_t> idx(ref.Bi * ref.M * ref.K);
    for (uint32_t i = 0; i < ref.Bi * ref.M; ++i) {
        idx[i * 2] = random_indices(1, -1, 6, rng)[0];
        idx[i * 2 + 1] = random_indices(1, -2, 9, rng)[0];
    }
    for (bool fill : {false, true}) {
        auto out = ttnn::experimental::rows_gather(
            to_device(table, {ref.B, ref.N, ref.R}, DataType::FLOAT32),
            index_tensor(ref, idx, DataType::INT32), IndexedRowsSpec{{6, 8}, {1, 3}}, fill);
        EXPECT_EQ(out.logical_shape(), ttnn::Shape({ref.B, ref.M * ref.W, ref.R}));
        EXPECT_EQ(out.cpu().to_vector<float>(), ref.gather(table, idx, fill)) << "fill=" << fill;
    }
}

TEST_F(IndexedRowsTest, GatherSharedUint32IndicesSmallRows) {
    // One index batch shared by three table batches; uint32 indices, a
    // one-element row of int32 and uint8 payloads.
    Ref ref(/*B=*/3, /*R=*/1, /*Bi=*/1, /*M=*/50, {40}, {1});
    std::mt19937 rng(3);
    auto idx = random_indices(ref.M, 0, 60, rng);
    idx[3] = -1;  // as uint32: past the end, clamps high
    auto table_i32 = iota_values<int32_t>(ref.B * ref.N, 1000.0f);
    auto out = ttnn::experimental::rows_gather(
        to_device(table_i32, {ref.B, ref.N, 1}, DataType::INT32), index_tensor(ref, idx, DataType::UINT32),
        IndexedRowsSpec{{40}, {1}});
    auto want_idx = idx;
    want_idx[3] = 39;
    EXPECT_EQ(out.cpu().to_vector<int32_t>(), ref.gather(table_i32, want_idx, false));

    auto table_u8 = iota_values<uint8_t>(ref.B * ref.N, 1.0f);
    out = ttnn::experimental::rows_gather(
        to_device(table_u8, {ref.B, ref.N, 1}, DataType::UINT8), index_tensor(ref, idx, DataType::UINT32),
        IndexedRowsSpec{{40}, {1}});
    EXPECT_EQ(out.cpu().to_vector<uint8_t>(), ref.gather(table_u8, want_idx, false));
}

TEST_F(IndexedRowsTest, GatherRowsLargerThanAChunk) {
    Ref ref(/*B=*/1, /*R=*/40000, /*Bi=*/1, /*M=*/5, {7}, {1});
    std::vector<int32_t> idx = {6, 0, 3, 3, 9};
    auto table = iota_values<bfloat16>(ref.N * ref.R, 0.25f);
    auto out = ttnn::experimental::rows_gather(
        to_device(table, {1, ref.N, ref.R}, DataType::BFLOAT16), index_tensor(ref, idx, DataType::INT32),
        IndexedRowsSpec{{7}, {1}});
    auto got = out.cpu().to_vector<bfloat16>();
    auto want = ref.gather(table, idx, false);
    ASSERT_EQ(got.size(), want.size());
    for (size_t i = 0; i < got.size(); ++i) {
        ASSERT_EQ(got[i].operator float(), want[i].operator float()) << "at " << i;
    }
}

TEST_F(IndexedRowsTest, ScatterOverwriteDropsAndOrdersDuplicates) {
    Ref ref(/*B=*/2, /*R=*/16, /*Bi=*/2, /*M=*/60, {50}, {1});
    std::mt19937 rng(4);
    auto table = iota_values<float>(ref.B * ref.N * ref.R, 1.0f);
    auto idx = random_indices(ref.Bi * ref.M, -3, 55, rng);  // duplicates and out-of-bounds
    auto source = iota_values<float>(ref.B * ref.M * ref.R, -1.0f);
    auto out = ttnn::experimental::rows_scatter(
        to_device(table, {ref.B, ref.N, ref.R}, DataType::FLOAT32),
        index_tensor(ref, idx, DataType::INT32), to_device(source, {ref.B, ref.M, ref.R}, DataType::FLOAT32),
        IndexedRowsSpec{{50}, {1}});
    EXPECT_EQ(out.logical_shape(), ttnn::Shape({ref.B, ref.N, ref.R}));
    EXPECT_EQ(out.cpu().to_vector<float>(), ref.scatter(table, idx, source, [](float, float s) { return s; }));
}

TEST_F(IndexedRowsTest, ScatterAddF32Bf16Int32) {
    Ref ref(/*B=*/1, /*R=*/33, /*Bi=*/1, /*M=*/200, {64}, {1});
    std::mt19937 rng(5);
    auto idx = random_indices(ref.M, 0, 63, rng);
    {
        auto table = iota_values<float>(ref.N * ref.R, 1.0f);
        auto source = iota_values<float>(ref.M * ref.R, 0.125f);
        auto out = ttnn::experimental::rows_scatter(
            to_device(table, {1, ref.N, ref.R}, DataType::FLOAT32), index_tensor(ref, idx, DataType::INT32),
            to_device(source, {1, ref.M, ref.R}, DataType::FLOAT32), IndexedRowsSpec{{64}, {1}}, RowsReduce::Add);
        EXPECT_EQ(out.cpu().to_vector<float>(), ref.scatter(table, idx, source, [](float a, float s) { return a + s; }));
    }
    {
        auto table = iota_values<bfloat16>(ref.N * ref.R, 1.0f);
        auto source = iota_values<bfloat16>(ref.M * ref.R, 0.125f);
        auto out = ttnn::experimental::rows_scatter(
            to_device(table, {1, ref.N, ref.R}, DataType::BFLOAT16), index_tensor(ref, idx, DataType::INT32),
            to_device(source, {1, ref.M, ref.R}, DataType::BFLOAT16), IndexedRowsSpec{{64}, {1}}, RowsReduce::Add);
        auto got = out.cpu().to_vector<bfloat16>();
        auto want = ref.scatter(table, idx, source, [](bfloat16 a, bfloat16 s) {
            return bfloat16(static_cast<float>(a) + static_cast<float>(s));
        });
        ASSERT_EQ(got.size(), want.size());
        for (size_t i = 0; i < got.size(); ++i) {
            ASSERT_EQ(got[i].operator float(), want[i].operator float()) << "at " << i;
        }
    }
    {
        auto table = iota_values<int32_t>(ref.N * ref.R, 1.0f);
        auto source = iota_values<int32_t>(ref.M * ref.R, 7.0f);
        table[0] = std::numeric_limits<int32_t>::max();  // wraps
        auto out = ttnn::experimental::rows_scatter(
            to_device(table, {1, ref.N, ref.R}, DataType::INT32), index_tensor(ref, idx, DataType::INT32),
            to_device(source, {1, ref.M, ref.R}, DataType::INT32), IndexedRowsSpec{{64}, {1}}, RowsReduce::Add);
        EXPECT_EQ(out.cpu().to_vector<int32_t>(), ref.scatter(table, idx, source, [](int32_t a, int32_t s) {
            return static_cast<int32_t>(static_cast<uint32_t>(a) + static_cast<uint32_t>(s));
        }));
    }
}

TEST_F(IndexedRowsTest, ScatterMaxMinPropagateNaNAndWindows) {
    // Two components with a window of 2 along the first; NaN wins in max/min.
    Ref ref(/*B=*/1, /*R=*/8, /*Bi=*/1, /*M=*/30, {10, 4}, {2, 1});
    std::mt19937 rng(6);
    std::vector<int32_t> idx(ref.M * 2);
    for (uint32_t i = 0; i < ref.M; ++i) {
        idx[i * 2] = random_indices(1, -1, 10, rng)[0];
        idx[i * 2 + 1] = random_indices(1, 0, 3, rng)[0];
    }
    auto table = iota_values<float>(ref.N * ref.R, 1.0f);
    auto source = iota_values<float>(ref.M * ref.W * ref.R, 3.0f);
    source[5] = std::numeric_limits<float>::quiet_NaN();
    source[9] = -0.0f;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (auto [reduce, combine] : {
             std::pair{RowsReduce::Max, +[](float a, float s) { return std::isnan(a) || std::isnan(s) ? std::numeric_limits<float>::quiet_NaN() : std::max(a, s); }},
             std::pair{RowsReduce::Min, +[](float a, float s) { return std::isnan(a) || std::isnan(s) ? std::numeric_limits<float>::quiet_NaN() : std::min(a, s); }},
             std::pair{RowsReduce::Mul, +[](float a, float s) { return a * s; }},
         }) {
        auto out = ttnn::experimental::rows_scatter(
            to_device(table, {1, ref.N, ref.R}, DataType::FLOAT32), index_tensor(ref, idx, DataType::INT32),
            to_device(source, {1, ref.M * ref.W, ref.R}, DataType::FLOAT32), IndexedRowsSpec{{10, 4}, {2, 1}}, reduce);
        auto got = out.cpu().to_vector<float>();
        auto want = ref.scatter(table, idx, source, combine);
        ASSERT_EQ(got.size(), want.size());
        for (size_t i = 0; i < got.size(); ++i) {
            if (std::isnan(want[i])) {
                EXPECT_TRUE(std::isnan(got[i])) << "at " << i;
            } else {
                EXPECT_EQ(got[i], want[i]) << "at " << i;
            }
        }
    }
    (void)nan;
}

TEST_F(IndexedRowsTest, GatherAndScatterRowsOfSeveralPages) {
    // A [B, N, S, R] table: each row is S pages, as a KV cache's [heads, dim].
    Ref ref(/*B=*/2, /*R=*/6 * 40, /*Bi=*/2, /*M=*/9, {50}, {1});
    std::mt19937 rng(7);
    auto table = iota_values<bfloat16>(ref.B * ref.N * ref.R, 0.5f);
    auto idx = random_indices(ref.Bi * ref.M, -2, 52, rng);
    auto out = ttnn::experimental::rows_gather(
        to_device(table, {2, 50, 6, 40}, DataType::BFLOAT16), index_tensor(ref, idx, DataType::INT32),
        IndexedRowsSpec{{50}, {1}});
    EXPECT_EQ(out.logical_shape(), ttnn::Shape({2, ref.M, 6, 40}));
    auto got = out.cpu().to_vector<bfloat16>();
    auto want = ref.gather(table, idx, false);
    ASSERT_EQ(got.size(), want.size());
    for (size_t i = 0; i < got.size(); ++i) {
        ASSERT_EQ(static_cast<float>(got[i]), static_cast<float>(want[i])) << "at " << i;
    }
    auto source = iota_values<bfloat16>(ref.B * ref.M * ref.R, -1.0f);
    out = ttnn::experimental::rows_scatter(
        to_device(table, {2, 50, 6, 40}, DataType::BFLOAT16), index_tensor(ref, idx, DataType::INT32),
        to_device(source, {2, ref.M, 6, 40}, DataType::BFLOAT16), IndexedRowsSpec{{50}, {1}});
    got = out.cpu().to_vector<bfloat16>();
    want = ref.scatter(table, idx, source, [](bfloat16, bfloat16 s) { return s; });
    for (size_t i = 0; i < got.size(); ++i) {
        ASSERT_EQ(static_cast<float>(got[i]), static_cast<float>(want[i])) << "at " << i;
    }
}

TEST_F(IndexedRowsTest, GatherAndScatterRowsOfManySmallPages) {
    // A short-convolution state [slots, 2560, 9] row-major: one slot is 2560 pages of 18 bytes, read by
    // many cores (each a range of the transfers). Then several positions, whose transfers straddle the
    // cores' ranges.
    for (uint32_t M : {1u, 3u}) {
        Ref ref(/*B=*/1, /*R=*/2560 * 9, /*Bi=*/1, M, {3}, {1});
        std::mt19937 rng(11 + M);
        auto table = iota_values<bfloat16>(ref.B * ref.N * ref.R, 0.25f);
        auto idx = random_indices(ref.Bi * ref.M, 0, 3, rng);
        auto out = ttnn::experimental::rows_gather(
            to_device(table, {1, 3, 2560, 9}, DataType::BFLOAT16), index_tensor(ref, idx, DataType::INT32),
            IndexedRowsSpec{{3}, {1}});
        EXPECT_EQ(out.logical_shape(), ttnn::Shape({1, M, 2560, 9}));
        auto got = out.cpu().to_vector<bfloat16>();
        auto want = ref.gather(table, idx, false);
        ASSERT_EQ(got.size(), want.size());
        for (size_t i = 0; i < got.size(); ++i) {
            ASSERT_EQ(static_cast<float>(got[i]), static_cast<float>(want[i])) << "M " << M << " at " << i;
        }
        // Writing the slots back: each core reads the source pages of a position together, and
        // duplicate positions still apply in order.
        auto source = iota_values<bfloat16>(ref.B * ref.M * ref.R, -0.5f);
        out = ttnn::experimental::rows_scatter(
            to_device(table, {1, 3, 2560, 9}, DataType::BFLOAT16), index_tensor(ref, idx, DataType::INT32),
            to_device(source, {1, M, 2560, 9}, DataType::BFLOAT16), IndexedRowsSpec{{3}, {1}});
        got = out.cpu().to_vector<bfloat16>();
        want = ref.scatter(table, idx, source, [](bfloat16, bfloat16 s) { return s; });
        ASSERT_EQ(got.size(), want.size());
        for (size_t i = 0; i < got.size(); ++i) {
            ASSERT_EQ(static_cast<float>(got[i]), static_cast<float>(want[i])) << "scatter M " << M << " at " << i;
        }
    }
}

TEST_F(IndexedRowsTest, ColumnWindowGatherAndScatterAdd) {
    // The last component picks a window of 3 elements inside a row of 100:
    // a sampler's logprobs[row, label] with window 1 would be the same with w=1.
    Ref ref(/*B=*/1, /*R=*/100, /*Bi=*/1, /*M=*/40, {16, 100}, {1, 3}, /*column=*/true);
    std::mt19937 rng(8);
    auto table = iota_values<float>(ref.N * ref.R, 1.0f);
    std::vector<int32_t> idx(ref.M * 2);
    for (uint32_t i = 0; i < ref.M; ++i) {
        idx[i * 2] = random_indices(1, -1, 17, rng)[0];
        idx[i * 2 + 1] = random_indices(1, -2, 102, rng)[0];
    }
    auto out = ttnn::experimental::rows_gather(
        to_device(table, {1, 16, 100}, DataType::FLOAT32), index_tensor(ref, idx, DataType::INT32),
        IndexedRowsSpec{{16, 100}, {1, 3}, true});
    EXPECT_EQ(out.logical_shape(), ttnn::Shape({1, ref.M * 3}));
    EXPECT_EQ(out.cpu().to_vector<float>(), ref.gather(table, idx, false));

    auto source = iota_values<float>(ref.M * 3, 0.25f);
    out = ttnn::experimental::rows_scatter(
        to_device(table, {1, 16, 100}, DataType::FLOAT32), index_tensor(ref, idx, DataType::INT32),
        to_device(source, {1, ref.M * 3}, DataType::FLOAT32), IndexedRowsSpec{{16, 100}, {1, 3}, true},
        RowsReduce::Add);
    EXPECT_EQ(out.cpu().to_vector<float>(), ref.scatter(table, idx, source, [](float a, float s) { return a + s; }));
}

TEST_F(IndexedRowsTest, ManyPositionsStreamAndResidentScatterAdd) {
    // 20000 positions: several index chunks; a 64-row table whose pages stay
    // in L1 on every core; duplicates accumulate.
    Ref ref(/*B=*/1, /*R=*/24, /*Bi=*/1, /*M=*/20000, {64}, {1});
    std::mt19937 rng(9);
    auto table = iota_values<float>(ref.N * ref.R, 1.0f);
    auto idx = random_indices(ref.M, -3, 70, rng);
    auto out = ttnn::experimental::rows_gather(
        to_device(table, {1, ref.N, ref.R}, DataType::FLOAT32), index_tensor(ref, idx, DataType::INT32),
        IndexedRowsSpec{{64}, {1}});
    EXPECT_EQ(out.cpu().to_vector<float>(), ref.gather(table, idx, false));
    auto source = iota_values<float>(ref.M * ref.R, 0.125f);
    out = ttnn::experimental::rows_scatter(
        to_device(table, {1, ref.N, ref.R}, DataType::FLOAT32), index_tensor(ref, idx, DataType::INT32),
        to_device(source, {1, ref.M, ref.R}, DataType::FLOAT32), IndexedRowsSpec{{64}, {1}}, RowsReduce::Add);
    auto got = out.cpu().to_vector<float>();
    auto want = ref.scatter(table, idx, source, [](float a, float s) { return a + s; });
    ASSERT_EQ(got.size(), want.size());
    for (size_t i = 0; i < got.size(); ++i) {
        ASSERT_NEAR(got[i], want[i], 1e-2f) << "at " << i;
    }
    // The same through a column window: a permutation scatter per row.
    Ref cref(/*B=*/1, /*R=*/4096, /*Bi=*/1, /*M=*/8 * 4096, {8, 4096}, {1, 1}, /*column=*/true);
    std::vector<int32_t> cidx(cref.M * 2);
    for (uint32_t r = 0; r < 8; ++r) {
        std::vector<int32_t> perm(4096);
        for (int32_t c = 0; c < 4096; ++c) {
            perm[c] = c;
        }
        std::shuffle(perm.begin(), perm.end(), rng);
        for (uint32_t c = 0; c < 4096; ++c) {
            cidx[(r * 4096 + c) * 2] = r;
            cidx[(r * 4096 + c) * 2 + 1] = perm[c];
        }
    }
    auto ctable = std::vector<float>(8 * 4096, 0.0f);
    auto csource = iota_values<float>(cref.M, 1.0f);
    out = ttnn::experimental::rows_scatter(
        to_device(ctable, {1, 8, 4096}, DataType::FLOAT32), index_tensor(cref, cidx, DataType::INT32),
        to_device(csource, {1, cref.M}, DataType::FLOAT32), IndexedRowsSpec{{8, 4096}, {1, 1}, true});
    EXPECT_EQ(out.cpu().to_vector<float>(), cref.scatter(ctable, cidx, csource, [](float, float s) { return s; }));
}

// Pages of a chunk (64 KiB) with more of them per core than fit the writer's
// 512 KiB batch, so each core takes its pages in several batches: an overwrite
// of a row and a reduction into a later batch must both land.
TEST_F(IndexedRowsTest, ScatterPagesInSeveralBatches) {
    Ref ref(/*B=*/1, /*R=*/32768, /*Bi=*/1, /*M=*/300, {1400}, {1});  // 1400 pages of 64 KiB bf16
    std::mt19937 rng(9);
    auto idx = random_indices(ref.M, -5, 1405, rng);
    auto table = iota_values<bfloat16>(static_cast<size_t>(ref.N) * ref.R, 0.001f);
    auto source = iota_values<bfloat16>(static_cast<size_t>(ref.M) * ref.R, -0.002f);
    auto table_dev = to_device(table, {1, ref.N, ref.R}, DataType::BFLOAT16);
    auto source_dev = to_device(source, {1, ref.M, ref.R}, DataType::BFLOAT16);
    auto out = ttnn::experimental::rows_scatter(
        table_dev, index_tensor(ref, idx, DataType::INT32), source_dev, IndexedRowsSpec{{1400}, {1}});
    auto got = out.cpu().to_vector<bfloat16>();
    auto want = ref.scatter(table, idx, source, [](bfloat16, bfloat16 s) { return s; });
    ASSERT_EQ(got.size(), want.size());
    for (size_t i = 0; i < got.size(); ++i) {
        ASSERT_EQ(got[i].operator float(), want[i].operator float()) << "at " << i;
    }
    out = ttnn::experimental::rows_scatter(
        table_dev, index_tensor(ref, idx, DataType::INT32), source_dev, IndexedRowsSpec{{1400}, {1}}, RowsReduce::Max);
    got = out.cpu().to_vector<bfloat16>();
    want = ref.scatter(table, idx, source, [](bfloat16 a, bfloat16 s) {
        return static_cast<float>(s) > static_cast<float>(a) ? s : a;
    });
    for (size_t i = 0; i < got.size(); ++i) {
        ASSERT_EQ(got[i].operator float(), want[i].operator float()) << "at " << i;
    }
}

// A row width whose entry does not divide the 512 KiB batch (bf16 rows of 96:
// 192-byte entries, 2730 per batch, filled 16 at a time): the batch must be a
// multiple of the reader's group or the reader waits for space at the end of
// the buffer that the writer frees only after the whole batch.
TEST_F(IndexedRowsTest, ScatterBatchIsAMultipleOfTheReaderGroup) {
    Ref ref(/*B=*/1, /*R=*/96, /*Bi=*/1, /*M=*/500, {400000}, {1});  // ~2860 pages per core
    std::mt19937 rng(10);
    auto idx = random_indices(ref.M, 0, 399999, rng);
    auto table = iota_values<bfloat16>(static_cast<size_t>(ref.N) * ref.R, 0.001f);
    auto source = iota_values<bfloat16>(static_cast<size_t>(ref.M) * ref.R, -0.5f);
    auto out = ttnn::experimental::rows_scatter(
        to_device(table, {1, ref.N, ref.R}, DataType::BFLOAT16), index_tensor(ref, idx, DataType::INT32),
        to_device(source, {1, ref.M, ref.R}, DataType::BFLOAT16), IndexedRowsSpec{{400000}, {1}});
    auto got = out.cpu().to_vector<bfloat16>();
    auto want = ref.scatter(table, idx, source, [](bfloat16, bfloat16 s) { return s; });
    ASSERT_EQ(got.size(), want.size());
    for (size_t i = 0; i < got.size(); ++i) {
        ASSERT_EQ(got[i].operator float(), want[i].operator float()) << "at " << i;
    }
}

// Column windows that straddle a 64 KiB chunk boundary, including the boundary
// between two L1 batches of a 600 KB row (10 chunks, 8 per batch): every byte
// of the window lands, by overwrite and by reduction.
TEST_F(IndexedRowsTest, ColumnWindowAcrossChunksAndBatches) {
    Ref cref(/*B=*/1, /*R=*/150000, /*Bi=*/1, /*M=*/6, {2, 150000}, {1, 4}, /*column=*/true);
    std::vector<int32_t> cidx = {
        0, 131071,  // across the batch boundary (chunk 7 | chunk 8)
        0, 16383,   // across chunk 0 | chunk 1
        1, 131071,  // the other row, across the batch boundary
        1, 149996,  // the last window of the row
        0, 0,       // the first
        1, 16383,   // duplicate of a crossing position's chunk pair
    };
    auto table = iota_values<float>(2 * 150000, 0.25f);
    auto source = iota_values<float>(cref.M * 4, -3.0f);
    auto out = ttnn::experimental::rows_scatter(
        to_device(table, {1, 2, 150000}, DataType::FLOAT32), index_tensor(cref, cidx, DataType::INT32),
        to_device(source, {1, cref.M * 4}, DataType::FLOAT32), IndexedRowsSpec{{2, 150000}, {1, 4}, true});
    EXPECT_EQ(out.cpu().to_vector<float>(), cref.scatter(table, cidx, source, [](float, float s) { return s; }));
    out = ttnn::experimental::rows_scatter(
        to_device(table, {1, 2, 150000}, DataType::FLOAT32), index_tensor(cref, cidx, DataType::INT32),
        to_device(source, {1, cref.M * 4}, DataType::FLOAT32), IndexedRowsSpec{{2, 150000}, {1, 4}, true},
        RowsReduce::Add);
    EXPECT_EQ(out.cpu().to_vector<float>(), cref.scatter(table, cidx, source, [](float a, float s) { return a + s; }));
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
