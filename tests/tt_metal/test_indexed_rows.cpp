// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

// ttnn::experimental::rows_gather and rows_scatter move table rows by index:
// any element type as bytes, clamped or dropped out-of-bounds coordinates,
// several index components with windows, batched or shared indices, rows
// larger than a chunk, and deterministic reductions.

#include <gtest/gtest.h>

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

    std::shared_ptr<MeshDevice> mesh_device_;
};

// Host reference of both ops over flat row-major [B, N, R] / [Bi, M, K] data.
struct Ref {
    uint32_t B, N, R, Bi, M, K;
    std::vector<uint32_t> sizes, windows, strides;
    uint32_t W = 1;

    Ref(uint32_t B, uint32_t R, uint32_t Bi, uint32_t M, std::vector<uint32_t> sizes_, std::vector<uint32_t> windows_) :
        B(B), R(R), Bi(Bi), M(M), sizes(std::move(sizes_)), windows(std::move(windows_)) {
        K = sizes.size();
        N = 1;
        strides.assign(K, 1);
        for (uint32_t k = K; k-- > 0;) {
            strides[k] = N;
            N *= sizes[k];
        }
        for (uint32_t w : windows) {
            W *= w;
        }
    }

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
        for (uint32_t k = K; k-- > 0;) {
            uint32_t o = j % windows[k];
            j /= windows[k];
            r += (c[k] + o) * strides[k];
        }
        return r;
    }

    template <typename T>
    std::vector<T> gather(const std::vector<T>& table, const std::vector<int32_t>& idx, bool fill) const {
        std::vector<T> out(static_cast<size_t>(B) * M * W * R, T{});
        std::vector<int64_t> c;
        for (uint32_t b = 0; b < B; ++b) {
            for (uint32_t m = 0; m < M; ++m) {
                bool oob = coords(idx, Bi == 1 ? 0 : b, m, c);
                for (uint32_t j = 0; j < W; ++j) {
                    T* dst = out.data() + ((static_cast<size_t>(b) * M + m) * W + j) * R;
                    if (oob && fill) {
                        continue;
                    }
                    const T* src = table.data() + (static_cast<size_t>(b) * N + row(c, j)) * R;
                    std::copy(src, src + R, dst);
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
                for (uint32_t j = 0; j < W; ++j) {
                    T* dst = out.data() + (static_cast<size_t>(b) * N + row(c, j)) * R;
                    const T* src = source.data() + ((static_cast<size_t>(b) * M + m) * W + j) * R;
                    for (uint32_t r = 0; r < R; ++r) {
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
        to_device(table, {1, ref.N, ref.R}, DataType::BFLOAT16), to_device(idx, {1, ref.M, 1}, DataType::INT32),
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
            to_device(idx, {ref.Bi, ref.M, ref.K}, DataType::INT32), IndexedRowsSpec{{6, 8}, {1, 3}}, fill);
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
        to_device(table_i32, {ref.B, ref.N, 1}, DataType::INT32), to_device(idx, {1, ref.M, 1}, DataType::UINT32),
        IndexedRowsSpec{{40}, {1}});
    auto want_idx = idx;
    want_idx[3] = 39;
    EXPECT_EQ(out.cpu().to_vector<int32_t>(), ref.gather(table_i32, want_idx, false));

    auto table_u8 = iota_values<uint8_t>(ref.B * ref.N, 1.0f);
    out = ttnn::experimental::rows_gather(
        to_device(table_u8, {ref.B, ref.N, 1}, DataType::UINT8), to_device(idx, {1, ref.M, 1}, DataType::UINT32),
        IndexedRowsSpec{{40}, {1}});
    EXPECT_EQ(out.cpu().to_vector<uint8_t>(), ref.gather(table_u8, want_idx, false));
}

TEST_F(IndexedRowsTest, GatherRowsLargerThanAChunk) {
    Ref ref(/*B=*/1, /*R=*/40000, /*Bi=*/1, /*M=*/5, {7}, {1});
    std::vector<int32_t> idx = {6, 0, 3, 3, 9};
    auto table = iota_values<bfloat16>(ref.N * ref.R, 0.25f);
    auto out = ttnn::experimental::rows_gather(
        to_device(table, {1, ref.N, ref.R}, DataType::BFLOAT16), to_device(idx, {1, ref.M, 1}, DataType::INT32),
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
        to_device(idx, {ref.Bi, ref.M, 1}, DataType::INT32), to_device(source, {ref.B, ref.M, ref.R}, DataType::FLOAT32),
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
            to_device(table, {1, ref.N, ref.R}, DataType::FLOAT32), to_device(idx, {1, ref.M, 1}, DataType::INT32),
            to_device(source, {1, ref.M, ref.R}, DataType::FLOAT32), IndexedRowsSpec{{64}, {1}}, RowsReduce::Add);
        EXPECT_EQ(out.cpu().to_vector<float>(), ref.scatter(table, idx, source, [](float a, float s) { return a + s; }));
    }
    {
        auto table = iota_values<bfloat16>(ref.N * ref.R, 1.0f);
        auto source = iota_values<bfloat16>(ref.M * ref.R, 0.125f);
        auto out = ttnn::experimental::rows_scatter(
            to_device(table, {1, ref.N, ref.R}, DataType::BFLOAT16), to_device(idx, {1, ref.M, 1}, DataType::INT32),
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
            to_device(table, {1, ref.N, ref.R}, DataType::INT32), to_device(idx, {1, ref.M, 1}, DataType::INT32),
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
            to_device(table, {1, ref.N, ref.R}, DataType::FLOAT32), to_device(idx, {1, ref.M, 2}, DataType::INT32),
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

}  // namespace
}  // namespace tt::tt_metal::distributed::test
