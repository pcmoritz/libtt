// SPDX-License-Identifier: Apache-2.0

// ttnn::avg_pool2d takes FLOAT16 inputs and returns the FLOAT16 mean;
// ttnn::max_pool2d rejects them.

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <vector>

#include <tt-metalium/mesh_device.hpp>

#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/core/to_layout/to_layout_op.hpp"
#include "ttnn/operations/core/to_memory_config/to_memory_config_op.hpp"
#include "ttnn/operations/pool/generic/generic_pools.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

constexpr uint32_t N = 2, H = 9, W = 11, C = 32, K = 3, S = 2, P = 1;
constexpr uint32_t OH = (H + 2 * P - K) / S + 1, OW = (W + 2 * P - K) / S + 1;

class Pool2dFloat16Test : public ::testing::Test {
protected:
    // Pooling keeps its configuration tensors in L1_SMALL.
    void SetUp() override { mesh_device_ = MeshDevice::create(MeshDeviceConfig(MeshShape{1, 1}), 1 << 16); }

    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
    }

    // NHWC values as [1, 1, N * H * W, C]: random float16 values with low
    // mantissa bits that bfloat16 drops, ties, and the largest finite values.
    static std::vector<float> inputs() {
        std::mt19937 rng(0);
        std::normal_distribution<float> normal(0.0f, 4.0f);
        std::vector<float> values(N * H * W * C);
        for (float& v : values) {
            const auto bits = static_cast<uint16_t>(std::bit_cast<uint16_t>(static_cast<_Float16>(normal(rng))) | 0x0005);
            v = static_cast<float>(std::bit_cast<_Float16>(bits));
        }
        values[0] = 65504.0f;
        values[1] = -65504.0f;
        values[C] = values[0];  // a tie with the largest value in the first window
        return values;
    }

    ttnn::Tensor device_tensor(const std::vector<float>& values) {
        const TensorSpec spec(
            ttnn::Shape({1, 1, N * H * W, C}), TensorLayout(DataType::FLOAT16, PageConfig(Layout::ROW_MAJOR), {}));
        return ttnn::Tensor::from_vector(values, spec).to_device(mesh_device_.get());
    }

    // The sum, the sum of magnitudes and the number of the values of the padded
    // window at (n, oh, ow, c) that lie inside the input.
    struct Window {
        double sum = 0, magnitudes = 0;
        uint32_t inside = 0;
    };
    static Window window(const std::vector<float>& x, uint32_t n, uint32_t oh, uint32_t ow, uint32_t c) {
        Window window;
        for (uint32_t kh = 0; kh < K; ++kh) {
            for (uint32_t kw = 0; kw < K; ++kw) {
                const int h = static_cast<int>(oh * S + kh) - static_cast<int>(P);
                const int w = static_cast<int>(ow * S + kw) - static_cast<int>(P);
                if (h >= 0 && h < static_cast<int>(H) && w >= 0 && w < static_cast<int>(W)) {
                    const double value = x[((n * H + h) * W + w) * C + c];
                    window.sum += value;
                    window.magnitudes += std::abs(value);
                    ++window.inside;
                }
            }
        }
        return window;
    }

    void check_avg_pool(bool count_include_pad, std::optional<int32_t> divisor_override);

    std::shared_ptr<MeshDevice> mesh_device_;
};

// Max pooling clears unused SrcA rows to -inf, which reads as NaN in SrcA's
// float16 format, so it rejects FLOAT16 inputs.
TEST_F(Pool2dFloat16Test, MaxPoolRejectsFloat16) {
    EXPECT_ANY_THROW(ttnn::max_pool2d(
        device_tensor(inputs()), N, H, W, C, {K, K}, {S, S}, std::array<uint32_t, 2>{P, P}, {1, 1}, false,
        std::nullopt, std::nullopt, std::nullopt, false, true, false, DataType::FLOAT16));
}

// The mean of each window: its sum over the window size, over the count of its
// values inside the input without count_include_pad, or over divisor_override.
void Pool2dFloat16Test::check_avg_pool(bool count_include_pad, std::optional<int32_t> divisor_override) {
    const std::vector<float> x = inputs();
    const ttnn::Tensor out = ttnn::avg_pool2d(
        device_tensor(x), N, H, W, C, {K, K}, {S, S}, std::array<uint32_t, 2>{P, P}, false, count_include_pad,
        divisor_override, std::nullopt, std::nullopt, std::nullopt, std::nullopt, false, true, DataType::FLOAT16);
    ASSERT_EQ(out.dtype(), DataType::FLOAT16);
    // FLOAT32 holds every float16 value; host tensors do not convert FLOAT16 to float.
    const std::vector<float> got = ttnn::typecast(out, DataType::FLOAT32).cpu().to_vector<float>();
    ASSERT_EQ(got.size(), N * OH * OW * C);
    size_t mismatches = 0;
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t oh = 0; oh < OH; ++oh) {
            for (uint32_t ow = 0; ow < OW; ++ow) {
                for (uint32_t c = 0; c < C; ++c) {
                    const Window pooled = window(x, n, oh, ow, c);
                    const double divisor = divisor_override ? *divisor_override
                                           : count_include_pad ? K * K
                                                               : pooled.inside;
                    const double expected = pooled.sum / divisor;
                    // Summing K * K float16 terms, with the scaling and output roundings, may err by
                    // K * K + 2 float16 unit roundoffs of the sum of magnitudes.
                    const double tolerance = (K * K + 2) * std::ldexp(pooled.magnitudes / divisor, -11);
                    const float actual = got[((n * OH + oh) * OW + ow) * C + c];
                    if ((!std::isfinite(actual) || std::abs(actual - expected) > tolerance) && mismatches++ < 8) {
                        ADD_FAILURE() << "avg at " << n << "," << oh << "," << ow << "," << c << ": " << actual
                                      << " vs " << expected;
                    }
                }
            }
        }
    }
    EXPECT_EQ(mismatches, 0u);
}

TEST_F(Pool2dFloat16Test, AvgPoolIsWithinFloat16Rounding) { check_avg_pool(true, std::nullopt); }

// Border windows divide by fewer values: per-core float16 scalars.
TEST_F(Pool2dFloat16Test, AvgPoolWithoutPaddingInCount) { check_avg_pool(false, std::nullopt); }

TEST_F(Pool2dFloat16Test, AvgPoolWithDivisorOverride) { check_avg_pool(true, 7); }

TEST_F(Pool2dFloat16Test, UnalignedOutputTilizesDirectly) {
    // Float16 and bfloat16 layout conversions use a UINT16 storage view.
    // Include a tile-aligned tensor whose individual shards are not aligned.
    for (const auto [height, width, cores] :
         {std::array<uint32_t, 3>{30, 32, 1}, {32, 48, 2}, {2, 64, 2}, {32, 64, 4}, {32, 64, 2}}) {
        SCOPED_TRACE(::testing::Message() << height << "x" << width << " across " << cores << " cores");
        std::vector<uint16_t> values(height * width);
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = static_cast<uint16_t>(i * 977 + 5);
        }
        const TensorSpec spec(
            ttnn::Shape({1, 1, height, width}), TensorLayout(DataType::UINT16, PageConfig(Layout::ROW_MAJOR), {}));
        for (const auto memory_layout : {TensorMemoryLayout::WIDTH_SHARDED, TensorMemoryLayout::HEIGHT_SHARDED}) {
            SCOPED_TRACE(::testing::Message() << "memory layout " << static_cast<int>(memory_layout));
            std::array<uint32_t, 2> shard_shape{height, width};
            shard_shape[memory_layout == TensorMemoryLayout::HEIGHT_SHARDED ? 0 : 1] /= cores;
            const MemoryConfig sharded(
                memory_layout, BufferType::L1,
                ShardSpec(CoreRangeSet({CoreRange({0, 0}, {cores - 1, 0})}), shard_shape, ShardOrientation::ROW_MAJOR));
            const auto input = ttnn::to_memory_config(
                ttnn::Tensor::from_vector(values, spec).to_device(mesh_device_.get()), sharded);
            for (const auto& memory_config : {ttnn::L1_MEMORY_CONFIG, ttnn::DRAM_MEMORY_CONFIG}) {
                const auto output = ttnn::to_layout(input, Layout::TILE, std::nullopt, memory_config);
                EXPECT_EQ(output.layout(), Layout::TILE);
                EXPECT_EQ(output.memory_config(), memory_config);
                EXPECT_EQ(output.logical_shape(), input.logical_shape());
                EXPECT_EQ(output.cpu().to_vector<uint16_t>(), values);
            }
        }
    }
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
