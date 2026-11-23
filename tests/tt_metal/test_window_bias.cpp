// SPDX-License-Identifier: Apache-2.0

// ttnn::experimental::window_bias: the causal decode SDPA's window bias of a
// learned relative-position bias, bias[t, 0, h, j] = rel_r[t, h, :] .
// table[W - 1 - d, :] with d = p - first - j the distance of the window's
// column j, against a double reference within BF16 rounding (the table's
// values reach the FPU as TF32); and the sink tiles, bf16(sink * scale) in
// column 0, exactly. Positions cover the first tiles (window start 0), tile
// boundaries and positions far past the window.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "ttnn/operations/experimental/window_bias/window_bias.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

class WindowBiasTest : public ::testing::Test {
protected:
    void SetUp() override { mesh_device_ = MeshDevice::create(MeshDeviceConfig(MeshShape{1, 1})); }

    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
    }

    template <typename T>
    ttnn::Tensor device_tensor(const std::vector<T>& values, const ttnn::Shape& shape, DataType dtype) {
        const TensorSpec spec(shape, TensorLayout(dtype, PageConfig(Layout::TILE), {}));
        return ttnn::Tensor::from_vector(values, spec).to_device(mesh_device_.get());
    }

    size_t check(uint32_t window_tiles, uint32_t heads, const std::vector<int32_t>& positions, bool with_sink,
                 uint32_t seed) {
        const uint32_t T = positions.size();
        const uint32_t W = 32 * window_tiles;
        std::mt19937 rng(seed);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        std::vector<float> rel_r(T * heads * 16), table(2 * W * 16), sink(T * heads);
        for (float& v : rel_r) {
            v = static_cast<float>(bfloat16(normal(rng)));
        }
        for (float& v : table) {
            v = 2.0f * normal(rng);
        }
        for (float& v : sink) {
            v = static_cast<float>(bfloat16(3.0f * normal(rng)));
        }
        constexpr float sink_scale = 11.3137085f;
        std::optional<ttnn::Tensor> sink_tensor;
        if (with_sink) {
            sink_tensor = device_tensor(sink, ttnn::Shape({T, heads}), DataType::BFLOAT16);
        }
        const auto outs = ttnn::experimental::window_bias(
            device_tensor(rel_r, ttnn::Shape({T, heads * 16}), DataType::BFLOAT16),
            device_tensor(table, ttnn::Shape({2 * W, 16}), DataType::FLOAT32),
            device_tensor(positions, ttnn::Shape({T}), DataType::INT32),
            heads,
            sink_tensor,
            sink_scale);
        EXPECT_EQ(outs.size(), with_sink ? 2u : 1u);
        EXPECT_EQ(outs.at(0).logical_shape(), ttnn::Shape({T, 1, heads, W}));
        const auto got = outs.at(0).cpu().to_vector<float>();
        size_t wrong = 0;
        for (uint32_t t = 0; t < T; ++t) {
            const uint32_t p = static_cast<uint32_t>(positions[t]);
            const uint32_t first = 32 * (p / 32 + 1 > window_tiles ? p / 32 + 1 - window_tiles : 0);
            for (uint32_t h = 0; h < heads; ++h) {
                std::vector<double> want(W);
                double scale = 0;
                for (uint32_t j = 0; j < W; ++j) {
                    const int64_t d = static_cast<int64_t>(p) - first - j;
                    const int64_t row = static_cast<int64_t>(W) - 1 - d;
                    double sum = 0;
                    for (uint32_t f = 0; f < 16; ++f) {
                        sum += double(rel_r[(t * heads + h) * 16 + f]) * table[row * 16 + f];
                    }
                    want[j] = sum;
                    scale = std::max(scale, std::fabs(sum));
                }
                for (uint32_t j = 0; j < W; ++j) {
                    const double value = got[(t * heads + h) * W + j];
                    const double tolerance = std::ldexp(std::fabs(want[j]), -7) + 2e-3 * scale;
                    if (!(std::fabs(value - want[j]) <= tolerance) && wrong++ < 8) {
                        ADD_FAILURE() << "p " << p << " [" << t << "][" << h << "][" << j << "] " << value
                                      << " != " << want[j];
                    }
                }
            }
        }
        if (with_sink) {
            EXPECT_EQ(outs.at(1).logical_shape(), ttnn::Shape({T, 1, heads, 32}));
            const auto s = outs.at(1).cpu().to_vector<float>();
            for (uint32_t t = 0; t < T; ++t) {
                for (uint32_t h = 0; h < heads; ++h) {
                    for (uint32_t c = 0; c < 32; ++c) {
                        const double want =
                            c == 0 ? double(sink[t * heads + h]) * sink_scale : 0.0;
                        const double value = s[(t * heads + h) * 32 + c];
                        if (!(std::fabs(value - want) <= std::ldexp(std::fabs(want), -8)) && wrong++ < 8) {
                            ADD_FAILURE() << "sink [" << t << "][" << h << "][" << c << "] " << value
                                          << " != " << want;
                        }
                    }
                }
            }
        }
        return wrong;
    }

    std::shared_ptr<MeshDevice> mesh_device_;
};

TEST_F(WindowBiasTest, MatchesReference) {
    // Inkling-2-Mini at TP4: six heads, a 1024 extent in 33 key tiles.
    EXPECT_EQ(check(33, 6, {0}, false, 1), 0u);
    EXPECT_EQ(check(33, 6, {5}, true, 2), 0u);
    EXPECT_EQ(check(33, 6, {1055}, false, 3), 0u);
    EXPECT_EQ(check(33, 6, {1056}, true, 4), 0u);
    EXPECT_EQ(check(33, 6, {31, 32, 1000, 70001}, true, 5), 0u);
    // Small windows, many heads and tokens.
    EXPECT_EQ(check(2, 32, {0, 17, 31, 32, 33, 63, 64, 65, 95, 96, 100, 129, 200, 255, 256, 1000}, true, 6), 0u);
    std::vector<int32_t> many(32);
    for (uint32_t t = 0; t < 32; ++t) {
        many[t] = static_cast<int32_t>(37 * t + 3);
    }
    EXPECT_EQ(check(3, 8, many, true, 7), 0u);
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
