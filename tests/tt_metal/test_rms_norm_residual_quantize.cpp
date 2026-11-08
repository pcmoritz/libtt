// SPDX-License-Identifier: Apache-2.0

// ttnn::normalization::rms_norm_residual with a float8 output: the norm and
// sum must equal the op's without it, and the codes and scale must equal
// ttnn::experimental::quantize_rows of that norm bit for bit. Shapes cover
// every row stride (4, 8, 16, 32 rows), one to three rows (whose codes the
// program computes on rows gathered into whole SFPU vectors; quantize_rows
// gathers one or two), one core, cores with one and several tiles, and a full
// 32 cores; values cover zero rows (the min_scale floor), outliers and
// zero-centered weights.

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "ttnn/operations/experimental/quantize_rows/quantize_rows.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/normalization/rms_norm_residual/rms_norm_residual.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

constexpr float kMinScale = 1e-12f;

class RMSNormResidualQuantizeTest : public ::testing::Test {
protected:
    void SetUp() override { mesh_device_ = MeshDevice::create(MeshDeviceConfig(MeshShape{1, 1})); }

    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
    }

    ttnn::Tensor device_tensor(const std::vector<float>& values, uint32_t rows, uint32_t cols) {
        const TensorSpec spec(
            ttnn::Shape({rows, cols}), TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), {}));
        return ttnn::Tensor::from_vector(values, spec).to_device(mesh_device_.get());
    }

    // Runs the fused op and the separate ones; returns the number of mismatches.
    size_t check(uint32_t rows, uint32_t cols, DataType dtype, float weight_offset, uint32_t seed, bool zero_row) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        std::uniform_real_distribution<float> unit(0.0f, 1.0f);
        std::vector<float> x(static_cast<size_t>(rows) * cols), r(x.size()), w(cols);
        for (size_t i = 0; i < x.size(); ++i) {
            const float scale = std::ldexp(1.0f, static_cast<int>(i / cols % 7) - 3);
            x[i] = normal(rng) * scale * (unit(rng) < 0.002f ? 40.0f : 1.0f);
            r[i] = normal(rng) * scale;
        }
        if (zero_row) {
            for (uint32_t k = 0; k < cols; ++k) {
                x[k] = 0.0f;
                r[k] = 0.0f;
            }
        }
        for (float& v : w) {
            v = weight_offset != 0.0f ? 0.1f * normal(rng) : 1.0f + 0.1f * normal(rng);
        }
        const ttnn::Tensor input = device_tensor(x, rows, cols);
        const ttnn::Tensor residual = device_tensor(r, rows, cols);
        const ttnn::Tensor weight = device_tensor(w, 1, cols);
        constexpr float eps = 1e-6f;

        const auto fused =
            ttnn::normalization::rms_norm_residual(input, residual, weight, eps, weight_offset, dtype, kMinScale);
        const auto plain = ttnn::normalization::rms_norm_residual(input, residual, weight, eps, weight_offset);
        const auto separate = ttnn::experimental::quantize_rows(plain[0], dtype, kMinScale);
        EXPECT_EQ(fused.size(), 4u);
        EXPECT_EQ(fused[2].dtype(), dtype);
        EXPECT_EQ(fused[3].logical_shape(), ttnn::Shape({rows, 1}));

        size_t wrong = 0;
        auto compare = [&](const auto& got, const auto& want, const std::string& what) {
            EXPECT_EQ(got.size(), want.size()) << what;
            for (size_t i = 0; i < got.size() && i < want.size(); ++i) {
                if (std::bit_cast<uint32_t>(static_cast<float>(got[i])) !=
                    std::bit_cast<uint32_t>(static_cast<float>(want[i]))) {
                    if (wrong++ < 8) {
                        ADD_FAILURE() << rows << "x" << cols << " " << dtype << " " << what << "[" << i / cols
                                      << "][" << i % cols << "] " << static_cast<float>(got[i])
                                      << " != " << static_cast<float>(want[i]);
                    }
                }
            }
        };
        compare(fused[0].cpu().to_vector<float>(), plain[0].cpu().to_vector<float>(), "norm");
        compare(fused[1].cpu().to_vector<float>(), plain[1].cpu().to_vector<float>(), "sum");
        compare(
            ttnn::bitcast(fused[2], DataType::UINT8).cpu().to_vector<uint8_t>(),
            ttnn::bitcast(separate[0], DataType::UINT8).cpu().to_vector<uint8_t>(),
            "q");
        compare(fused[3].cpu().to_vector<float>(), separate[1].cpu().to_vector<float>(), "scale");
        return wrong;
    }

    std::shared_ptr<MeshDevice> mesh_device_;
};

TEST_F(RMSNormResidualQuantizeTest, MatchesQuantizeRowsOfTheNorm) {
    // Widths: one tile (one core), 3 tiles, 32 tiles (32 cores of one), 160
    // (32 cores of 5) and 136 (24 cores of 4, 8 of 5: Qwen3.8's 4352 per chip).
    for (uint32_t rows : {1u, 2u, 3u, 4u, 5u, 13u, 17u, 32u}) {
        for (uint32_t cols : {32u, 96u, 1024u, 5120u, 4352u}) {
            EXPECT_EQ(check(rows, cols, DataType::FP8_E4M3, 0.0f, rows * 131 + cols, false), 0u) << rows << "x" << cols;
        }
    }
}

TEST_F(RMSNormResidualQuantizeTest, E5m2ZeroCenteredWeightsAndZeroRow) {
    for (uint32_t rows : {1u, 3u, 8u, 32u}) {
        EXPECT_EQ(check(rows, 5120, DataType::FP8_E5M2, 0.0f, rows, false), 0u) << rows;
        EXPECT_EQ(check(rows, 5120, DataType::FP8_E4M3, 1.0f, rows + 1, false), 0u) << rows;
        EXPECT_EQ(check(rows, 2048, DataType::FP8_E4M3, 1.0f, rows + 2, true), 0u) << rows;
    }
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
