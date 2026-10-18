// SPDX-License-Identifier: Apache-2.0

// A decode matmul with few output tiles splits K over grid rows: each core
// reduces one share of K for one output tile, and the core of the first share
// adds the others' FP32 partial sums. These matmuls must match a reference,
// whether the split comes from the automatic config or an explicit one.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include <tt-metalium/mesh_device.hpp>

#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

using ttnn::operations::matmul::MatmulDecodeProgramConfig;

class MatmulDecodeSplitKTest : public ::testing::Test {
protected:
    void SetUp() override { mesh_device_ = MeshDevice::create(MeshDeviceConfig(MeshShape{1, 1})); }

    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
    }

    // A TILE tensor in DRAM; `values` come back rounded to `dtype`.
    ttnn::Tensor to_device(std::vector<float>& values, const ttnn::Shape& shape, DataType dtype) {
        const TensorSpec spec(shape, TensorLayout(dtype, PageConfig(Layout::TILE), {}));
        ttnn::Tensor host = ttnn::Tensor::from_vector(values, spec);
        values = host.to_vector<float>();
        return host.to_device(mesh_device_.get());
    }

    // Relative RMS error of a [1, K] x [K, N] matmul (b stored [N, K] when
    // transposed) against a double-precision reference.
    double matmul_error(
        uint32_t K,
        uint32_t N,
        bool transpose_b,
        DataType b_dtype,
        DataType out_dtype,
        const std::optional<MatmulDecodeProgramConfig>& config) {
        std::mt19937 rng(K * 31 + N);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        std::vector<float> a(K), b(K * N);
        for (float& v : a) {
            v = normal(rng);
        }
        for (float& v : b) {
            v = normal(rng) / std::sqrt(static_cast<float>(K));
        }
        const ttnn::Tensor a_device = to_device(a, ttnn::Shape({1, K}), DataType::BFLOAT16);
        const ttnn::Tensor b_device =
            to_device(b, transpose_b ? ttnn::Shape({N, K}) : ttnn::Shape({K, N}), b_dtype);
        const auto program_config = config.has_value()
                                        ? std::optional<const ttnn::operations::matmul::MatmulProgramConfig>(*config)
                                        : std::nullopt;
        const ttnn::Tensor output = ttnn::matmul(
            a_device, b_device, /*transpose_a=*/false, transpose_b, /*memory_config=*/std::nullopt, out_dtype,
            program_config);

        const std::vector<float> actual = output.cpu().to_vector<float>();
        EXPECT_EQ(actual.size(), N);
        double error = 0.0, norm = 0.0;
        for (uint32_t n = 0; n < N && n < actual.size(); ++n) {
            double expected = 0.0;
            for (uint32_t k = 0; k < K; ++k) {
                expected += static_cast<double>(a[k]) * (transpose_b ? b[n * K + k] : b[k * N + n]);
            }
            error += (actual[n] - expected) * (actual[n] - expected);
            norm += expected * expected;
        }
        return std::sqrt(error / norm);
    }

    MatmulDecodeProgramConfig split_config(uint32_t k_splits, uint32_t in0_block_w) const {
        return MatmulDecodeProgramConfig{
            .compute_with_storage_grid_size = mesh_device_->compute_with_storage_grid_size(),
            .in0_block_w = in0_block_w,
            .fuse_swiglu = false,
            .k_splits = k_splits,
        };
    }

    std::shared_ptr<MeshDevice> mesh_device_;
};

// The hyper-connection down projection of Qwen3.8-Flash-Next decode: 11
// output tiles, 320 K tiles split over ten shares.
TEST_F(MatmulDecodeSplitKTest, ExplicitSplitMatchesReference) {
    EXPECT_LT(matmul_error(10240, 324, false, DataType::BFLOAT16, DataType::BFLOAT16, split_config(10, 8)), 0.03);
}

TEST_F(MatmulDecodeSplitKTest, AutomaticConfigSplitsNarrowMatmuls) {
    // BFP8 weights as the model stores them, a logical N that ends inside a
    // tile, an output wider than one grid row (two rows per share), and K
    // shares of a few tiles.
    EXPECT_LT(matmul_error(10240, 324, false, DataType::BFLOAT8_B, DataType::BFLOAT16, std::nullopt), 0.03);
    EXPECT_LT(matmul_error(2560, 321, false, DataType::BFLOAT16, DataType::BFLOAT16, std::nullopt), 0.03);
    EXPECT_LT(matmul_error(2560, 512, false, DataType::BFLOAT16, DataType::BFLOAT16, std::nullopt), 0.03);
    EXPECT_LT(matmul_error(1024, 32, false, DataType::BFLOAT16, DataType::BFLOAT16, std::nullopt), 0.03);
}

TEST_F(MatmulDecodeSplitKTest, TransposedWeight) {
    EXPECT_LT(matmul_error(4096, 96, true, DataType::BFLOAT16, DataType::BFLOAT16, split_config(8, 4)), 0.03);
}

TEST_F(MatmulDecodeSplitKTest, FP32OutputKeepsPartialSumPrecision) {
    // Integers small enough that FP32 sums are exact, so any rounding of a
    // partial sum to a narrower type shows.
    constexpr uint32_t K = 8192;
    constexpr uint32_t N = 64;
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> small(-8, 8);
    std::vector<float> a(K), b(K * N);
    for (float& v : a) {
        v = small(rng);
    }
    for (float& v : b) {
        v = small(rng);
    }
    const ttnn::Tensor a_device = to_device(a, ttnn::Shape({1, K}), DataType::BFLOAT16);
    const ttnn::Tensor b_device = to_device(b, ttnn::Shape({K, N}), DataType::BFLOAT16);
    const ttnn::Tensor output = ttnn::matmul(
        a_device, b_device, false, false, std::nullopt, DataType::FLOAT32,
        std::optional<const ttnn::operations::matmul::MatmulProgramConfig>(split_config(8, 8)));
    const std::vector<float> actual = output.cpu().to_vector<float>();
    ASSERT_EQ(actual.size(), N);
    for (uint32_t n = 0; n < N; ++n) {
        double expected = 0.0;
        for (uint32_t k = 0; k < K; ++k) {
            expected += static_cast<double>(a[k]) * b[k * N + n];
        }
        EXPECT_EQ(actual[n], expected) << "output " << n;
    }
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
