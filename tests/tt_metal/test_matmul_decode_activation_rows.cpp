// SPDX-License-Identifier: Apache-2.0

// Unsplit decode matmuls of an activation of up to 16 rows move only the rows it has: the activation and the output
// travel as tiles of 2, 4, 8 or 16 rows, and the output's padding rows are not written. Every logical output row
// must match a reference, for either output type; matmuls split over K (full tiles) must too.

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

class MatmulDecodeActivationRowsTest : public ::testing::Test {
protected:
    void SetUp() override { mesh_device_ = MeshDevice::create(MeshDeviceConfig(MeshShape{1, 1})); }

    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
    }

    ttnn::Tensor to_device(std::vector<float>& values, const ttnn::Shape& shape, DataType dtype) {
        const TensorSpec spec(shape, TensorLayout(dtype, PageConfig(Layout::TILE), {}));
        ttnn::Tensor host = ttnn::Tensor::from_vector(values, spec);
        values = host.to_vector<float>();
        return host.to_device(mesh_device_.get());
    }

    // Worst relative RMS error over the rows of an [M, K] x [K, N] matmul (BFP8 weights, automatic config)
    // against a double-precision reference.
    double matmul_error(uint32_t M, uint32_t K, uint32_t N, DataType out_dtype) {
        std::mt19937 rng(M * 7919 + K * 31 + N);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        std::vector<float> a(M * K), b(K * N);
        for (float& v : a) {
            v = normal(rng);
        }
        for (float& v : b) {
            v = normal(rng) / std::sqrt(static_cast<float>(K));
        }
        const ttnn::Tensor a_device = to_device(a, ttnn::Shape({M, K}), DataType::BFLOAT16);
        const ttnn::Tensor b_device = to_device(b, ttnn::Shape({K, N}), DataType::BFLOAT8_B);
        const ttnn::Tensor output = ttnn::matmul(
            a_device, b_device, /*transpose_a=*/false, /*transpose_b=*/false, /*memory_config=*/std::nullopt,
            out_dtype);
        const std::vector<float> actual = output.cpu().to_vector<float>();
        EXPECT_EQ(actual.size(), static_cast<size_t>(M) * N);
        double worst = 0.0;
        for (uint32_t m = 0; m < M; ++m) {
            double error = 0.0, norm = 0.0;
            for (uint32_t n = 0; n < N && m * N + n < actual.size(); ++n) {
                double expected = 0.0;
                for (uint32_t k = 0; k < K; ++k) {
                    expected += static_cast<double>(a[m * K + k]) * b[k * N + n];
                }
                error += (actual[m * N + n] - expected) * (actual[m * N + n] - expected);
                norm += expected * expected;
            }
            worst = std::max(worst, std::sqrt(error / norm));
        }
        return worst;
    }

    std::shared_ptr<MeshDevice> mesh_device_;
};

// The decode matmuls of Qwen3.8-Flash-Next at one row: hyper-connection down (split over K) and up, GDN qkv, the
// PLE projection.
TEST_F(MatmulDecodeActivationRowsTest, OneRow) {
    EXPECT_LT(matmul_error(1, 10240, 324, DataType::BFLOAT16), 0.03);
    EXPECT_LT(matmul_error(1, 320, 10240, DataType::BFLOAT16), 0.03);
    EXPECT_LT(matmul_error(1, 2560, 4120, DataType::BFLOAT16), 0.03);
    EXPECT_LT(matmul_error(1, 2560, 12800, DataType::BFLOAT16), 0.03);
}

TEST_F(MatmulDecodeActivationRowsTest, Fp32Output) {
    EXPECT_LT(matmul_error(1, 2560, 4096, DataType::FLOAT32), 0.03);
    EXPECT_LT(matmul_error(3, 2560, 3584, DataType::FLOAT32), 0.03);
}

// Activation rows that fill a 2-, 4-, 8- or 16-row tile, or need the full tile.
TEST_F(MatmulDecodeActivationRowsTest, SeveralRows) {
    for (const uint32_t M : {2u, 3u, 5u, 9u, 16u, 17u}) {
        EXPECT_LT(matmul_error(M, 2560, 3584, DataType::BFLOAT16), 0.03) << M << " rows";
        EXPECT_LT(matmul_error(M, 10240, 324, DataType::BFLOAT16), 0.03) << M << " rows";
    }
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
