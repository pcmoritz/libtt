// SPDX-License-Identifier: Apache-2.0
// The decode matmul's exact SwiGLU (ttnn::matmul_swiglu with `exact`, and
// ttnn::matmul_scaled with `swiglu`): PyTorch's SiluAndMul after a merged
// [gate | up] projection, as SGLang-TorchTPU's decode MLP runs it on TT. The
// result must equal the projection followed by ttnn::silu_and_mul's ops (the
// halves sliced, the gate cast to FP32, silu, cast back to BF16, times up) bit
// for bit: BF16 weights in either layout, FP8 weights with row, column and
// single column scales, one to 32 rows, and the fallback past one tile row.
// The shapes keep the unfused projection off the split over K, which sums in
// another order.
#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include <tt-metalium/mesh_device.hpp>

#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

class MatmulExactSwiGLUTest : public ::testing::Test {
protected:
    void SetUp() override { mesh_device_ = MeshDevice::create(MeshDeviceConfig(MeshShape{1, 1})); }

    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
    }

    // A TILE tensor in DRAM of normal values times `scale`, in `dtype` (float8 by
    // an exact typecast of FP32).
    ttnn::Tensor random(const ttnn::Shape& shape, DataType dtype, uint32_t seed, float scale, float offset = 0.0f) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        std::vector<float> values(shape.volume());
        for (float& v : values) {
            v = offset + scale * normal(rng);
        }
        const bool float8 = dtype == DataType::FP8_E4M3;
        const TensorSpec spec(
            shape, TensorLayout(float8 ? DataType::FLOAT32 : dtype, PageConfig(Layout::TILE), {}));
        const ttnn::Tensor t = ttnn::Tensor::from_vector(values, spec).to_device(mesh_device_.get());
        return float8 ? ttnn::typecast(t, dtype) : t;
    }

    // The compute config SGLang-TorchTPU's decode matmuls carry.
    static ttnn::DeviceComputeKernelConfig hifi4() {
        return ttnn::DeviceComputeKernelConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true};
    }

    size_t mismatches(const ttnn::Tensor& fused, const ttnn::Tensor& unfused, uint32_t M, uint32_t N) {
        EXPECT_EQ(fused.dtype(), DataType::BFLOAT16);
        EXPECT_EQ(fused.logical_shape(), ttnn::Shape({M, N}));
        const auto got = fused.cpu().to_vector<float>();
        const auto want = unfused.cpu().to_vector<float>();
        EXPECT_EQ(got.size(), want.size());
        size_t wrong = 0;
        for (size_t i = 0; i < got.size() && i < want.size(); ++i) {
            if (std::bit_cast<uint32_t>(got[i]) != std::bit_cast<uint32_t>(want[i]) && wrong++ < 5) {
                ADD_FAILURE() << M << "x" << N << " [" << i / N << ", " << i % N << "]: " << got[i]
                              << " != " << want[i];
            }
        }
        return wrong;
    }

    // A BF16 [M, K] activation times a BF16 [K, 2N] weight (or PyTorch's [2N, K]).
    size_t bf16_mismatches(uint32_t M, uint32_t K, uint32_t N, bool transpose_b) {
        const uint32_t seed = M * 131 + K * 7 + N + transpose_b;
        const ttnn::Tensor a = random(ttnn::Shape({M, K}), DataType::BFLOAT16, seed, 1.0f);
        const ttnn::Tensor b = random(
            transpose_b ? ttnn::Shape({2 * N, K}) : ttnn::Shape({K, 2 * N}), DataType::BFLOAT16, seed + 1, 0.05f);
        const ttnn::Tensor fused = ttnn::matmul_swiglu(
            a, b, std::nullopt, DataType::BFLOAT16, hifi4(), transpose_b, /*exact=*/true);
        const ttnn::Tensor product = ttnn::matmul(
            a,
            b,
            /*transpose_a=*/false,
            transpose_b,
            /*memory_config=*/std::nullopt,
            DataType::BFLOAT16,
            /*program_config=*/std::nullopt,
            /*activation=*/std::nullopt,
            hifi4());
        return mismatches(fused, ttnn::silu_and_mul(product), M, N);
    }

    // The column scales: [1, 2N], the same scales as [2N, 1] (the official FP8
    // checkpoints' per-channel scales), or one scale.
    enum class Col { Row, Column, Scalar };

    // An FP8 [M, K] activation times an FP8 [K, 2N] weight, dequantized by FP32
    // row and column scales to BF16 (SGLang-TorchTPU's FP8 linears).
    size_t fp8_mismatches(uint32_t M, uint32_t K, uint32_t N, Col col) {
        const uint32_t seed = M * 131 + K * 7 + N + 2 * static_cast<uint32_t>(col);
        const ttnn::Tensor a = random(ttnn::Shape({M, K}), DataType::FP8_E4M3, seed, 40.0f);
        const ttnn::Tensor b = random(ttnn::Shape({K, 2 * N}), DataType::FP8_E4M3, seed + 1, 30.0f);
        const ttnn::Tensor row_scale = random(ttnn::Shape({M, 1}), DataType::FLOAT32, seed + 2, 1e-3f, 5e-3f);
        const ttnn::Tensor col_scale = random(
            col == Col::Scalar   ? ttnn::Shape({1, 1})
            : col == Col::Column ? ttnn::Shape({2 * N, 1})
                                 : ttnn::Shape({1, 2 * N}),
            DataType::FLOAT32,
            seed + 3,
            1e-3f,
            4e-3f);
        const ttnn::Tensor fused = ttnn::matmul_scaled(
            a, b, row_scale, col_scale, false, std::nullopt, DataType::BFLOAT16, hifi4(), /*swiglu=*/true);
        const ttnn::Tensor product =
            ttnn::matmul_scaled(a, b, row_scale, col_scale, false, std::nullopt, DataType::BFLOAT16, hifi4());
        return mismatches(fused, ttnn::silu_and_mul(product), M, N);
    }

    std::shared_ptr<MeshDevice> mesh_device_;
};

// Qwen3-8B's gate and up projection on four chips and one, for one and four
// tokens, in PyTorch's weight layout and transposed; then more rows.
TEST_F(MatmulExactSwiGLUTest, BFloat16) {
    for (const auto [K, N] : {std::pair{4096u, 3072u}, {4096u, 12288u}}) {
        for (uint32_t M : {1u, 4u}) {
            for (bool transpose_b : {true, false}) {
                EXPECT_EQ(bf16_mismatches(M, K, N, transpose_b), 0u)
                    << M << "x" << K << "x" << N << (transpose_b ? " transposed" : "");
            }
        }
    }
    for (uint32_t M : {2u, 13u, 32u}) {
        EXPECT_EQ(bf16_mismatches(M, 1024, 2048, true), 0u) << M;
    }
}

TEST_F(MatmulExactSwiGLUTest, Float8Scaled) {
    for (const auto [K, N] : {std::pair{4096u, 3072u}, {4096u, 12288u}}) {
        for (uint32_t M : {1u, 4u}) {
            for (Col col : {Col::Row, Col::Column, Col::Scalar}) {
                EXPECT_EQ(fp8_mismatches(M, K, N, col), 0u)
                    << M << "x" << K << "x" << N << " col " << static_cast<int>(col);
            }
        }
    }
    for (uint32_t M : {2u, 13u, 32u}) {
        EXPECT_EQ(fp8_mismatches(M, 1024, 2048, Col::Column), 0u) << M;
    }
}

// Past one tile row of activations the decode program does not apply.
TEST_F(MatmulExactSwiGLUTest, FallsBackPastOneTileRow) {
    EXPECT_EQ(bf16_mismatches(64, 1024, 2048, true), 0u);
    EXPECT_EQ(fp8_mismatches(64, 1024, 2048, Col::Column), 0u);
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
