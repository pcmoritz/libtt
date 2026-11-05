// SPDX-License-Identifier: Apache-2.0
// ttnn::matmul_scaled: a one-tile-row matmul whose FP32 product is multiplied
// by a scale per row and then by a scale per column (or one scale), rounded to
// the output type, as SGLang's FP8 linears dequantize torch._scaled_mm. The
// decode program does it in its epilogue (or its split over K, in the owner's);
// the result must equal ttnn.matmul then two ttnn.multiply ops bit for bit,
// for FP8 and BF16 operands, either weight layout, per-column scales as a row or
// a column and single column scales, BF16 and FP32 outputs, and fall back to
// those ops past the decode program's reach.
#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include <tt-metalium/mesh_device.hpp>

#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

class MatmulScaledTest : public ::testing::Test {
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

    // The column scales: [1, N], the same scales as [N, 1] (as the official FP8
    // checkpoints store per-channel scales), or one scale.
    enum class Col { Row, Column, Scalar };

    // Mismatching elements between matmul_scaled and the unfused ops.
    size_t mismatches(
        uint32_t M, uint32_t K, uint32_t N, DataType in_dtype, bool transpose_b, Col col, DataType out_dtype) {
        const uint32_t seed = M * 131 + K * 7 + N + transpose_b + 2 * (col == Col::Scalar);
        const ttnn::Tensor a = random(ttnn::Shape({M, K}), in_dtype, seed, 40.0f);
        const ttnn::Tensor b =
            random(transpose_b ? ttnn::Shape({N, K}) : ttnn::Shape({K, N}), in_dtype, seed + 1, 30.0f);
        // Positive scales of varied magnitudes.
        const ttnn::Tensor row_scale = random(ttnn::Shape({M, 1}), DataType::FLOAT32, seed + 2, 1e-3f, 5e-3f);
        auto col_scale_of = [&](const ttnn::Shape& shape) {
            return random(shape, DataType::FLOAT32, seed + 3, 1e-4f, 4e-4f);
        };
        const ttnn::Tensor col_scale = col_scale_of(col == Col::Scalar ? ttnn::Shape({1, 1}) : ttnn::Shape({1, N}));

        const ttnn::Tensor fused = ttnn::matmul_scaled(
            a,
            b,
            row_scale,
            col == Col::Column ? col_scale_of(ttnn::Shape({N, 1})) : col_scale,
            transpose_b,
            std::nullopt,
            out_dtype);
        const ttnn::Tensor product = ttnn::matmul(
            a, b, /*transpose_a=*/false, transpose_b, /*memory_config=*/std::nullopt, DataType::FLOAT32);
        const ttnn::Tensor unfused = ttnn::multiply(ttnn::multiply(product, row_scale, DataType::FLOAT32), col_scale, out_dtype);
        EXPECT_EQ(fused.dtype(), out_dtype);
        EXPECT_EQ(fused.logical_shape(), ttnn::Shape({M, N}));
        const auto got = fused.cpu().to_vector<float>();
        const auto want = unfused.cpu().to_vector<float>();
        EXPECT_EQ(got.size(), want.size());
        size_t wrong = 0;
        for (size_t i = 0; i < got.size() && i < want.size(); ++i) {
            if (std::bit_cast<uint32_t>(got[i]) != std::bit_cast<uint32_t>(want[i]) && wrong++ < 5) {
                ADD_FAILURE() << M << "x" << K << "x" << N << " [" << i / N << ", " << i % N << "]: " << got[i]
                              << " != " << want[i];
            }
        }
        return wrong;
    }

    std::shared_ptr<MeshDevice> mesh_device_;
};

// Qwen3-8B's FP8 linears on four chips and one: qkv, o, gate_up and down; one
// and four tokens; per-channel (as rows, and as the official checkpoint's
// columns) and per-tensor (online quantization) weight scales.
TEST_F(MatmulScaledTest, Float8DecodeShapes) {
    for (const auto [K, N] : {std::pair{4096u, 1536u}, {1024u, 4096u}, {4096u, 6144u}, {3072u, 4096u}, {12288u, 4096u}}) {
        for (uint32_t M : {1u, 4u}) {
            for (Col col : {Col::Row, Col::Column, Col::Scalar}) {
                EXPECT_EQ(mismatches(M, K, N, DataType::FP8_E4M3, false, col, DataType::BFLOAT16), 0u)
                    << M << "x" << K << "x" << N << " col " << static_cast<int>(col);
            }
        }
    }
}

TEST_F(MatmulScaledTest, WeightLayoutsRowsAndOutputs) {
    for (bool transpose_b : {false, true}) {
        for (uint32_t M : {1u, 2u, 13u, 16u, 32u}) {
            EXPECT_EQ(mismatches(M, 2048, 3072, DataType::FP8_E4M3, transpose_b, Col::Row, DataType::BFLOAT16), 0u)
                << M << (transpose_b ? " transposed" : "");
            EXPECT_EQ(mismatches(M, 1024, 256, DataType::FP8_E4M3, transpose_b, Col::Scalar, DataType::FLOAT32), 0u)
                << M << (transpose_b ? " transposed" : "");
            EXPECT_EQ(mismatches(M, 2048, 1024, DataType::FP8_E4M3, transpose_b, Col::Column, DataType::FLOAT32), 0u)
                << M << (transpose_b ? " transposed" : "");
        }
    }
    EXPECT_EQ(mismatches(1, 2048, 1024, DataType::BFLOAT16, false, Col::Row, DataType::BFLOAT16), 0u);
    EXPECT_EQ(mismatches(4, 2048, 1024, DataType::BFLOAT16, true, Col::Scalar, DataType::FLOAT32), 0u);
    EXPECT_EQ(mismatches(2, 2048, 1024, DataType::BFLOAT16, true, Col::Column, DataType::BFLOAT16), 0u);
}

// Past one tile row of activations the decode program does not apply.
TEST_F(MatmulScaledTest, FallsBackPastOneTileRow) {
    EXPECT_EQ(mismatches(64, 1024, 512, DataType::FP8_E4M3, false, Col::Row, DataType::BFLOAT16), 0u);
    EXPECT_EQ(mismatches(64, 1024, 512, DataType::FP8_E4M3, false, Col::Column, DataType::BFLOAT16), 0u);
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
