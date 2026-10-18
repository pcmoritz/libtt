// SPDX-License-Identifier: Apache-2.0

// An H sum or mean of a TILE tensor whose rows past the logical height hold NaN: the reduce reader zeroes those
// rows of the last tile row as it reads them, so the result is the reduction of the data rows alone without a
// fill_pad program first.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include <tt-metalium/mesh_device.hpp>

#include "ttnn/operations/experimental/reshape/view.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

class ReduceZeroPaddingRowsTest : public ::testing::Test {
protected:
    void SetUp() override { mesh_device_ = MeshDevice::create(MeshDeviceConfig(MeshShape{1, 1})); }

    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
    }

    // Sums (or averages) a [1, 1, rows, width] view of a [1, 1, round_up(rows, 32), width] tensor over H, its
    // padding rows NaN, and checks every column against the reduction of the data rows on the host.
    void check(uint32_t rows, uint32_t width, DataType dtype, bool mean) {
        const uint32_t padded_rows = (rows + 31) / 32 * 32;
        std::mt19937 rng(rows * 131 + width + (mean ? 1 : 0));
        std::normal_distribution<float> normal(0.0f, 1.0f);
        std::vector<float> values(static_cast<size_t>(padded_rows) * width);
        for (uint32_t r = 0; r < padded_rows; ++r) {
            for (uint32_t c = 0; c < width; ++c) {
                values[r * width + c] = r < rows ? normal(rng) : std::numeric_limits<float>::quiet_NaN();
            }
        }
        const TensorSpec spec(
            ttnn::Shape({1, 1, padded_rows, width}), TensorLayout(dtype, PageConfig(Layout::TILE), {}));
        ttnn::Tensor host = ttnn::Tensor::from_vector(values, spec);
        values = host.to_vector<float>();
        ttnn::Tensor full = host.to_device(mesh_device_.get());
        ttnn::Tensor input =
            ttnn::experimental::view(full, ttnn::Shape({1, 1, rows, width}), ttnn::Shape({1, 1, padded_rows, width}));
        ASSERT_EQ(input.logical_shape(), ttnn::Shape({1, 1, rows, width}));
        ttnn::Tensor output = mean ? ttnn::mean(input, -2, /*keepdim=*/true) : ttnn::sum(input, -2, /*keepdim=*/true);
        ASSERT_EQ(output.logical_shape(), ttnn::Shape({1, 1, 1, width}));
        const std::vector<float> got = output.cpu().to_vector<float>();
        ASSERT_EQ(got.size(), width);
        const float rel = dtype == DataType::BFLOAT16 ? 1.0f / 64 : 1.0f / 1024;
        for (uint32_t c = 0; c < width; ++c) {
            double want = 0;
            for (uint32_t r = 0; r < rows; ++r) {
                want += values[r * width + c];
            }
            if (mean) {
                want /= rows;
            }
            ASSERT_TRUE(std::isfinite(got[c])) << "rows " << rows << " column " << c;
            EXPECT_NEAR(got[c], want, rel * (1.0 + std::abs(want))) << "rows " << rows << " column " << c;
        }
    }

    std::shared_ptr<MeshDevice> mesh_device_;
};

TEST_F(ReduceZeroPaddingRowsTest, Bfloat16) {
    for (const bool mean : {true, false}) {
        check(4, 2560, DataType::BFLOAT16, mean);   // rows 4-15 of the top faces, all of the bottom ones
        check(20, 256, DataType::BFLOAT16, mean);   // rows 4-15 of the bottom faces
        check(40, 128, DataType::BFLOAT16, mean);   // the second of two tile rows
        check(32, 128, DataType::BFLOAT16, mean);   // no padding
    }
}

TEST_F(ReduceZeroPaddingRowsTest, Float32) {
    for (const bool mean : {true, false}) {
        check(4, 2560, DataType::FLOAT32, mean);
        check(20, 256, DataType::FLOAT32, mean);
        check(40, 128, DataType::FLOAT32, mean);
    }
}

// Logical heights that share a padded shape get their own programs.
TEST_F(ReduceZeroPaddingRowsTest, HeightsSharingAPaddedShape) {
    for (const uint32_t rows : {4u, 8u, 1u, 31u, 4u}) {
        check(rows, 512, DataType::BFLOAT16, /*mean=*/true);
    }
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
