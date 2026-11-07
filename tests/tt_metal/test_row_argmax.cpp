// SPDX-License-Identifier: Apache-2.0

// ttnn::experimental::row_argmax: the argmax of each row of a tiled
// [..., R, N] tensor against a host reference, with ties to the first index,
// BFLOAT16 and FLOAT32, rows past the top faces, logits-sized rows over many
// cores, and floors (the argmax of max(x, floor): index 0 where no element
// exceeds it).

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "ttnn/operations/experimental/row_argmax/row_argmax.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

class RowArgmaxTest : public ::testing::Test {
protected:
    void SetUp() override { mesh_device_ = MeshDevice::create(MeshDeviceConfig(MeshShape{1, 1})); }

    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
    }

    // Values rounded to dtype, with every 7th one a copy of an earlier value
    // so that ties occur, and the expected first index of each row's maximum.
    void check(const ttnn::Shape& shape, DataType dtype, std::optional<float> floor, uint32_t seed, float scale = 1.0f) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        const uint32_t width = shape[-1];
        const uint32_t rows = shape.volume() / width;
        std::vector<float> values(shape.volume());
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = i % 7 == 6 && i % width >= 6 ? values[i - 6] : scale * normal(rng);
            if (dtype == DataType::BFLOAT16) {
                values[i] = static_cast<float>(bfloat16(values[i]));
            }
        }
        const TensorSpec spec(shape, TensorLayout(dtype, PageConfig(Layout::TILE), {}));
        const auto input = ttnn::Tensor::from_vector(values, spec).to_device(mesh_device_.get());
        const auto output = ttnn::experimental::row_argmax(input, /*keep_dim=*/false, floor, std::nullopt);
        const auto got = output.cpu().to_vector<uint32_t>();
        ASSERT_EQ(got.size(), rows);
        for (uint32_t row = 0; row < rows; ++row) {
            uint32_t want = 0;
            float best = floor.value_or(-std::numeric_limits<float>::infinity());
            bool found = false;
            for (uint32_t column = 0; column < width; ++column) {
                const float value = values[row * width + column];
                if (value > best || (!found && !floor.has_value() && value >= best)) {
                    best = value;
                    want = column;
                    found = true;
                }
            }
            EXPECT_EQ(got[row], want) << shape << " " << dtype << " row " << row;
        }
    }

    std::shared_ptr<MeshDevice> mesh_device_;
};

TEST_F(RowArgmaxTest, MatchesReference) {
    uint32_t seed = 1;
    for (DataType dtype : {DataType::FLOAT32, DataType::BFLOAT16}) {
        check(ttnn::Shape({1, 64}), dtype, std::nullopt, seed++);
        check(ttnn::Shape({1, 4096}), dtype, std::nullopt, seed++);
        check(ttnn::Shape({4, 2048}), dtype, std::nullopt, seed++);
        check(ttnn::Shape({1, 1, 3, 96}), dtype, std::nullopt, seed++);
        check(ttnn::Shape({20, 160}), dtype, std::nullopt, seed++);
        check(ttnn::Shape({32, 64}), dtype, std::nullopt, seed++);
    }
    // Qwen3.8's and Qwen3-8B's logits rows, on every core.
    check(ttnn::Shape({1, 248320}), DataType::FLOAT32, std::nullopt, seed++);
    check(ttnn::Shape({4, 151936}), DataType::FLOAT32, std::nullopt, seed++);
}

TEST_F(RowArgmaxTest, Floor) {
    // torch_tpu's argmax reduces from the lowest finite float with index 0.
    const float lowest = std::numeric_limits<float>::lowest();
    check(ttnn::Shape({1, 248320}), DataType::FLOAT32, lowest, 11);
    check(ttnn::Shape({2, 1024}), DataType::BFLOAT16, lowest, 12);
    // Rows whose elements all lie below the floor give index 0; rows with some
    // above it their first maximum.
    check(ttnn::Shape({4, 512}), DataType::FLOAT32, 3.0f, 13);
    check(ttnn::Shape({4, 512}), DataType::FLOAT32, 0.5f, 14, 0.25f);
    check(ttnn::Shape({1, 4096}), DataType::FLOAT32, 100.0f, 15);
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
