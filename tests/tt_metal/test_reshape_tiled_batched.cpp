// SPDX-License-Identifier: Apache-2.0

// The tiled reshape reads a batch of mapping pages and keeps several input
// tiles in flight per core. Reshapes with one or many input tiles per output
// page, many output pages per core and odd shapes must still move every
// element to its place.

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include <tt-metalium/mesh_device.hpp>

#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

class ReshapeTiledBatchedTest : public ::testing::Test {
protected:
    void SetUp() override { mesh_device_ = MeshDevice::create(MeshDeviceConfig(MeshShape{1, 1})); }

    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
    }

    // Reshapes a TILE tensor of distinct small integers (exact in BF16) on the
    // device; the row-major order of the elements must not change.
    void expect_reshape(const ttnn::Shape& from, const ttnn::Shape& to, DataType dtype) {
        std::vector<float> values(from.volume());
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = static_cast<float>(static_cast<int>(i % 251) - 125);
        }
        const TensorSpec spec(from, TensorLayout(dtype, PageConfig(Layout::TILE), {}));
        const ttnn::Tensor input = ttnn::Tensor::from_vector(values, spec).to_device(mesh_device_.get());
        const ttnn::Tensor output = ttnn::reshape(input, to);
        ASSERT_EQ(output.logical_shape(), to);
        const std::vector<float> actual = output.cpu().to_vector<float>();
        ASSERT_EQ(actual.size(), values.size());
        size_t mismatches = 0;
        for (size_t i = 0; i < values.size(); ++i) {
            mismatches += actual[i] != values[i];
        }
        EXPECT_EQ(mismatches, 0u) << from << " -> " << to << " " << dtype;
    }

    std::shared_ptr<MeshDevice> mesh_device_;
};

TEST_F(ReshapeTiledBatchedTest, OneTilePerOutputPage) {
    expect_reshape(ttnn::Shape({1, 4, 2560}), ttnn::Shape({1, 10240}), DataType::BFLOAT16);
    expect_reshape(ttnn::Shape({1, 12, 128}), ttnn::Shape({1, 1536}), DataType::BFLOAT16);
}

TEST_F(ReshapeTiledBatchedTest, SeveralTilesPerOutputPage) {
    expect_reshape(ttnn::Shape({1, 10240}), ttnn::Shape({1, 4, 2560}), DataType::BFLOAT16);
    expect_reshape(ttnn::Shape({1, 10240}), ttnn::Shape({1, 4, 2560}), DataType::FLOAT32);
    expect_reshape(ttnn::Shape({1, 2560}), ttnn::Shape({1, 2560, 1}), DataType::BFLOAT16);
    expect_reshape(ttnn::Shape({1, 4}), ttnn::Shape({1, 4, 1}), DataType::BFLOAT16);
}

TEST_F(ReshapeTiledBatchedTest, ManyOutputPagesPerCore) {
    // More output pages per core than mapping entries and input tiles in flight.
    expect_reshape(ttnn::Shape({64, 4096}), ttnn::Shape({4, 16, 4096}), DataType::BFLOAT16);
    expect_reshape(ttnn::Shape({96, 2560}), ttnn::Shape({3, 320, 256}), DataType::BFLOAT16);
    expect_reshape(ttnn::Shape({3, 50, 96}), ttnn::Shape({150, 96}), DataType::FLOAT32);
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
