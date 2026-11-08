// SPDX-License-Identifier: Apache-2.0

// A row-major reshape whose rows are too wide to stage in L1 (Qwen3.8's
// [4, 248320] FP32 logits as one row, as torch_tpu's view inverse makes them)
// streams each piece of a source row to its offset in the destination row.
// The values must come out in order, whichever way the rows widen or narrow.

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include <tt-metalium/mesh_device.hpp>

#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

class ReshapeRowMajorWideTest : public ::testing::Test {
protected:
    void SetUp() override { mesh_device_ = MeshDevice::create(MeshDeviceConfig(MeshShape{1, 1})); }

    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
    }

    // Reshapes a row-major [rows, width] tensor of 0, 1, 2, ... to `shape`
    // and checks every value.
    void check(uint32_t rows, uint32_t width, const ttnn::Shape& shape, DataType dtype) {
        std::vector<float> values(static_cast<size_t>(rows) * width);
        // Exact in BF16 up to 256; wrap so every value is.
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = dtype == DataType::BFLOAT16 ? static_cast<float>(i % 251) : static_cast<float>(i);
        }
        const TensorSpec spec(ttnn::Shape({rows, width}), TensorLayout(dtype, PageConfig(Layout::ROW_MAJOR), {}));
        const ttnn::Tensor input = ttnn::Tensor::from_vector(values, spec).to_device(mesh_device_.get());
        const ttnn::Tensor output = ttnn::reshape(input, shape);
        ASSERT_EQ(output.layout(), Layout::ROW_MAJOR);
        ASSERT_EQ(output.logical_shape(), shape);
        const std::vector<float> actual = output.cpu().to_vector<float>();
        ASSERT_EQ(actual.size(), values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            ASSERT_EQ(actual[i], values[i]) << "element " << i;
        }
    }

    std::shared_ptr<MeshDevice> mesh_device_;
};

TEST_F(ReshapeRowMajorWideTest, RowsJoinIntoOneWideRow) {
    check(4, 248320, ttnn::Shape({1, 1, 993280}), DataType::FLOAT32);
    check(4, 248320, ttnn::Shape({993280}), DataType::FLOAT32);
    check(12, 248320, ttnn::Shape({1, 2979840}), DataType::BFLOAT16);
}

TEST_F(ReshapeRowMajorWideTest, WideRowSplitsAndRegroups) {
    check(1, 993280, ttnn::Shape({4, 248320}), DataType::FLOAT32);
    check(4, 248320, ttnn::Shape({2, 496640}), DataType::FLOAT32);
}

TEST_F(ReshapeRowMajorWideTest, NarrowRowsKeepStaging) {
    check(64, 1024, ttnn::Shape({1, 65536}), DataType::FLOAT32);
    check(8, 4096, ttnn::Shape({32, 1024}), DataType::BFLOAT16);
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
