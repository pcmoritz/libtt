// SPDX-License-Identifier: Apache-2.0

// A tile-aligned prefix of the last dim of an interleaved TILE tensor of one tile row is a view of the input:
// same buffer, no program. Other slices still copy. Either way the values are the input's.

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include <tt-metalium/mesh_device.hpp>

#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

class SlicePrefixViewTest : public ::testing::Test {
protected:
    void SetUp() override { mesh_device_ = MeshDevice::create(MeshDeviceConfig(MeshShape{1, 1})); }

    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
    }

    // Slices [rows, width] to its first `end` columns; returns whether the result shares the input's buffer.
    bool slice_matches(uint32_t rows, uint32_t width, uint32_t end, bool deallocate_input) {
        std::mt19937 rng(rows * 7 + width + end);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        std::vector<float> values(static_cast<size_t>(rows) * width);
        for (float& v : values) {
            v = normal(rng);
        }
        const TensorSpec spec(
            ttnn::Shape({rows, width}), TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), {}));
        ttnn::Tensor host = ttnn::Tensor::from_vector(values, spec);
        values = host.to_vector<float>();
        ttnn::Tensor input = host.to_device(mesh_device_.get());
        const ttsl::SmallVector<uint32_t> begins = {0, 0}, ends = {rows, end}, step = {1, 1};
        ttnn::Tensor output = ttnn::slice(input, begins, ends, step);
        const bool view = output.buffer()->address() == input.buffer()->address();
        // The view's buffer holds just its pages, so ops that size their work by the buffer stay inside it.
        EXPECT_EQ(output.buffer()->num_pages(), output.padded_shape().volume() / (32 * 32));
        if (deallocate_input) {
            input.deallocate(/*force=*/false);
        }
        EXPECT_EQ(output.logical_shape(), ttnn::Shape({rows, end}));
        const std::vector<float> got = output.cpu().to_vector<float>();
        EXPECT_EQ(got.size(), static_cast<size_t>(rows) * end);
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t c = 0; c < end; ++c) {
                EXPECT_EQ(got[r * end + c], values[r * width + c]) << "at " << r << ", " << c;
            }
        }
        return view;
    }

    std::shared_ptr<MeshDevice> mesh_device_;
};

TEST_F(SlicePrefixViewTest, OneTileRowPrefixIsAView) {
    EXPECT_TRUE(slice_matches(1, 4120, 4096, false));
    EXPECT_TRUE(slice_matches(4, 324, 320, false));
    // The view keeps the input's buffer alive.
    EXPECT_TRUE(slice_matches(1, 4120, 4096, true));
}

// typecast runs over its input buffer's pages: of a view, only the view's. A tensor allocated after it must keep its
// values.
TEST_F(SlicePrefixViewTest, TypecastOfAView) {
    std::mt19937 rng(5);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    auto make = [&](uint32_t width, float offset) {
        std::vector<float> values(width);
        for (float& v : values) {
            v = normal(rng) + offset;
        }
        const TensorSpec spec(ttnn::Shape({1, width}), TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), {}));
        return ttnn::Tensor::from_vector(values, spec);
    };
    ttnn::Tensor host = make(12800, 0.0f);
    const std::vector<float> values = host.to_vector<float>();
    ttnn::Tensor input = host.to_device(mesh_device_.get());
    const ttsl::SmallVector<uint32_t> begins = {0, 0}, ends = {1, 2560}, step = {1, 1};
    ttnn::Tensor view = ttnn::slice(input, begins, ends, step);
    ASSERT_EQ(view.buffer()->address(), input.buffer()->address());
    ttnn::Tensor other_host = make(10240, 100.0f);
    const std::vector<float> other_values = other_host.to_vector<float>();
    ttnn::Tensor other = other_host.to_device(mesh_device_.get());
    ttnn::Tensor wide = ttnn::typecast(view, DataType::FLOAT32);
    const std::vector<float> got = wide.cpu().to_vector<float>();
    ASSERT_EQ(got.size(), 2560u);
    for (uint32_t c = 0; c < 2560; ++c) {
        EXPECT_EQ(got[c], values[c]) << "at " << c;
    }
    const std::vector<float> other_got = other.cpu().to_vector<float>();
    for (uint32_t c = 0; c < 10240; ++c) {
        EXPECT_EQ(other_got[c], other_values[c]) << "at " << c;
    }
}

TEST_F(SlicePrefixViewTest, OtherSlicesCopy) {
    EXPECT_FALSE(slice_matches(64, 128, 64, false));  // two tile rows
    EXPECT_FALSE(slice_matches(1, 24, 12, false));    // ends inside a tile
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
