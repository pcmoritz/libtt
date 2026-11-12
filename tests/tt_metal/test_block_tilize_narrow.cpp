// SPDX-License-Identifier: Apache-2.0

// Blackhole tilizes uint8 tensors with the block tilize programs, which split a
// tensor into square blocks of tiles. A tensor narrower than a block, such as
// the bytes of 64 of Qwen3.5-35B-A3B's experts' gate or up projections
// (131072 x 512), has no full-width blocks: it failed with "Kernel
// 'compute_cliff_col' references unknown DFB 'in_full'".

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include <tt-metalium/mesh_device.hpp>

#include "ttnn/operations/data_movement/tilize/tilize.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

class BlockTilizeNarrowTest : public ::testing::Test {
protected:
    void SetUp() override { mesh_device_ = MeshDevice::create(MeshDeviceConfig(MeshShape{1, 1})); }

    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
    }

    ttnn::Tensor uint8_rows(uint32_t rows, uint32_t cols, const std::vector<uint8_t>& values) {
        const TensorSpec spec(
            ttnn::Shape({rows, cols}), TensorLayout(DataType::UINT8, PageConfig(Layout::ROW_MAJOR), {}));
        return ttnn::Tensor::from_vector(values, spec).to_device(mesh_device_.get());
    }

    std::shared_ptr<MeshDevice> mesh_device_;
};

std::vector<uint8_t> bytes(size_t n) {
    std::vector<uint8_t> values(n);
    for (size_t i = 0; i < n; ++i) {
        values[i] = static_cast<uint8_t>((i * 37 + i / 7) % 251);
    }
    return values;
}

TEST_F(BlockTilizeNarrowTest, Narrow) {
    const auto values = bytes(131072 * 512);
    const ttnn::Tensor tiled = ttnn::tilize(uint8_rows(131072, 512, values));
    ASSERT_EQ(tiled.layout(), Layout::TILE);
    EXPECT_EQ(tiled.cpu().to_vector<uint8_t>(), values);
}

TEST_F(BlockTilizeNarrowTest, NarrowWithPadding) {
    // Two rows and 12 columns short of whole tiles.
    const auto values = bytes(131070 * 500);
    const ttnn::Tensor tiled = ttnn::tilize_with_val_padding(
        uint8_rows(131070, 500, values), ttnn::Shape({131072, 512}), ttnn::PadValue(uint32_t{0}));
    ASSERT_EQ(tiled.layout(), Layout::TILE);
    EXPECT_EQ(tiled.cpu().to_vector<uint8_t>(), values);
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
