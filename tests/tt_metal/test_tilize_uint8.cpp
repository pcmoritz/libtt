// SPDX-License-Identifier: Apache-2.0

// Tilizing row-major tensors with leading dimensions keeps every value. On
// Blackhole, uint8 tilizes in the block factories (tilize, and tilize with
// padding when H or W is ragged), which stack the leading matrices into the
// rows they split over cores; the reader maps each tile row back to its matrix.

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include <tt-metalium/mesh_device.hpp>

#include "ttnn/operations/core/core.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

class TilizeLeadingDimsTest : public ::testing::Test {
protected:
    void SetUp() override { mesh_device_ = MeshDevice::create(MeshDeviceConfig(MeshShape{1, 1})); }

    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
    }

    // Tilizes on the device, then checks the tiles on the host and after untilizing.
    template <typename T>
    void expect_round_trip(const ttnn::Shape& shape, DataType dtype) {
        std::vector<T> values(shape.volume());
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = static_cast<T>((i * 37) % 251);
        }
        const TensorSpec spec(shape, TensorLayout(dtype, PageConfig(Layout::ROW_MAJOR), {}));
        const ttnn::Tensor input = ttnn::Tensor::from_vector(values, spec).to_device(mesh_device_.get());
        const ttnn::Tensor tiled = ttnn::to_layout(input, Layout::TILE);
        EXPECT_EQ(tiled.cpu().to_vector<T>(), values) << shape << " " << dtype << " tiled";
        EXPECT_EQ(ttnn::to_layout(tiled, Layout::ROW_MAJOR).cpu().to_vector<T>(), values) << shape << " " << dtype;
    }

    std::shared_ptr<MeshDevice> mesh_device_;
};

TEST_F(TilizeLeadingDimsTest, Uint8) {
    // Ragged H and W: tilize with padding pads each matrix's rows to a tile.
    expect_round_trip<uint8_t>(ttnn::Shape({3, 37, 50}), DataType::UINT8);
    expect_round_trip<uint8_t>(ttnn::Shape({2, 3, 28, 28}), DataType::UINT8);
    expect_round_trip<uint8_t>(ttnn::Shape({1, 224, 224, 3}), DataType::UINT8);
    // Tile-aligned H, with a ragged and an aligned W (plain tilize).
    expect_round_trip<uint8_t>(ttnn::Shape({4, 64, 3}), DataType::UINT8);
    expect_round_trip<uint8_t>(ttnn::Shape({2, 3, 64, 96}), DataType::UINT8);
}

TEST_F(TilizeLeadingDimsTest, WideRows) {
    // Rows too wide for a row of tiles in L1 take the block factory for any dtype.
    expect_round_trip<uint16_t>(ttnn::Shape({2, 40, 20000}), DataType::UINT16);
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
