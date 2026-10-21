// SPDX-License-Identifier: Apache-2.0

// ttnn.argmax takes uint8 row-major inputs, over the last dimension (each
// core's slice of a row starts DRAM-aligned) and over all dimensions. Ties
// return the first index.

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <tt-metalium/mesh_device.hpp>

#include "ttnn/operations/reduction/argmax/argmax.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

class ArgmaxUint8Test : public ::testing::Test {
protected:
    void SetUp() override { mesh_device_ = MeshDevice::create(MeshDeviceConfig(MeshShape{1, 1})); }

    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
    }

    // Argmax of a [rows, cols] tensor over its rows (dim 1) or all of it, against the host.
    void expect_argmax(uint32_t rows, uint32_t cols, std::optional<int> dim, const std::vector<uint8_t>& values) {
        const TensorSpec spec(ttnn::Shape({rows, cols}), TensorLayout(DataType::UINT8, PageConfig(Layout::ROW_MAJOR), {}));
        const ttnn::Tensor input = ttnn::Tensor::from_vector(values, spec).to_device(mesh_device_.get());
        const std::vector<uint32_t> actual = ttnn::argmax(input, dim).cpu().to_vector<uint32_t>();

        // The first index of the largest value along the reduced dimension.
        const uint32_t outer = dim ? rows : 1;
        const uint32_t length = dim ? cols : rows * cols;
        std::vector<uint32_t> want(outer, 0);
        for (uint32_t o = 0; o < outer; ++o) {
            auto at = [&](uint32_t i) { return values[o * length + i]; };
            for (uint32_t i = 1; i < length; ++i) {
                if (at(i) > at(want[o])) {
                    want[o] = i;
                }
            }
        }
        EXPECT_EQ(actual, want) << rows << "x" << cols << " dim " << (dim ? *dim : -1);
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

TEST_F(ArgmaxUint8Test, LastDim) {
    // Wide rows split over many cores, with the maximum late in a row.
    auto wide = bytes(8 * 32768);
    wide[3 * 32768 + 30000] = 255;
    wide[5 * 32768 + 40] = 255;
    wide[5 * 32768 + 32000] = 255;
    expect_argmax(8, 32768, 1, wide);
    // Many short rows, one of them all equal.
    auto rows = bytes(1024 * 1000);
    std::fill(rows.begin() + 7 * 1000, rows.begin() + 8 * 1000, 9);
    expect_argmax(1024, 1000, 1, rows);
    expect_argmax(5, 3, 1, bytes(15));
}

TEST_F(ArgmaxUint8Test, AllDims) {
    auto all = bytes(1000 * 1000);
    all[123456] = 255;
    all[654321] = 255;
    expect_argmax(1000, 1000, std::nullopt, all);
    expect_argmax(4, 4, std::nullopt, std::vector<uint8_t>(16, 0));
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
