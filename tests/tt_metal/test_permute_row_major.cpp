// SPDX-License-Identifier: Apache-2.0

// A row-major permute that moves the last dimension transposes blocks on the
// compute engine (tilize, transpose, untilize). It must move every element to
// its place for every element size, 1-byte uint8 included.

#include <gtest/gtest.h>

#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

class PermuteRowMajorTest : public ::testing::Test {
protected:
    void SetUp() override { mesh_device_ = MeshDevice::create(MeshDeviceConfig(MeshShape{1, 1})); }

    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
    }

    // Permutes a ROW_MAJOR tensor of distinct small values on the device and
    // compares it with the permutation on the host.
    template <typename T>
    void expect_permute(
        const ttnn::Shape& shape,
        const ttsl::SmallVector<int64_t>& dims,
        DataType dtype,
        Layout layout = Layout::ROW_MAJOR) {
        const size_t rank = shape.rank();
        std::vector<T> values(shape.volume());
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = static_cast<T>(static_cast<float>((i * 7 + 3) % 251));
        }
        const TensorSpec spec(shape, TensorLayout(dtype, PageConfig(layout), {}));
        const ttnn::Tensor input = ttnn::Tensor::from_vector(values, spec).to_device(mesh_device_.get());
        const ttnn::Tensor output = ttnn::permute(input, dims);
        const std::vector<T> actual = output.cpu().to_vector<T>();

        ttsl::SmallVector<uint32_t> out_shape(rank);
        for (size_t d = 0; d < rank; ++d) {
            out_shape[d] = shape[dims[d]];
        }
        ASSERT_EQ(actual.size(), values.size());
        // Row-major strides of the input.
        std::vector<size_t> stride(rank, 1);
        for (size_t d = rank - 1; d > 0; --d) {
            stride[d - 1] = stride[d] * shape[d];
        }
        size_t mismatches = 0;
        std::vector<size_t> index(rank, 0);
        for (size_t o = 0; o < actual.size(); ++o) {
            size_t source = 0;
            for (size_t d = 0; d < rank; ++d) {
                source += index[d] * stride[dims[d]];
            }
            if (static_cast<float>(actual[o]) != static_cast<float>(values[source])) {
                if (mismatches < 12) {
                    std::cout << "MISMATCH out " << o << " want " << static_cast<float>(values[source]) << " got "
                              << static_cast<float>(actual[o]) << std::endl;
                }
                ++mismatches;
            }
            for (size_t d = rank; d-- > 0;) {
                if (++index[d] < out_shape[d]) {
                    break;
                }
                index[d] = 0;
            }
        }
        EXPECT_EQ(mismatches, 0u) << shape << " permuted by "
                                  << ttnn::Shape(ttsl::SmallVector<uint32_t>(dims.begin(), dims.end())) << " " << dtype
                                  << " " << layout;
    }

    // Permutations that move the last dimension: one, several and partial W and X blocks.
    template <typename T>
    void expect_permutes(DataType dtype, Layout layout = Layout::ROW_MAJOR) {
        expect_permute<T>(ttnn::Shape({1, 1, 32, 64}), {0, 1, 3, 2}, dtype, layout);
        expect_permute<T>(ttnn::Shape({2, 3, 5, 7}), {0, 3, 1, 2}, dtype, layout);
        expect_permute<T>(ttnn::Shape({3, 37, 50}), {2, 0, 1}, dtype, layout);
        expect_permute<T>(ttnn::Shape({4, 224, 3}), {2, 0, 1}, dtype, layout);
        expect_permute<T>(ttnn::Shape({65, 33}), {1, 0}, dtype, layout);
        expect_permute<T>(ttnn::Shape({2, 96, 130}), {0, 2, 1}, dtype, layout);
        expect_permute<T>(ttnn::Shape({2, 3, 4, 40, 72}), {4, 0, 3, 1, 2}, dtype, layout);
    }

    std::shared_ptr<MeshDevice> mesh_device_;
};

TEST_F(PermuteRowMajorTest, Uint8) { expect_permutes<uint8_t>(DataType::UINT8); }

TEST_F(PermuteRowMajorTest, Uint8Tiled) { expect_permutes<uint8_t>(DataType::UINT8, Layout::TILE); }

TEST_F(PermuteRowMajorTest, Uint16) { expect_permutes<uint16_t>(DataType::UINT16); }

TEST_F(PermuteRowMajorTest, Bfloat16) { expect_permutes<bfloat16>(DataType::BFLOAT16); }

TEST_F(PermuteRowMajorTest, Int32) { expect_permutes<int32_t>(DataType::INT32); }

TEST_F(PermuteRowMajorTest, Float32) { expect_permutes<float>(DataType::FLOAT32); }

}  // namespace
}  // namespace tt::tt_metal::distributed::test
