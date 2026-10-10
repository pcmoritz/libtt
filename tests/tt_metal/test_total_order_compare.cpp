// SPDX-License-Identifier: Apache-2.0

// The *_TOTAL_ORDER binary ops compare FLOAT32 tensors in IEEE 754 total
// order: -NaN < -inf < ... < -0 < +0 < ... < +inf < +NaN, NaNs by payload.
// Every pair of special values must compare like their order-preserving
// integer keys, also when the operands broadcast against each other.

#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <tt-metalium/mesh_device.hpp>

#include "ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

using ttnn::operations::binary_ng::BinaryOpType;

// NaNs of both signs with different payloads, infinities, signed zeros,
// adjacent values and subnormals.
constexpr uint32_t kSpecialBits[] = {
    0xFFFFFFFF,  // -NaN, largest payload
    0xFFC00000,  // -NaN, quiet
    0xFF800001,  // -NaN, smallest payload
    0xFF800000,  // -inf
    0xBF800001,  // next float below -1
    0xBF800000,  // -1
    0x80000001,  // smallest negative subnormal
    0x80000000,  // -0
    0x00000000,  // +0
    0x00000001,  // smallest positive subnormal
    0x3F800000,  // 1
    0x3F800001,  // next float above 1
    0x7F800000,  // +inf
    0x7F800001,  // +NaN, smallest payload
    0x7FC00000,  // +NaN, quiet
    0x7FFFFFFF,  // +NaN, largest payload
};
constexpr uint32_t kN = std::size(kSpecialBits);

int32_t total_order_key(uint32_t bits) {
    const auto s = static_cast<int32_t>(bits);
    return s ^ ((s >> 31) & 0x7FFFFFFF);
}

struct Case {
    BinaryOpType op;
    std::string name;
    bool (*reference)(int32_t, int32_t);
};

class TotalOrderCompareTest : public ::testing::TestWithParam<Case> {
protected:
    void SetUp() override { mesh_device_ = MeshDevice::create(MeshDeviceConfig(MeshShape{1, 1})); }

    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
    }

    ttnn::Tensor to_device(const std::vector<float>& values, const std::vector<uint32_t>& shape) {
        const TensorSpec spec(
            ttnn::Shape(shape), TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), MemoryConfig{}));
        return ttnn::Tensor::from_vector(values, spec).to_device(mesh_device_.get());
    }

    void check(const ttnn::Tensor& a, const ttnn::Tensor& b, uint32_t (*lhs)(uint32_t, uint32_t),
               uint32_t (*rhs)(uint32_t, uint32_t)) {
        const Case& c = GetParam();
        // The result is FLOAT32, like the inputs. (Asking binary_ng for a
        // BFLOAT16 result of a FLOAT32 SFPU comparison, total-order or not,
        // packs wrong values; callers typecast instead.)
        const std::vector<float> got = ttnn::prim::binary_ng(a, b, c.op).cpu().to_vector<float>();
        ASSERT_EQ(got.size(), kN * kN);
        for (uint32_t i = 0; i < kN; ++i) {
            for (uint32_t j = 0; j < kN; ++j) {
                const uint32_t x = kSpecialBits[lhs(i, j)];
                const uint32_t y = kSpecialBits[rhs(i, j)];
                const float expected = c.reference(total_order_key(x), total_order_key(y)) ? 1.0f : 0.0f;
                EXPECT_EQ(got[i * kN + j], expected)
                    << c.name << std::hex << " lhs 0x" << x << " rhs 0x" << y;
            }
        }
    }

    std::shared_ptr<MeshDevice> mesh_device_;
};

std::vector<float> special_values(uint32_t (*index)(uint32_t, uint32_t), uint32_t rows, uint32_t cols) {
    std::vector<float> values;
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
            values.push_back(std::bit_cast<float>(kSpecialBits[index(i, j)]));
        }
    }
    return values;
}

uint32_t row(uint32_t i, uint32_t) { return i; }
uint32_t col(uint32_t, uint32_t j) { return j; }

TEST_P(TotalOrderCompareTest, AllPairs) {
    check(to_device(special_values(row, kN, kN), {kN, kN}), to_device(special_values(col, kN, kN), {kN, kN}), row,
          col);
}

// [N, 1] against [1, N]: binary_ng broadcasts both operands in the kernel.
TEST_P(TotalOrderCompareTest, RowColumnBroadcast) {
    check(to_device(special_values(row, kN, 1), {kN, 1}), to_device(special_values(col, 1, kN), {1, kN}), row, col);
}

INSTANTIATE_TEST_SUITE_P(
    Ops,
    TotalOrderCompareTest,
    ::testing::Values(
        Case{BinaryOpType::EQ_TOTAL_ORDER, "eq", [](int32_t x, int32_t y) { return x == y; }},
        Case{BinaryOpType::NE_TOTAL_ORDER, "ne", [](int32_t x, int32_t y) { return x != y; }},
        Case{BinaryOpType::LT_TOTAL_ORDER, "lt", [](int32_t x, int32_t y) { return x < y; }},
        Case{BinaryOpType::GT_TOTAL_ORDER, "gt", [](int32_t x, int32_t y) { return x > y; }},
        Case{BinaryOpType::LE_TOTAL_ORDER, "le", [](int32_t x, int32_t y) { return x <= y; }},
        Case{BinaryOpType::GE_TOTAL_ORDER, "ge", [](int32_t x, int32_t y) { return x >= y; }}),
    [](const auto& info) { return info.param.name; });

}  // namespace
}  // namespace tt::tt_metal::distributed::test
