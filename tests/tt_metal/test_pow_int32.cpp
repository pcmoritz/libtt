// SPDX-License-Identifier: Apache-2.0

// ttnn::pow on Int32 and UInt32 tiles computes exact powers that wrap like the
// tiles' multiplication: a tensor exponent squares and multiplies per lane, a
// scalar exponent in a fixed chain. A negative Int32 exponent gives 1 for a
// base of 1, +-1 for -1 and 0 otherwise, as in StableHLO.

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include <tt-metalium/mesh_device.hpp>

#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

uint32_t reference_pow(uint32_t base, uint32_t exponent, bool is_signed) {
    if (is_signed && static_cast<int32_t>(exponent) < 0) {
        const auto b = static_cast<int32_t>(base);
        return b == 1 ? 1u : b == -1 ? ((exponent & 1) ? 0xFFFFFFFFu : 1u) : 0u;
    }
    uint32_t result = 1;
    for (uint32_t e = exponent, b = base; e != 0; e >>= 1, b *= b) {
        if (e & 1) {
            result *= b;
        }
    }
    return result;
}

class PowInt32Test : public ::testing::Test {
protected:
    void SetUp() override { mesh_device_ = MeshDevice::create(MeshDeviceConfig(MeshShape{1, 1})); }

    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
    }

    template <typename T>
    ttnn::Tensor device_tensor(const std::vector<T>& values, DataType dtype) {
        const TensorSpec spec(shape_, TensorLayout(dtype, PageConfig(Layout::TILE), {}));
        return ttnn::Tensor::from_vector(values, spec).to_device(mesh_device_.get());
    }

    // Edge cases first, then random bases and exponents over the whole range
    // and small ones, whose powers neither vanish nor wrap.
    void make_inputs(bool is_signed) {
        bases_.clear();
        exponents_.clear();
        const std::vector<std::pair<uint32_t, uint32_t>> edges = {
            {0, 0}, {5, 0}, {0, 3}, {9, 9}, {2, 31}, {2, 32}, {3, 1u << 30}, {3, (1u << 30) + 5},
            {2, 1u << 30}, {6, (1u << 30) + 1}, {3, 1u << 31}, {3, 0xFFFFFFFFu}, {2, 0x80000002u},
            {0xFFFFFFFFu, 3}, {1, 0xFFFFFFFDu}, {0xFFFFFFFFu, 0xFFFFFFFDu}, {0xFFFFFFFFu, 0xFFFFFFFEu},
            {5, 0xFFFFFFFFu}, {12345, (1u << 30) - 1}, {0x80000000u, 1}, {0x80000000u, 2}};
        for (const auto& [base, exponent] : edges) {
            bases_.push_back(base);
            exponents_.push_back(exponent);
        }
        std::mt19937 rng(is_signed ? 1 : 2);
        std::uniform_int_distribution<uint32_t> any;
        std::uniform_int_distribution<int32_t> small_base(-12, 12);
        std::uniform_int_distribution<int32_t> small_exponent(is_signed ? -3 : 0, 40);
        while (bases_.size() < shape_.volume()) {
            const bool small = bases_.size() % 2;
            bases_.push_back(small ? static_cast<uint32_t>(small_base(rng)) : any(rng));
            exponents_.push_back(small ? static_cast<uint32_t>(small_exponent(rng)) : any(rng));
        }
    }

    template <typename T>
    void expect_powers(const ttnn::Tensor& result, const std::vector<uint32_t>& exponents, bool is_signed) {
        const std::vector<T> got = result.cpu().to_vector<T>();
        ASSERT_EQ(got.size(), bases_.size());
        size_t mismatches = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            const uint32_t expected = reference_pow(bases_[i], exponents[i], is_signed);
            if (static_cast<uint32_t>(got[i]) != expected && mismatches++ < 8) {
                ADD_FAILURE() << bases_[i] << " ** " << exponents[i] << ": got " << static_cast<uint32_t>(got[i])
                              << ", expected " << expected;
            }
        }
        EXPECT_EQ(mismatches, 0u);
    }

    template <typename T>
    void check(DataType dtype) {
        const bool is_signed = dtype == DataType::INT32;
        make_inputs(is_signed);
        const std::vector<T> bases(bases_.begin(), bases_.end());
        const std::vector<T> exponents(exponents_.begin(), exponents_.end());
        const ttnn::Tensor base = device_tensor(bases, dtype);
        expect_powers<T>(ttnn::pow(base, device_tensor(exponents, dtype)), exponents_, is_signed);
        for (const int32_t exponent : {0, 1, 2, 3, 5, 9, 31, 32, 1 << 30, (1 << 30) + 3, 0x7FFFFFFF}) {
            const ttnn::Tensor result = ttnn::pow(base, exponent);
            expect_powers<T>(result, std::vector<uint32_t>(bases_.size(), exponent), is_signed);
        }
    }

    const ttnn::Shape shape_{64, 96};
    std::vector<uint32_t> bases_;
    std::vector<uint32_t> exponents_;
    std::shared_ptr<MeshDevice> mesh_device_;
};

TEST_F(PowInt32Test, Int32) { check<int32_t>(DataType::INT32); }

TEST_F(PowInt32Test, UInt32) { check<uint32_t>(DataType::UINT32); }

}  // namespace
}  // namespace tt::tt_metal::distributed::test
