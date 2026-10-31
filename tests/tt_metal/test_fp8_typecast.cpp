// SPDX-License-Identifier: Apache-2.0

// ttnn.typecast between float8 (FP8_E4M3, FP8_E5M2) and FLOAT32 is exact:
// every code decodes to its value (subnormals, NaN and -0 included), and
// float32 values encode rounded to nearest even, with subnormal results kept,
// NaN or infinity past the largest finite value and the sign of zero kept.
// The references find the nearest code by brute force. ttnn.bitcast between
// UINT8 and float8 is a view of the same bytes.

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include <tt-metalium/mesh_device.hpp>

#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

struct Format {
    DataType dtype;
    int mantissa_bits;
    int bias;
    bool has_infinity;
};

constexpr Format kE4M3{DataType::FP8_E4M3, 3, 7, false};
constexpr Format kE5M2{DataType::FP8_E5M2, 2, 15, true};

// The value of a code from the definition of the format.
float decode(const Format& f, uint8_t code) {
    const int magnitude = code & 0x7f;
    const int exponent = magnitude >> f.mantissa_bits;
    const int mantissa = magnitude & ((1 << f.mantissa_bits) - 1);
    float value;
    if (f.has_infinity && exponent == (1 << (7 - f.mantissa_bits)) - 1) {
        value = mantissa == 0 ? std::numeric_limits<float>::infinity() : std::numeric_limits<float>::quiet_NaN();
    } else if (!f.has_infinity && magnitude == 0x7f) {
        value = std::numeric_limits<float>::quiet_NaN();
    } else if (exponent == 0) {
        value = std::ldexp(static_cast<float>(mantissa), 1 - f.bias - f.mantissa_bits);
    } else {
        value = std::ldexp(static_cast<float>((1 << f.mantissa_bits) | mantissa), exponent - f.bias - f.mantissa_bits);
    }
    return (code & 0x80) ? -value : value;
}

// The nearest code (ties to the even code). One step past the largest finite value is the
// overflow code: NaN for e4m3fn, infinity for e5m2. Every result keeps the input's sign, zeros
// and NaNs included; a NaN encodes as code 0x7f.
uint8_t encode(const Format& f, float value) {
    const uint8_t sign = std::signbit(value) ? 0x80 : 0;
    if (std::isnan(value)) {
        return sign | 0x7f;
    }
    const int top = f.has_infinity ? 0x7c : 0x7f;  // the overflow code
    // The overflow code stands for the next finite step. Below it, distances in double are exact.
    auto at = [&](int code) -> double {
        return code == top ? 2.0 * decode(f, top - 1) - decode(f, top - 2) : decode(f, code);
    };
    const double magnitude = std::fabs(static_cast<double>(value));
    if (magnitude >= at(top)) {
        return sign | top;
    }
    int best = 0;
    for (int code = 1; code <= top; ++code) {
        const double d = std::fabs(at(code) - magnitude), best_d = std::fabs(at(best) - magnitude);
        if (d < best_d || (d == best_d && code % 2 == 0)) {
            best = code;
        }
    }
    return sign | best;
}

class Fp8TypecastTest : public ::testing::Test {
protected:
    void SetUp() override { mesh_device_ = MeshDevice::create(MeshDeviceConfig(MeshShape{1, 1})); }

    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
    }

    std::shared_ptr<MeshDevice> mesh_device_;
};

// Every bfloat16 pattern; each rounding midpoint between neighboring codes of both formats with its
// float32 neighbors, both signs; values past the ends; and random float32 patterns.
std::vector<float> encode_inputs() {
    std::vector<float> values;
    for (uint32_t bits = 0; bits < (1 << 16); ++bits) {
        values.push_back(std::bit_cast<float>(bits << 16));
    }
    for (const Format& f : {kE4M3, kE5M2}) {
        const int top = f.has_infinity ? 0x7c : 0x7f;
        for (int code = 0; code < top; ++code) {
            const double next = code + 1 == top ? 2.0 * decode(f, top - 1) - decode(f, top - 2) : decode(f, code + 1);
            const float mid = static_cast<float>((decode(f, code) + next) / 2);
            for (float v : {mid, std::nextafter(mid, 0.0f), std::nextafter(mid, INFINITY)}) {
                values.push_back(v);
                values.push_back(-v);
            }
        }
    }
    for (float v : {1e30f, std::numeric_limits<float>::max(), std::numeric_limits<float>::infinity(),
                    std::ldexp(1.0f, -149), std::ldexp(1.0f, -126), 0.0f, std::numeric_limits<float>::quiet_NaN()}) {
        values.push_back(v);
        values.push_back(-v);
    }
    uint32_t state = 12345;
    for (int i = 0; i < 4096; ++i) {
        state = state * 1664525u + 1013904223u;
        values.push_back(std::bit_cast<float>(state));
    }
    return values;
}

TEST_F(Fp8TypecastTest, DecodesEveryCode) {
    std::vector<uint8_t> codes(32 * 32);
    for (size_t i = 0; i < codes.size(); ++i) {
        codes[i] = static_cast<uint8_t>(i);
    }
    const TensorSpec spec(ttnn::Shape({32, 32}), TensorLayout(DataType::UINT8, PageConfig(Layout::TILE), {}));
    for (const Format& f : {kE4M3, kE5M2}) {
        const ttnn::Tensor bytes = ttnn::Tensor::from_vector(codes, spec).to_device(mesh_device_.get());
        const auto values =
            ttnn::typecast(ttnn::bitcast(bytes, f.dtype), DataType::FLOAT32).cpu().to_vector<float>();
        ASSERT_EQ(values.size(), codes.size());
        size_t wrong = 0;
        for (size_t i = 0; i < codes.size(); ++i) {
            const float want = decode(f, codes[i]);
            wrong += std::isnan(want) ? !std::isnan(values[i])
                                      : std::bit_cast<uint32_t>(values[i]) != std::bit_cast<uint32_t>(want);
        }
        EXPECT_EQ(wrong, 0u) << f.dtype;
    }
}

TEST_F(Fp8TypecastTest, EncodesRoundingToNearestEven) {
    std::vector<float> values = encode_inputs();
    values.resize((values.size() + 1023) / 1024 * 1024, 1.0f);
    const TensorSpec spec(
        ttnn::Shape({static_cast<uint32_t>(values.size() / 32), 32}),
        TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), {}));
    const ttnn::Tensor input = ttnn::Tensor::from_vector(values, spec).to_device(mesh_device_.get());
    for (const Format& f : {kE4M3, kE5M2}) {
        const auto codes =
            ttnn::bitcast(ttnn::typecast(input, f.dtype), DataType::UINT8).cpu().to_vector<uint8_t>();
        ASSERT_EQ(codes.size(), values.size());
        size_t wrong = 0;
        for (size_t i = 0; i < values.size(); ++i) {
            if (codes[i] != encode(f, values[i]) && wrong++ < 5) {
                ADD_FAILURE() << f.dtype << " " << values[i] << " (0x" << std::hex << std::bit_cast<uint32_t>(values[i])
                              << "): got 0x" << int(codes[i]) << ", want 0x" << int(encode(f, values[i])) << std::dec;
            }
        }
        EXPECT_EQ(wrong, 0u) << f.dtype;
    }
}

// Both directions and layouts keep the bytes and the buffer. A bitcast that asks for a preallocated
// output or another memory config, which a view can't give, fails.
TEST_F(Fp8TypecastTest, BitcastsBytesAsViews) {
    std::vector<uint8_t> codes(32 * 32);
    for (size_t i = 0; i < codes.size(); ++i) {
        codes[i] = static_cast<uint8_t>(i);
    }
    for (Layout layout : {Layout::TILE, Layout::ROW_MAJOR}) {
        const TensorSpec spec(ttnn::Shape({32, 32}), TensorLayout(DataType::UINT8, PageConfig(layout), {}));
        const ttnn::Tensor bytes = ttnn::Tensor::from_vector(codes, spec).to_device(mesh_device_.get());
        for (const Format& f : {kE4M3, kE5M2}) {
            const ttnn::Tensor view = ttnn::bitcast(bytes, f.dtype, bytes.memory_config());
            const ttnn::Tensor back = ttnn::bitcast(view, DataType::UINT8);
            EXPECT_EQ(view.dtype(), f.dtype);
            EXPECT_EQ(view.buffer()->address(), bytes.buffer()->address());
            EXPECT_EQ(back.buffer()->address(), bytes.buffer()->address());
            EXPECT_EQ(back.cpu().to_vector<uint8_t>(), codes) << f.dtype << (layout == Layout::TILE ? " TILE" : " RM");
            EXPECT_ANY_THROW((void)ttnn::bitcast(bytes, f.dtype, ttnn::L1_MEMORY_CONFIG));
            EXPECT_ANY_THROW((void)ttnn::bitcast(view, DataType::UINT8, ttnn::L1_MEMORY_CONFIG));
            EXPECT_ANY_THROW((void)ttnn::bitcast(bytes, f.dtype, std::nullopt, view));
            EXPECT_ANY_THROW((void)ttnn::bitcast(view, DataType::UINT8, std::nullopt, bytes));
        }
    }
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
