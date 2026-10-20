// SPDX-License-Identifier: Apache-2.0
//
// ttnn::matmul on FP8_E4M3 and FP8_E5M2 tiles: the unpacker reads the
// operands as Fp8_e4m3 and Lf8, one byte per element. It decodes normal values
// exactly and flushes subnormals to zero, so the products match a reference
// with flushed subnormals up to the accumulation's rounding.

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tensor/tensor_apis.hpp>

#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

class Fp8MatmulTest : public ::testing::Test {
protected:
    void SetUp() override { mesh_device_ = MeshDevice::create(MeshDeviceConfig(MeshShape{1, 1})); }

    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
    }

    // A tile-layout FP8_E4M3 or FP8_E5M2 tensor on the device with the given codes in row-major order.
    ttnn::Tensor fp8_device(
        const std::vector<uint8_t>& codes, uint32_t rows, uint32_t cols, DataType dtype = DataType::FP8_E4M3) {
        const TensorSpec spec(ttnn::Shape({rows, cols}), TensorLayout(dtype, PageConfig(Layout::ROW_MAJOR), {}));
        const HostTensor row_major = HostTensor::allocate_for_overwrite(spec).transform([&](const HostBuffer&) {
            auto bytes = [&]<typename T>() {
                std::vector<T> buffer(codes.size());
                std::memcpy(buffer.data(), codes.data(), codes.size());
                return HostBuffer(std::move(buffer));
            };
            return dtype == DataType::FP8_E5M2 ? bytes.template operator()<float8_e5m2>()
                                               : bytes.template operator()<float8_e4m3>();
        });
        return ttnn::Tensor(to_tile_layout(row_major, Tile())).to_device(mesh_device_.get());
    }

    ttnn::Tensor float_device(const std::vector<float>& values, uint32_t rows, uint32_t cols, DataType dtype) {
        const TensorSpec spec(ttnn::Shape({rows, cols}), TensorLayout(dtype, PageConfig(Layout::TILE), {}));
        return ttnn::Tensor::from_vector(values, spec).to_device(mesh_device_.get());
    }

    std::shared_ptr<MeshDevice> mesh_device_;
};

// float8_e4m3fn to float; subnormals flushed to zero when `flush` is set.
float decode(uint8_t bits, bool flush) {
    const int exponent = (bits >> 3) & 0xF;
    const int mantissa = bits & 0x7;
    float value;
    if (exponent == 15 && mantissa == 7) {
        value = std::nanf("");
    } else if (exponent == 0) {
        value = flush ? 0.0f : std::ldexp(static_cast<float>(mantissa), -9);
    } else {
        value = std::ldexp(static_cast<float>(8 + mantissa), exponent - 10);
    }
    return bits & 0x80 ? -value : value;
}

// float8_e5m2 to float; subnormals flushed to zero when `flush` is set.
float decode_e5m2(uint8_t bits, bool flush) {
    const int exponent = (bits >> 2) & 0x1F;
    const int mantissa = bits & 0x3;
    float value;
    if (exponent == 31) {
        value = mantissa != 0 ? std::nanf("") : INFINITY;
    } else if (exponent == 0) {
        value = flush ? 0.0f : std::ldexp(static_cast<float>(mantissa), -16);
    } else {
        value = std::ldexp(static_cast<float>(4 + mantissa), exponent - 17);
    }
    return bits & 0x80 ? -value : value;
}

// Random finite e5m2 codes, subnormals included.
std::vector<uint8_t> random_e5m2_codes(size_t count, std::mt19937& rng) {
    std::uniform_int_distribution<int> code(0, 247);
    std::vector<uint8_t> codes(count);
    for (uint8_t& c : codes) {
        const int v = code(rng);
        c = static_cast<uint8_t>(v < 124 ? v : v + 4);  // skip 0x7C-0x7F; 0xFC-0xFF are out of range
    }
    return codes;
}

// Random e4m3 codes without NaN: every finite code, subnormals included.
std::vector<uint8_t> random_codes(size_t count, std::mt19937& rng) {
    std::uniform_int_distribution<int> code(0, 253);
    std::vector<uint8_t> codes(count);
    for (uint8_t& c : codes) {
        const int v = code(rng);
        c = static_cast<uint8_t>(v < 127 ? v : v + 1);  // skip 0x7F; 0xFF is out of range
    }
    return codes;
}

struct Errors {
    double exact = 0.0;    // worst |c - ref| / sum |a_ik b_kj| against exact decoding
    double flushed = 0.0;  // the same against decoding with subnormals flushed
};

Errors compare(
    const std::vector<float>& c, const std::vector<float>& a_exact, const std::vector<float>& a_flushed,
    const std::vector<float>& b_exact, const std::vector<float>& b_flushed, uint32_t M, uint32_t K, uint32_t N) {
    Errors errors;
    for (uint32_t i = 0; i < M; ++i) {
        for (uint32_t j = 0; j < N; ++j) {
            double exact = 0.0, flushed = 0.0, scale = 0.0;
            for (uint32_t k = 0; k < K; ++k) {
                exact += static_cast<double>(a_exact[i * K + k]) * b_exact[k * N + j];
                flushed += static_cast<double>(a_flushed[i * K + k]) * b_flushed[k * N + j];
                scale += std::abs(static_cast<double>(a_exact[i * K + k]) * b_exact[k * N + j]);
            }
            scale = std::max(scale, 1e-30);
            errors.exact = std::max(errors.exact, std::abs(c[i * N + j] - exact) / scale);
            errors.flushed = std::max(errors.flushed, std::abs(c[i * N + j] - flushed) / scale);
        }
    }
    return errors;
}

std::vector<float> decode_all(const std::vector<uint8_t>& codes, bool flush, bool e5m2 = false) {
    std::vector<float> values(codes.size());
    for (size_t i = 0; i < codes.size(); ++i) {
        values[i] = e5m2 ? decode_e5m2(codes[i], flush) : decode(codes[i], flush);
    }
    return values;
}

TEST_F(Fp8MatmulTest, MatchesFlushedReference) {
    struct Shape {
        uint32_t M, K, N;
    };
    for (const Shape& shape : {Shape{32, 32, 32}, Shape{64, 256, 96}, Shape{256, 1024, 128}}) {
        const auto [M, K, N] = shape;
        std::mt19937 rng(M * 131 + K * 7 + N);
        const std::vector<uint8_t> a = random_codes(M * K, rng);
        const std::vector<uint8_t> b = random_codes(K * N, rng);
        const std::vector<float> a_exact = decode_all(a, false), a_flushed = decode_all(a, true);
        const std::vector<float> b_exact = decode_all(b, false), b_flushed = decode_all(b, true);
        const ttnn::Tensor a_device = fp8_device(a, M, K);
        const ttnn::Tensor b_device = fp8_device(b, K, N);
        // The DEST register has to be in 32-bit mode for Fp8 inputs.
        for (const MathFidelity fidelity : {MathFidelity::LoFi, MathFidelity::HiFi4}) {
            const ttnn::Tensor c_device = ttnn::matmul(
                a_device, b_device, false, false, std::nullopt, DataType::FLOAT32, std::nullopt, std::nullopt,
                ttnn::DeviceComputeKernelConfig{.math_fidelity = fidelity, .fp32_dest_acc_en = true});
            const Errors errors =
                compare(c_device.cpu().to_vector<float>(), a_exact, a_flushed, b_exact, b_flushed, M, K, N);
            std::printf(
                "fp8 x fp8 %ux%ux%u %s: worst error %.3g vs exact, %.3g vs flushed subnormals\n",
                M,
                K,
                N,
                fidelity == MathFidelity::LoFi ? "LoFi" : "HiFi4",
                errors.exact,
                errors.flushed);
            // About the error of a BFLOAT16 matmul of the same values (below).
            EXPECT_LE(errors.flushed, std::ldexp(1.0, -9));
        }
        // The same values (exact in BFLOAT16) multiplied as BFLOAT16, for comparison.
        const ttnn::Tensor c_bf16 = ttnn::matmul(
            float_device(a_exact, M, K, DataType::BFLOAT16), float_device(b_exact, K, N, DataType::BFLOAT16), false,
            false, std::nullopt, DataType::FLOAT32, std::nullopt, std::nullopt,
            ttnn::DeviceComputeKernelConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true});
        const Errors bf16_errors =
            compare(c_bf16.cpu().to_vector<float>(), a_exact, a_flushed, b_exact, b_flushed, M, K, N);
        std::printf("bf16 x bf16 %ux%ux%u HiFi4: worst error %.3g vs exact\n", M, K, N, bf16_errors.exact);
    }
}

// FP8_E5M2 operands, and one of each format: both unpack to Float16.
TEST_F(Fp8MatmulTest, E5m2AndMixedOperands) {
    // The larger shape's program also has a BFLOAT16 buffer for partial sums.
    for (const auto& [M, K, N] : {std::array<uint32_t, 3>{64, 256, 96}, std::array<uint32_t, 3>{1024, 1024, 256}}) {
        std::mt19937 rng(11);
        const std::vector<uint8_t> a4 = random_codes(M * K, rng), b4 = random_codes(K * N, rng);
        const std::vector<uint8_t> a5 = random_e5m2_codes(M * K, rng), b5 = random_e5m2_codes(K * N, rng);
        struct Case {
            const char* name;
            const std::vector<uint8_t>& a;
            bool a_e5m2;
            const std::vector<uint8_t>& b;
            bool b_e5m2;
        };
        for (const Case& c : {Case{"e5m2 x e5m2", a5, true, b5, true}, Case{"e4m3 x e5m2", a4, false, b5, true},
                              Case{"e5m2 x e4m3", a5, true, b4, false}}) {
            const auto dtype = [](bool e5m2) { return e5m2 ? DataType::FP8_E5M2 : DataType::FP8_E4M3; };
            const ttnn::Tensor product = ttnn::matmul(
                fp8_device(c.a, M, K, dtype(c.a_e5m2)), fp8_device(c.b, K, N, dtype(c.b_e5m2)), false, false,
                std::nullopt, DataType::FLOAT32, std::nullopt, std::nullopt,
                ttnn::DeviceComputeKernelConfig{.math_fidelity = MathFidelity::LoFi, .fp32_dest_acc_en = true});
            const Errors errors = compare(
                product.cpu().to_vector<float>(), decode_all(c.a, false, c.a_e5m2), decode_all(c.a, true, c.a_e5m2),
                decode_all(c.b, false, c.b_e5m2), decode_all(c.b, true, c.b_e5m2), M, K, N);
            std::printf(
                "%s %ux%ux%u LoFi: worst error %.3g vs exact, %.3g vs flushed subnormals\n",
                c.name,
                M,
                K,
                N,
                errors.exact,
                errors.flushed);
            EXPECT_LE(std::min(errors.exact, errors.flushed), std::ldexp(1.0, -9)) << c.name;
        }
    }
}

// The operands unpack to different exponent families (Float16 for Fp8,
// Float16_b for BFLOAT16), and the products are garbage.
TEST_F(Fp8MatmulTest, MixedWithBfloat16IsWrong) {
    const uint32_t M = 64, K = 256, N = 96;
    std::mt19937 rng(7);
    const std::vector<uint8_t> a = random_codes(M * K, rng);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> b(K * N);
    for (float& v : b) {
        v = static_cast<float>(bfloat16(normal(rng)));
    }
    const ttnn::Tensor c_device = ttnn::matmul(
        fp8_device(a, M, K), float_device(b, K, N, DataType::BFLOAT16), false, false, std::nullopt,
        DataType::FLOAT32, std::nullopt, std::nullopt,
        ttnn::DeviceComputeKernelConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true});
    const Errors errors =
        compare(c_device.cpu().to_vector<float>(), decode_all(a, false), decode_all(a, true), b, b, M, K, N);
    std::printf("fp8 x bf16 %ux%ux%u HiFi4: worst error %.3g\n", M, K, N, errors.flushed);
    EXPECT_GT(errors.flushed, 0.1);
}

// Time per matmul of a few large shapes, fp8 against bf16 and bfp8 operands.
TEST_F(Fp8MatmulTest, Performance) {
    auto time = [&](const std::function<ttnn::Tensor()>& run) {
        for (int i = 0; i < 3; ++i) {
            run();
        }
        Finish(mesh_device_->mesh_command_queue());
        const int iterations = 20;
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; ++i) {
            run();
        }
        Finish(mesh_device_->mesh_command_queue());
        return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() /
               iterations;
    };
    struct Shape {
        uint32_t M, K, N;
    };
    for (const Shape& shape : {Shape{32, 4096, 4096}, Shape{1024, 4096, 4096}, Shape{4096, 4096, 4096}}) {
        const auto [M, K, N] = shape;
        std::mt19937 rng(1);
        const ttnn::Tensor a8 = fp8_device(random_codes(M * K, rng), M, K);
        const ttnn::Tensor b8 = fp8_device(random_codes(K * N, rng), K, N);
        const ttnn::Tensor a5 = fp8_device(random_e5m2_codes(M * K, rng), M, K, DataType::FP8_E5M2);
        const ttnn::Tensor b5 = fp8_device(random_e5m2_codes(K * N, rng), K, N, DataType::FP8_E5M2);
        std::vector<float> a(M * K, 0.5f), b(K * N, 0.25f);
        const ttnn::Tensor a16 = float_device(a, M, K, DataType::BFLOAT16);
        const ttnn::Tensor b16 = float_device(b, K, N, DataType::BFLOAT16);
        const ttnn::Tensor b_bfp8 = float_device(b, K, N, DataType::BFLOAT8_B);
        const ttnn::DeviceComputeKernelConfig lofi{.math_fidelity = MathFidelity::LoFi, .fp32_dest_acc_en = true};
        const ttnn::DeviceComputeKernelConfig hifi2{.math_fidelity = MathFidelity::HiFi2, .fp32_dest_acc_en = true};
        auto mm = [&](const ttnn::Tensor& x, const ttnn::Tensor& y, const ttnn::DeviceComputeKernelConfig& config) {
            return [&, config] {
                return ttnn::matmul(
                    x, y, false, false, std::nullopt, DataType::BFLOAT16, std::nullopt, std::nullopt, config);
            };
        };
        const double flops = 2.0 * M * K * N;
        for (const auto& [name, us] : std::vector<std::pair<std::string, double>>{
                 {"fp8 x fp8 LoFi", time(mm(a8, b8, lofi))},
                 {"e5m2 x e5m2 LoFi", time(mm(a5, b5, lofi))},
                 {"bf16 x bf16 LoFi", time(mm(a16, b16, lofi))},
                 {"bf16 x bf16 HiFi2", time(mm(a16, b16, hifi2))},
                 {"bf16 x bfp8 LoFi", time(mm(a16, b_bfp8, lofi))},
             }) {
            std::printf("%ux%ux%u %s: %.1f us, %.1f TFLOP/s\n", M, K, N, name.c_str(), us, flops / us * 1e-6);
        }
    }
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
