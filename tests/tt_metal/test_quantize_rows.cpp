// SPDX-License-Identifier: Apache-2.0

// ttnn::experimental::quantize_rows: per row, scale = max(amax(|x|) / fp8_max,
// min_scale) and q = float8(x / scale). The scale must equal the IEEE float32
// one, and q must equal ttnn.typecast(ttnn.divide(x, scale)) bit for bit (the
// ops SGLang's FP8 linears ran as before) and, apart from the sign of zero and
// NaN, the nearest-even float8 code of the IEEE quotient or of a float32
// neighbor of it (ttnn.divide is within an ulp). Shapes cover every
// row stride (4, 8, 16, 32 rows), partial last dense tiles, one core and core
// rectangles of one and several NoC rows; values cover zero rows, outliers,
// subnormal quotients, infinities and NaN (an FP32 NaN makes its row's scale
// NaN as in PyTorch; the unpacker widens a BF16 NaN to infinity).

#include <gtest/gtest.h>

#include <bit>
#include <chrono>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/experimental/quantize_rows/device/quantize_rows_device_operation.hpp"
#include "ttnn/operations/experimental/quantize_rows/quantize_rows.hpp"
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
    float max;
};

constexpr Format kE4M3{DataType::FP8_E4M3, 3, 7, false, 448.0f};
constexpr Format kE5M2{DataType::FP8_E5M2, 2, 15, true, 57344.0f};
constexpr float kMinScale = 1e-12f;

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

// The nearest code, ties to even; past the largest finite value, the overflow code.
uint8_t encode(const Format& f, float value) {
    const uint8_t sign = std::signbit(value) ? 0x80 : 0;
    if (std::isnan(value)) {
        return sign | 0x7f;
    }
    const int top = f.has_infinity ? 0x7c : 0x7f;
    auto at = [&](int code) -> double {
        return code == top ? 2.0 * decode(f, top - 1) - decode(f, top - 2) : decode(f, code);
    };
    const double magnitude = std::fabs(static_cast<double>(value));
    if (magnitude >= at(top)) {
        return sign | top;
    }
    // Codes increase with magnitude: binary search for the neighbors.
    int lo = 0, hi = top;
    while (hi - lo > 1) {
        const int mid = (lo + hi) / 2;
        (at(mid) <= magnitude ? lo : hi) = mid;
    }
    const double dlo = magnitude - at(lo), dhi = at(hi) - magnitude;
    const int best = dlo < dhi || (dlo == dhi && lo % 2 == 0) ? lo : hi;
    return sign | best;
}

bool is_nan_code(const Format& f, uint8_t code) { return std::isnan(decode(f, code)); }

class QuantizeRowsTest : public ::testing::Test {
protected:
    void SetUp() override { mesh_device_ = MeshDevice::create(MeshDeviceConfig(MeshShape{1, 1})); }

    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
    }

    // Quantizes `values` ([rows, cols], rounded to the input type) with at most
    // max_cores cores and checks both results. Returns the number of mismatches.
    size_t check(
        const std::vector<float>& values,
        uint32_t rows,
        uint32_t cols,
        DataType in_dtype,
        const Format& f,
        uint32_t max_cores) {
        std::vector<float> x(values);
        if (in_dtype == DataType::BFLOAT16) {
            for (float& v : x) {
                v = static_cast<float>(bfloat16(v));
            }
        }
        const TensorSpec spec(ttnn::Shape({rows, cols}), TensorLayout(in_dtype, PageConfig(Layout::TILE), {}));
        const ttnn::Tensor input = ttnn::Tensor::from_vector(x, spec).to_device(mesh_device_.get());

        using Op = ttnn::experimental::prim::QuantizeRowsOperation;
        const auto outputs = ttnn::device_operation::launch<Op>(
            ttnn::experimental::prim::QuantizeRowsParams{
                .dtype = f.dtype,
                .min_scale = kMinScale,
                .max_cores = max_cores,
                .output_mem_config = input.memory_config()},
            ttnn::experimental::prim::QuantizeRowsInputs{.input = input});
        const auto q = ttnn::bitcast(outputs[0], DataType::UINT8).cpu().to_vector<uint8_t>();
        const auto scale = outputs[1].cpu().to_vector<float>();
        EXPECT_EQ(outputs[1].logical_shape(), ttnn::Shape({rows, 1}));

        // ttnn.divide by the device's scale, then ttnn.typecast: SGLang's ops.
        const ttnn::Tensor chain = ttnn::typecast(
            ttnn::divide(ttnn::typecast(input, DataType::FLOAT32), outputs[1]), f.dtype);
        const auto chain_q = ttnn::bitcast(chain, DataType::UINT8).cpu().to_vector<uint8_t>();

        size_t wrong = 0;
        auto fail = [&](const std::string& what) {
            if (wrong++ < 8) {
                ADD_FAILURE() << rows << "x" << cols << " " << f.dtype << " cores<=" << max_cores << ": " << what;
            }
        };
        for (uint32_t m = 0; m < rows; ++m) {
            float amax = 0.0f;
            bool nan = false;
            for (uint32_t k = 0; k < cols; ++k) {
                float a = std::fabs(x[m * cols + k]);
                // The unpacker widens a BF16 NaN to infinity.
                if (std::isnan(a) && in_dtype == DataType::BFLOAT16) {
                    a = std::numeric_limits<float>::infinity();
                }
                nan |= std::isnan(a);
                amax = std::max(amax, a);
            }
            const float want_scale =
                nan ? std::numeric_limits<float>::quiet_NaN() : std::max(amax / f.max, kMinScale);
            if (std::bit_cast<uint32_t>(scale[m]) != std::bit_cast<uint32_t>(want_scale) &&
                !(std::isnan(scale[m]) && std::isnan(want_scale))) {
                fail("row " + std::to_string(m) + " scale " + std::to_string(scale[m]) + ", want " +
                     std::to_string(want_scale));
            }
            for (uint32_t k = 0; k < cols; ++k) {
                const size_t i = m * cols + k;
                if (q[i] != chain_q[i]) {
                    fail("q[" + std::to_string(m) + "][" + std::to_string(k) + "] " + std::to_string(q[i]) +
                         " != ttnn chain " + std::to_string(chain_q[i]) + " for " + std::to_string(x[i]));
                }
                // ttnn.divide's FP32 quotient can be one ulp off the IEEE one,
                // which moves a quotient on a float8 rounding midpoint.
                const float quotient = x[i] / want_scale;
                const uint8_t want = encode(f, quotient);
                auto matches = [&](uint8_t code) {
                    return q[i] == code || (is_nan_code(f, q[i]) && is_nan_code(f, code)) ||
                           ((q[i] & 0x7f) == 0 && (code & 0x7f) == 0);
                };
                const bool same = matches(want) || matches(encode(f, std::nextafter(quotient, -INFINITY))) ||
                                  matches(encode(f, std::nextafter(quotient, INFINITY)));
                if (!same) {
                    fail("q[" + std::to_string(m) + "][" + std::to_string(k) + "] " + std::to_string(q[i]) +
                         ", want " + std::to_string(want) + " for " + std::to_string(x[i]) + " / " +
                         std::to_string(want_scale));
                }
            }
        }
        return wrong;
    }

    std::shared_ptr<MeshDevice> mesh_device_;
};

// Activation-like rows: N(0, 1) with occasional large outliers, each row on its
// own scale.
std::vector<float> activations(uint32_t rows, uint32_t cols, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);
    std::vector<float> x(static_cast<size_t>(rows) * cols);
    for (uint32_t m = 0; m < rows; ++m) {
        const float row_scale = std::ldexp(1.0f, static_cast<int>(m % 13) - 6);
        for (uint32_t k = 0; k < cols; ++k) {
            float v = normal(rng) * row_scale;
            if (unit(rng) < 0.002f) {
                v *= 40.0f;
            }
            x[m * cols + k] = v;
        }
    }
    return x;
}

TEST_F(QuantizeRowsTest, MatchesTtnnOpsForEveryStride) {
    for (uint32_t rows : {1u, 2u, 3u, 4u, 5u, 8u, 9u, 13u, 16u, 17u, 31u, 32u}) {
        for (uint32_t cols : {32u, 96u, 1024u, 4096u}) {
            EXPECT_EQ(check(activations(rows, cols, rows * 7919 + cols), rows, cols, DataType::BFLOAT16, kE4M3, 0), 0u)
                << rows << "x" << cols;
        }
    }
}

TEST_F(QuantizeRowsTest, CoreCounts) {
    // One core, one NoC row, several NoC rows; dense tiles split unevenly.
    for (uint32_t max_cores : {1u, 2u, 7u, 13u, 24u, 64u, 128u}) {
        for (uint32_t rows : {1u, 6u, 16u, 32u}) {
            EXPECT_EQ(check(activations(rows, 12288, max_cores + rows), rows, 12288, DataType::BFLOAT16, kE4M3, max_cores), 0u)
                << rows << " rows, cores <= " << max_cores;
        }
    }
}

TEST_F(QuantizeRowsTest, Float32InputAndE5m2) {
    for (uint32_t rows : {1u, 4u, 20u}) {
        const auto x = activations(rows, 3072, 99 + rows);
        EXPECT_EQ(check(x, rows, 3072, DataType::FLOAT32, kE4M3, 32), 0u) << rows;
        EXPECT_EQ(check(x, rows, 3072, DataType::BFLOAT16, kE5M2, 32), 0u) << rows;
        EXPECT_EQ(check(x, rows, 3072, DataType::FLOAT32, kE5M2, 32), 0u) << rows;
    }
}

TEST_F(QuantizeRowsTest, SpecialRows) {
    const uint32_t rows = 8, cols = 1024;
    auto x = activations(rows, cols, 5);
    for (uint32_t k = 0; k < cols; ++k) {
        x[0 * cols + k] = 0.0f;                                   // all zero: the floor
        x[1 * cols + k] = k % 2 ? -1e-30f : 3e-12f;               // tiny: floor, subnormal codes
        x[3 * cols + k] = std::ldexp(1.0f, static_cast<int>(k % 40) - 20);  // every exponent
    }
    x[2 * cols + 700] = std::numeric_limits<float>::infinity();  // inf scale
    x[4 * cols + 5] = std::numeric_limits<float>::quiet_NaN();   // NaN row
    x[5 * cols + 1023] = -3.0e38f;                              // huge
    x[6 * cols + 0] = -0.0f;
    EXPECT_EQ(check(x, rows, cols, DataType::BFLOAT16, kE4M3, 32), 0u);
    EXPECT_EQ(check(x, rows, cols, DataType::FLOAT32, kE4M3, 3), 0u);
    EXPECT_EQ(check(x, rows, cols, DataType::BFLOAT16, kE5M2, 32), 0u);
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test

namespace tt::tt_metal::distributed::test {
namespace {
// One shape from QR_SHAPE=rows,cols,max_cores (for debugging); skipped otherwise.
TEST_F(QuantizeRowsTest, OneShapeFromEnv) {
    const char* env = std::getenv("QR_SHAPE");
    if (env == nullptr) {
        GTEST_SKIP() << "QR_SHAPE unset";
    }
    uint32_t rows = 0, cols = 0, cores = 0;
    ASSERT_EQ(std::sscanf(env, "%u,%u,%u", &rows, &cols, &cores), 3);
    if (std::getenv("QR_COORDS")) {
        const auto grid = mesh_device_->compute_with_storage_grid_size();
        for (uint32_t x = 0; x < grid.x; ++x) {
            const auto c = mesh_device_->worker_core_from_logical_core(CoreCoord{x, 0});
            std::printf("logical x %u -> noc (%zu, %zu)\n", x, c.x, c.y);
        }
        for (uint32_t y = 0; y < grid.y; ++y) {
            const auto c = mesh_device_->worker_core_from_logical_core(CoreCoord{0, y});
            std::printf("logical y %u -> noc (%zu, %zu)\n", y, c.x, c.y);
        }
    }
    EXPECT_EQ(check(activations(rows, cols, 1), rows, cols, DataType::BFLOAT16, kE4M3, cores), 0u);
}
}  // namespace
}  // namespace tt::tt_metal::distributed::test

namespace tt::tt_metal::distributed::test {
namespace {
// Device time per op, traced (QR_BENCH=rows,cols,max_cores[;...]); skipped otherwise.
TEST(QuantizeRowsBench, Traced) {
    const char* env = std::getenv("QR_BENCH");
    if (env == nullptr) {
        GTEST_SKIP() << "QR_BENCH unset";
    }
    auto mesh = MeshDevice::create(MeshDeviceConfig(MeshShape{1, 1}), DEFAULT_L1_SMALL_SIZE, 64 << 20);
    std::string spec(env);
    size_t pos = 0;
    while (pos < spec.size()) {
        const size_t end = spec.find(';', pos) == std::string::npos ? spec.size() : spec.find(';', pos);
        uint32_t rows = 0, cols = 0, cores = 0;
        ASSERT_EQ(std::sscanf(spec.substr(pos, end - pos).c_str(), "%u,%u,%u", &rows, &cols, &cores), 3);
        pos = end + 1;
        const auto x = activations(rows, cols, 3);
        const TensorSpec tspec(ttnn::Shape({rows, cols}), TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), {}));
        const ttnn::Tensor input = ttnn::Tensor::from_vector(x, tspec).to_device(mesh.get());
        using Op = ttnn::experimental::prim::QuantizeRowsOperation;
        const ttnn::experimental::prim::QuantizeRowsParams params{
            .dtype = DataType::FP8_E4M3, .min_scale = kMinScale, .max_cores = cores, .output_mem_config = input.memory_config()};
        auto run = [&] { return ttnn::device_operation::launch<Op>(params, ttnn::experimental::prim::QuantizeRowsInputs{.input = input}); };
        run();  // compile
        Finish(mesh->mesh_command_queue());
        constexpr int kOps = 100;
        std::vector<std::vector<ttnn::Tensor>> keep;
        keep.reserve(kOps);
        const MeshTraceId trace = mesh->begin_mesh_trace(mesh->mesh_command_queue());
        for (int i = 0; i < kOps; ++i) {
            keep.push_back(run());
        }
        mesh->end_mesh_trace(mesh->mesh_command_queue(), trace);
        mesh->replay_mesh_trace(mesh->mesh_command_queue(), trace, true);
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < 5; ++r) {
            mesh->replay_mesh_trace(mesh->mesh_command_queue(), trace, false);
        }
        Finish(mesh->mesh_command_queue());
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / (5 * kOps);
        mesh->release_mesh_trace(trace);
        std::printf("QR_BENCH %ux%u cores<=%u: %.2f us/op\n", rows, cols, cores, us);
    }
    mesh->close();
}
}  // namespace
}  // namespace tt::tt_metal::distributed::test
