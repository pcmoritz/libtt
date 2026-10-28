// SPDX-License-Identifier: Apache-2.0

// Ops on at most 16 rows of a vocabulary (third_party/tt_metal/top_half_tiles.patch
// and libtt's rng_bits). A decode batch of logits, [2, 151936], fills the top
// halves of its 32 x 32 tiles, and the ops move and compute only those, as
// 16 x 32 tiles. Their rows match the same op on 33 rows, which takes whole
// tiles, bit for bit, also when the other operand is a column or scalar that
// the op broadcasts. 32 rows have the same tiles as 2 rows and run after them,
// so they check that the two take different cached programs. Times are device
// times of a replayed trace.

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/data_movement/copy/copy.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/experimental/rng_bits/rng_bits.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

constexpr uint32_t kVocab = 151936;
constexpr uint32_t kAllRows = 33;

class TopHalfTilesTest : public ::testing::Test {
protected:
    using Op = std::function<ttnn::Tensor(const ttnn::Tensor&, const ttnn::Tensor&)>;

    void SetUp() override {
        mesh_device_ = MeshDevice::create(MeshDeviceConfig(MeshShape{1, 1}), DEFAULT_L1_SMALL_SIZE, 64 << 20);
    }

    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
    }

    static std::vector<float> normal(size_t count, uint32_t seed) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> distribution(0.0f, 1.0f);
        std::vector<float> values(count);
        for (float& v : values) {
            v = distribution(rng);
        }
        return values;
    }

    template <typename T>
    ttnn::Tensor tensor(const std::vector<T>& values, const ttnn::Shape& shape, DataType dtype) {
        const TensorSpec spec(shape, TensorLayout(dtype, PageConfig(Layout::TILE), {}));
        return ttnn::Tensor::from_vector(values, spec).to_device(mesh_device_.get());
    }

    // The first `rows` rows of [kAllRows, width] values.
    ttnn::Tensor rows_of(const std::vector<float>& values, uint32_t rows, uint32_t width, DataType dtype) {
        return tensor(std::vector<float>(values.begin(), values.begin() + rows * width), ttnn::Shape({rows, width}), dtype);
    }

    static std::vector<uint32_t> bits(const ttnn::Tensor& t) {
        if (t.dtype() == DataType::UINT32) {
            return t.cpu().to_vector<uint32_t>();
        }
        const std::vector<float> values = t.cpu().to_vector<float>();
        std::vector<uint32_t> words(values.size());
        std::memcpy(words.data(), values.data(), values.size() * sizeof(float));
        return words;
    }

    // Device milliseconds per call of `op`, replayed from a trace as libtt runs a program.
    double time_ms(const std::function<ttnn::Tensor()>& op, int calls = 20, int replays = 5) {
        MeshCommandQueue& queue = mesh_device_->mesh_command_queue();
        op();
        Finish(queue);
        const MeshTraceId trace = mesh_device_->begin_mesh_trace(queue);
        for (int i = 0; i < calls; ++i) {
            op();
        }
        mesh_device_->end_mesh_trace(queue, trace);
        mesh_device_->replay_mesh_trace(queue, trace, /*blocking=*/true);
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < replays; ++i) {
            mesh_device_->replay_mesh_trace(queue, trace, /*blocking=*/false);
        }
        Finish(queue);
        const double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        mesh_device_->release_mesh_trace(trace);
        return elapsed / (calls * replays);
    }

    // `op` on the first 2 and then 32 rows of x and y matches the same rows of `op` on all of them. y is
    // [kAllRows, y_width] values.
    void check(const std::string& what, const Op& op, DataType dtype, uint32_t seed, uint32_t y_width = kVocab) {
        const std::vector<float> x = normal(kAllRows * kVocab, seed), y = normal(kAllRows * y_width, seed + 1);
        const auto inputs = [&](uint32_t rows) {
            return std::pair{rows_of(x, rows, kVocab, dtype), rows_of(y, rows, y_width, dtype)};
        };
        const auto all = inputs(kAllRows);
        const std::vector<uint32_t> want = bits(op(all.first, all.second));
        for (uint32_t rows : {2u, 32u}) {
            const auto some = inputs(rows);
            const std::vector<uint32_t> got = bits(op(some.first, some.second));
            ASSERT_EQ(got.size(), rows * kVocab) << what;
            EXPECT_TRUE(std::equal(got.begin(), got.end(), want.begin())) << what << ", " << rows << " rows";
            std::printf(
                "%s, %u rows: %.1f us\n", what.c_str(), rows, time_ms([&] { return op(some.first, some.second); }) * 1e3);
        }
    }

    std::shared_ptr<MeshDevice> mesh_device_;
};

const char* name(DataType dtype) { return dtype == DataType::FLOAT32 ? "f32" : "bf16"; }

TEST_F(TopHalfTilesTest, Binary) {
    for (DataType dtype : {DataType::FLOAT32, DataType::BFLOAT16}) {
        const ttnn::Tensor one = tensor(std::vector<float>{0.25f}, ttnn::Shape({1, 1}), dtype);
        const std::string type = name(dtype);
        check("multiply " + type, [](const auto& x, const auto& y) { return ttnn::multiply(x, y); }, dtype, 0);
        check("add " + type, [](const auto& x, const auto& y) { return ttnn::add(x, y); }, dtype, 2);
        check(
            "divide by column " + type, [](const auto& x, const auto& t) { return ttnn::divide(x, t); }, dtype, 4, 1);
        check(
            "subtract column " + type, [](const auto& x, const auto& t) { return ttnn::subtract(x, t); }, dtype, 6, 1);
        check("add scalar " + type, [&](const auto& x, const auto&) { return ttnn::add(x, one); }, dtype, 8);
        check("maximum of scalar " + type, [&](const auto& x, const auto&) { return ttnn::maximum(one, x); }, dtype, 10);
    }
}

TEST_F(TopHalfTilesTest, UnaryAndCopy) {
    for (DataType dtype : {DataType::FLOAT32, DataType::BFLOAT16}) {
        const std::string type = name(dtype);
        check("exp " + type, [](const auto& x, const auto&) { return ttnn::exp(x); }, dtype, 12);
        check("neg " + type, [](const auto& x, const auto&) { return ttnn::neg(x); }, dtype, 14);
        check("log " + type, [](const auto& x, const auto&) { return ttnn::log(ttnn::abs(x)); }, dtype, 16);
        check(
            "typecast to bf16 " + type,
            [](const auto& x, const auto&) { return ttnn::typecast(x, DataType::BFLOAT16); },
            dtype,
            18);
        check(
            "typecast to f32 " + type,
            [](const auto& x, const auto&) { return ttnn::typecast(x, DataType::FLOAT32); },
            dtype,
            20);
        check("copy " + type, [](const auto& x, const auto& slot) { return ttnn::copy(x, slot); }, dtype, 22);
    }
}

// The steps of a Gumbel-max draw on f32 rows, from random bits to scores, and log probabilities on bf16 rows.
TEST_F(TopHalfTilesTest, SamplerChain) {
    using ttnn::operations::unary::EltwiseUnaryWithParam;
    using ttnn::operations::unary::UnaryOpType;
    const ttnn::Shape scalar({1, 1});
    const ttnn::Tensor state = tensor(std::vector<uint32_t>{0xDEADBEEFu, 0, 3, 0}, ttnn::Shape({2, 2}), DataType::UINT32);
    const ttnn::Tensor nine = tensor(std::vector<uint32_t>{9}, scalar, DataType::UINT32);
    const ttnn::Tensor exponent = tensor(std::vector<uint32_t>{0x3F800000u}, scalar, DataType::UINT32);
    const ttnn::Tensor one = tensor(std::vector<float>{1.0f}, scalar, DataType::FLOAT32);
    const ttnn::Tensor tiny_value = tensor(std::vector<float>{1e-30f}, scalar, DataType::BFLOAT16);
    const std::vector<EltwiseUnaryWithParam> gumbel = {
        {UnaryOpType::LOG}, {UnaryOpType::NEG}, {UnaryOpType::LOG}, {UnaryOpType::NEG}};
    const std::vector<EltwiseUnaryWithParam> log = {{UnaryOpType::LOG}};

    // Element (r, c) of the bits is element r * W + c of the draw for every number of rows.
    const auto draw = [&](const ttnn::Tensor& x) {
        return ttnn::experimental::rng_bits(state, x.logical_shape(), ttnn::experimental::RngAlgorithm::Threefry).at(0);
    };
    const auto uniform = [&](const ttnn::Tensor& x) {
        const ttnn::Tensor mantissa = ttnn::bitwise_or(ttnn::logical_right_shift(draw(x), nine), exponent);
        return ttnn::subtract(ttnn::bitcast(mantissa, DataType::FLOAT32), one);
    };
    check("rng bits", [&](const auto& x, const auto&) { return draw(x); }, DataType::FLOAT32, 24);
    check("uniform", [&](const auto& x, const auto&) { return uniform(x); }, DataType::FLOAT32, 26);
    check(
        "gumbel scores",
        [&](const auto& x, const auto&) {
            return ttnn::add(uniform(x), x, std::nullopt, std::nullopt, std::nullopt, {}, gumbel);
        },
        DataType::FLOAT32,
        28);
    check(
        "log probabilities",
        [&](const auto& x, const auto&) {
            return ttnn::maximum(tiny_value, ttnn::exp(x), std::nullopt, std::nullopt, std::nullopt, {}, {}, log);
        },
        DataType::BFLOAT16,
        30);
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
