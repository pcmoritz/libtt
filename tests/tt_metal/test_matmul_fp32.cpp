// SPDX-License-Identifier: Apache-2.0

// ttnn::experimental::matmul_fp32 multiplies FLOAT32 matrices on the vector
// unit. Its results must be FP32 accurate (the matrix unit's ttnn::matmul is
// off by about 1e-3 relative), and infinities and NaNs must only reach the
// outputs whose dot products contain them.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "ttnn/operations/experimental/matmul_fp32/matmul_fp32.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

struct Case {
    std::vector<uint32_t> a_shape;  // [..., M, K]
    std::vector<uint32_t> b_shape;  // [K, N] or [..., K, N]
};

class MatmulFp32Test : public ::testing::TestWithParam<Case> {
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

    std::shared_ptr<MeshDevice> mesh_device_;
};

size_t volume(const std::vector<uint32_t>& shape) {
    size_t v = 1;
    for (uint32_t d : shape) {
        v *= d;
    }
    return v;
}

std::vector<float> random_values(size_t count, std::mt19937& rng) {
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> values(count);
    for (float& value : values) {
        value = normal(rng);
    }
    return values;
}

struct Reference {
    std::vector<double> value;    // exact a @ b (up to double rounding)
    std::vector<double> abs_sum;  // sum of |a_ik * b_kj|, which bounds the FP32 rounding error
};

// a @ b in double, with b broadcast over a's batch when it has none.
Reference reference_matmul(
    const std::vector<float>& a, const std::vector<uint32_t>& a_shape, const std::vector<float>& b,
    const std::vector<uint32_t>& b_shape) {
    const size_t M = a_shape[a_shape.size() - 2], K = a_shape.back(), N = b_shape.back();
    const size_t batch = volume(a_shape) / (M * K);
    const bool bcast_b = b_shape.size() == 2;
    Reference ref{std::vector<double>(batch * M * N), std::vector<double>(batch * M * N)};
    for (size_t n = 0; n < batch; ++n) {
        const float* an = a.data() + n * M * K;
        const float* bn = b.data() + (bcast_b ? 0 : n * K * N);
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                double sum = 0.0, abs_sum = 0.0;
                for (size_t k = 0; k < K; ++k) {
                    const double p = static_cast<double>(an[i * K + k]) * bn[k * N + j];
                    sum += p;
                    abs_sum += std::abs(p);
                }
                ref.value[(n * M + i) * N + j] = sum;
                ref.abs_sum[(n * M + i) * N + j] = abs_sum;
            }
        }
    }
    return ref;
}

TEST_P(MatmulFp32Test, MatchesFp32Reference) {
    const auto& [a_shape, b_shape] = GetParam();
    std::mt19937 rng(volume(a_shape) * 31 + volume(b_shape));
    const std::vector<float> a = random_values(volume(a_shape), rng);
    const std::vector<float> b = random_values(volume(b_shape), rng);

    const ttnn::Tensor c_device =
        ttnn::experimental::matmul_fp32(to_device(a, a_shape), to_device(b, b_shape));
    std::vector<uint32_t> c_shape = a_shape;
    c_shape.back() = b_shape.back();
    EXPECT_EQ(c_device.logical_shape(), ttnn::Shape(c_shape));
    EXPECT_EQ(c_device.dtype(), DataType::FLOAT32);
    const std::vector<float> c = c_device.cpu().to_vector<float>();

    const Reference ref = reference_matmul(a, a_shape, b, b_shape);
    ASSERT_EQ(c.size(), ref.value.size());
    const double K = a_shape.back();
    double worst = 0.0;
    for (size_t i = 0; i < c.size(); ++i) {
        // A K-term FP32 dot product with one rounding per step is within about
        // K * 2^-24 of sum |a_ik * b_kj|; random signs make it far smaller.
        const double error = std::abs(c[i] - ref.value[i]) / std::max(ref.abs_sum[i], 1e-30);
        worst = std::max(worst, error);
        ASSERT_LE(error, K * std::ldexp(1.0, -24)) << "element " << i << ": " << c[i] << " vs " << ref.value[i];
    }
    char worst_text[32];
    std::snprintf(worst_text, sizeof(worst_text), "%.3g", worst);
    RecordProperty("worst_relative_error", worst_text);
}

INSTANTIATE_TEST_SUITE_P(
    Shapes,
    MatmulFp32Test,
    ::testing::Values(
        Case{{32, 32}, {32, 32}},
        Case{{64, 96}, {96, 128}},
        Case{{32, 512}, {512, 32}},
        // b broadcast over a's batch, and b with a's batch.
        Case{{2, 3, 64, 32}, {32, 96}},
        Case{{2, 64, 64}, {2, 64, 32}},
        // Not tile aligned.
        Case{{30, 50}, {50, 20}},
        // 520 output tiles: several runs per core, crossing output rows (13 tiles wide).
        Case{{1280, 64}, {64, 416}},
        // The A row kept in L1 for a run of two tiles: double buffered up to
        // Kt = 128, single buffered up to Kt = 256, streamed one tile per k beyond.
        Case{{32, 4096}, {4096, 64}},
        Case{{32, 4128}, {4128, 64}},
        Case{{32, 8192}, {8192, 64}},
        Case{{32, 8224}, {8224, 64}}),
    [](const ::testing::TestParamInfo<Case>& info) {
        auto dims = [](const std::vector<uint32_t>& shape) {
            std::string s;
            for (uint32_t d : shape) {
                s += (s.empty() ? "" : "x") + std::to_string(d);
            }
            return s;
        };
        return dims(info.param.a_shape) + "_" + dims(info.param.b_shape);
    });

TEST_F(MatmulFp32Test, InfinitiesAndNaNsStayInTheirRowsAndColumns) {
    constexpr uint32_t M = 64, K = 64, N = 64;
    std::vector<float> a(M * K), b(K * N);
    for (uint32_t i = 0; i < M * K; ++i) {
        a[i] = 0.25f + static_cast<float>(i % 7) * 0.125f;  // all positive
    }
    for (uint32_t i = 0; i < K * N; ++i) {
        b[i] = 0.5f + static_cast<float>(i % 5) * 0.25f;  // all positive
    }
    // A NaN in row 3 of a (column 37, past the first k tile), and +inf in
    // column 18 of b (row 9): every output of row 3 is NaN, every other output
    // of column 18 is +inf, and nothing else changes.
    a[3 * K + 37] = std::numeric_limits<float>::quiet_NaN();
    b[9 * N + 18] = std::numeric_limits<float>::infinity();

    const std::vector<float> c =
        ttnn::experimental::matmul_fp32(to_device(a, {M, K}), to_device(b, {K, N})).cpu().to_vector<float>();
    const Reference ref = reference_matmul(a, {M, K}, b, {K, N});
    for (uint32_t i = 0; i < M; ++i) {
        for (uint32_t j = 0; j < N; ++j) {
            const float value = c[i * N + j];
            if (i == 3) {
                EXPECT_TRUE(std::isnan(value)) << i << ", " << j << ": " << value;
            } else if (j == 18) {
                EXPECT_EQ(value, std::numeric_limits<float>::infinity()) << i << ", " << j;
            } else {
                EXPECT_NEAR(value, ref.value[i * N + j], K * std::ldexp(1.0, -24) * ref.abs_sum[i * N + j])
                    << i << ", " << j;
            }
        }
    }
}

// A K that is not a multiple of 32 pads the last k tile of both inputs. The
// padding of a tensor that an earlier operation produced can hold anything,
// including infinities and NaNs, and none of it may reach the outputs.
TEST_F(MatmulFp32Test, PaddingOfTheContractionDimensionIsIgnored) {
    constexpr uint32_t M = 40, K = 50, N = 70;
    std::mt19937 rng(7);
    const std::vector<float> a = random_values(M * K, rng);
    const std::vector<float> b = random_values(K * N, rng);

    // The tensor with `poison` in its padding: the padding of the tiled host
    // data is where a tensor of ones has zeros.
    auto poisoned = [&](const std::vector<float>& values, const std::vector<uint32_t>& shape, float poison) {
        const TensorSpec spec(
            ttnn::Shape(shape), TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), MemoryConfig{}));
        ttnn::Tensor host = ttnn::Tensor::from_vector(values, spec);
        ttnn::Tensor ones = ttnn::Tensor::from_vector(std::vector<float>(values.size(), 1.0f), spec);
        const auto data = host_buffer::get_as<float>(host);
        const auto mask = host_buffer::get_as<float>(ones);
        std::vector<float> physical(data.begin(), data.end());
        size_t padding = 0;
        for (size_t i = 0; i < physical.size(); ++i) {
            if (mask[i] == 0.0f) {
                physical[i] = poison;
                ++padding;
            }
        }
        EXPECT_GT(padding, 0u);
        return ttnn::Tensor(HostBuffer(std::move(physical)), host.tensor_spec()).to_device(mesh_device_.get());
    };

    const ttnn::Tensor c_device = ttnn::experimental::matmul_fp32(
        poisoned(a, {M, K}, std::numeric_limits<float>::infinity()),
        poisoned(b, {K, N}, std::numeric_limits<float>::quiet_NaN()));
    const std::vector<float> c = c_device.cpu().to_vector<float>();
    const Reference ref = reference_matmul(a, {M, K}, b, {K, N});
    ASSERT_EQ(c.size(), ref.value.size());
    for (size_t i = 0; i < c.size(); ++i) {
        ASSERT_TRUE(std::isfinite(c[i])) << "element " << i << ": " << c[i];
        ASSERT_LE(std::abs(c[i] - ref.value[i]), K * std::ldexp(1.0, -24) * ref.abs_sum[i]) << "element " << i;
    }
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
