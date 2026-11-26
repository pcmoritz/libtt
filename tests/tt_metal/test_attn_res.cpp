// SPDX-License-Identifier: Apache-2.0

// ttnn::experimental::attn_res: Inkling 2's block attention residuals (softmax
// over the sources of their products with each query, scaled by the sources'
// inverse RMS, then the RMS norm of the mixture times a weight) against a
// double reference: within a BF16 ulp for a BF16 output and 1e-5 relative for
// FP32. Cases cover one to eight sources, one and three queries, one to four
// tokens, the addend, normalized and raw values, no weight (the FP32 mixture),
// FP32 sources and widths of one core's tiles up to Inkling's 2304.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "ttnn/operations/experimental/attn_res/attn_res.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

struct Case {
    uint32_t sources;
    uint32_t queries;
    uint32_t tokens;
    uint32_t width;
    bool addend;
    bool weight;
    bool normalize;
    DataType dtype;
};

class AttnResTest : public ::testing::Test {
protected:
    void SetUp() override { mesh_device_ = MeshDevice::create(MeshDeviceConfig(MeshShape{1, 1})); }

    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
    }

    ttnn::Tensor device_tensor(std::vector<float>& values, const ttnn::Shape& shape, DataType dtype) {
        if (dtype == DataType::BFLOAT16) {
            for (float& v : values) {
                v = static_cast<float>(bfloat16(v));
            }
        }
        const TensorSpec spec(shape, TensorLayout(dtype, PageConfig(Layout::TILE), {}));
        return ttnn::Tensor::from_vector(values, spec).to_device(mesh_device_.get());
    }

    // Returns the number of elements off the reference.
    size_t check(const Case& k, uint32_t seed) {
        const uint32_t T = k.tokens, W = k.width, Q = k.queries, n = k.sources;
        std::mt19937 rng(seed);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        std::vector<std::vector<float>> s(n, std::vector<float>(T * W));
        std::vector<float> addend(T * W), q(Q * W), w(Q * W);
        for (uint32_t j = 0; j < n; ++j) {
            // Sources of different scales, as the residual blocks grow.
            const float scale = std::ldexp(1.0f, static_cast<int>(j % 5) - 2);
            for (float& v : s[j]) {
                v = normal(rng) * scale;
            }
        }
        for (float& v : addend) {
            v = 0.5f * normal(rng);
        }
        // Query qi leans towards source qi % n's first token, a logit of
        // about 3 there and noise elsewhere, so the softmax is far from
        // uniform.
        for (uint32_t qi = 0; qi < Q; ++qi) {
            const auto& toward = s[qi % n];
            double norm = 0;
            for (uint32_t c = 0; c < W; ++c) {
                norm += static_cast<double>(toward[c]) * toward[c];
            }
            const double a = 3.0 / std::sqrt(norm * W);
            for (uint32_t c = 0; c < W; ++c) {
                q[qi * W + c] = static_cast<float>(a * toward[c] + 0.5 * normal(rng) / W);
            }
        }
        for (float& v : w) {
            v = 1.0f + 0.2f * normal(rng);
        }
        const ttnn::Shape row_shape({T, W});
        std::vector<ttnn::Tensor> sources;
        for (auto& values : s) {
            sources.push_back(device_tensor(values, row_shape, k.dtype));
        }
        std::optional<ttnn::Tensor> addend_tensor;
        if (k.addend) {
            addend_tensor = device_tensor(addend, row_shape, DataType::BFLOAT16);
        }
        const auto queries = device_tensor(q, ttnn::Shape({Q, W}), DataType::FLOAT32);
        std::optional<ttnn::Tensor> weight;
        if (k.weight) {
            weight = device_tensor(w, ttnn::Shape({Q, W}), DataType::FLOAT32);
        }
        constexpr float eps_res = 1e-6f, eps_norm = 1e-6f;
        const auto out =
            ttnn::experimental::attn_res(sources, addend_tensor, queries, weight, eps_res, eps_norm, k.normalize);
        const DataType out_dtype = k.weight ? DataType::BFLOAT16 : DataType::FLOAT32;
        EXPECT_EQ(out.logical_shape(), ttnn::Shape({Q, T, W}));
        EXPECT_EQ(out.dtype(), out_dtype);
        const auto got = out.cpu().to_vector<float>();

        size_t wrong = 0;
        for (uint32_t t = 0; t < T; ++t) {
            std::vector<std::vector<double>> v(n, std::vector<double>(W));
            std::vector<double> r(n);
            for (uint32_t j = 0; j < n; ++j) {
                double sum = 0;
                for (uint32_t c = 0; c < W; ++c) {
                    v[j][c] = s[j][t * W + c] + (k.addend && j == n - 1 ? addend[t * W + c] : 0.0);
                    sum += v[j][c] * v[j][c];
                }
                r[j] = 1.0 / std::sqrt(sum / W + eps_res);
            }
            for (uint32_t qi = 0; qi < Q; ++qi) {
                std::vector<double> logit(n);
                double top = -1e300;
                for (uint32_t j = 0; j < n; ++j) {
                    double dot = 0;
                    for (uint32_t c = 0; c < W; ++c) {
                        dot += v[j][c] * q[qi * W + c];
                    }
                    logit[j] = dot * r[j];
                    top = std::max(top, logit[j]);
                }
                double total = 0;
                for (double& l : logit) {
                    l = std::exp(l - top);
                    total += l;
                }
                std::vector<double> mix(W, 0.0);
                for (uint32_t j = 0; j < n; ++j) {
                    const double p = logit[j] / total * (k.normalize ? r[j] : 1.0);
                    for (uint32_t c = 0; c < W; ++c) {
                        mix[c] += p * v[j][c];
                    }
                }
                double rstd = 1.0;
                if (k.weight) {
                    double sum = 0;
                    for (double m : mix) {
                        sum += m * m;
                    }
                    rstd = 1.0 / std::sqrt(sum / W + eps_norm);
                }
                for (uint32_t c = 0; c < W; ++c) {
                    const double want = k.weight ? mix[c] * rstd * w[qi * W + c] : mix[c];
                    const double value = got[(qi * T + t) * W + c];
                    const double tolerance =
                        k.weight ? std::ldexp(std::fabs(want), -7) + 1e-30 : std::fabs(want) * 2e-5 + 1e-6;
                    if (!(std::fabs(value - want) <= tolerance) && wrong++ < 8) {
                        ADD_FAILURE() << "n " << n << " Q " << Q << " T " << T << " W " << W << " [" << qi << "]["
                                      << t << "][" << c << "] " << value << " != " << want;
                    }
                }
            }
        }
        return wrong;
    }

    std::shared_ptr<MeshDevice> mesh_device_;
};

TEST_F(AttnResTest, MatchesReference) {
    const std::vector<Case> cases = {
        // Inkling-2-Mini decode: a local layer's attention input, an MLP
        // input with the attention output added, a global layer's Q/K/V.
        {3, 1, 1, 2304, false, true, true, DataType::BFLOAT16},
        {4, 1, 1, 2304, true, true, true, DataType::BFLOAT16},
        {5, 3, 1, 2304, false, true, true, DataType::BFLOAT16},
        // Four tokens, every source count.
        {1, 1, 4, 2304, false, true, true, DataType::BFLOAT16},
        {8, 3, 4, 2304, true, true, true, DataType::BFLOAT16},
        {2, 2, 3, 2304, true, true, true, DataType::BFLOAT16},
        // Small widths: one core, and a last core with one tile.
        {3, 1, 2, 64, false, true, true, DataType::BFLOAT16},
        {6, 3, 1, 96, true, true, true, DataType::BFLOAT16},
        // Raw values, no weight (the FP32 mixture), FP32 sources.
        {4, 3, 2, 2304, false, true, false, DataType::BFLOAT16},
        {3, 3, 1, 2304, false, false, true, DataType::BFLOAT16},
        {5, 1, 4, 2304, true, false, true, DataType::FLOAT32},
        // Core counts around a gather tile's eight slots (7, 8, 16, 17 cores)
        // and a grid row of 11 (11, 12).
        {3, 1, 1, 448, false, true, true, DataType::BFLOAT16},
        {3, 1, 1, 512, false, true, true, DataType::BFLOAT16},
        {3, 1, 1, 1024, false, true, true, DataType::BFLOAT16},
        {3, 1, 1, 1088, false, true, true, DataType::BFLOAT16},
        {3, 1, 1, 704, false, true, true, DataType::BFLOAT16},
        {3, 1, 1, 768, false, true, true, DataType::BFLOAT16},
    };
    // ATTN_RES_CASE=i runs case i alone (for debugging a hang).
    const char* only = std::getenv("ATTN_RES_CASE");
    for (uint32_t i = 0; i < cases.size(); ++i) {
        if (only == nullptr || std::atoi(only) == static_cast<int>(i)) {
            EXPECT_EQ(check(cases[i], i + 1), 0u) << "case " << i;
        }
    }
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
