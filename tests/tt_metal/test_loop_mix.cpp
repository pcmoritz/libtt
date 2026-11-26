// SPDX-License-Identifier: Apache-2.0

// ttnn::experimental::loop_mix: Inkling 2's loop-message controller against a
// double reference that rounds where the model does (the normalized inputs,
// hidden and the gate and forget pre-activations to BF16):
//   hidden = bf16([bf16(rms(dest)) | bf16(rms(msg))] @ down)
//   out = dest + (tanh(bf16(hidden @ gain)) * msg - tanh(bf16(hidden @ forget)) * dest)
// within 2e-3 of the row's magnitude, except that a few elements (0.5%) may
// be off by up to 1e-2 where an FP32 sum lands on the other side of a BF16
// rounding boundary from the double one and moves a gate by an ulp. Cases
// are Inkling-2-Mini's decode
// at TP4 (W = 2304, R = 64): a local layer's three Q/K/V streams from one
// destination, a global layer's three destinations, the MLP side's one, and
// four tokens.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "ttnn/operations/experimental/loop_mix/loop_mix.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

double bf16(double x) { return static_cast<float>(bfloat16(static_cast<float>(x))); }

class LoopMixTest : public ::testing::Test {
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

    size_t check(uint32_t Q, uint32_t Qd, uint32_t T, uint32_t W, uint32_t R, uint32_t seed, bool norm = false) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        std::vector<float> dest(Qd * T * W), msg(Q * T * W), down(Q * 2 * W * R), gain(Q * R * W), forget(Q * R * W);
        for (float& v : dest) {
            v = 3.0f * normal(rng);
        }
        for (float& v : msg) {
            v = normal(rng);
        }
        // Weights scaled so that the gates' pre-activations are O(1).
        for (float& v : down) {
            v = normal(rng) / std::sqrt(2.0f * W);
        }
        for (float& v : gain) {
            v = normal(rng) / std::sqrt(static_cast<float>(R));
        }
        for (float& v : forget) {
            v = normal(rng) / std::sqrt(static_cast<float>(R));
        }
        constexpr float eps = 1e-6f;
        std::vector<float> norm_weight(Q * W);
        for (float& v : norm_weight) {
            v = 1.0f + 0.2f * normal(rng);
        }
        std::optional<ttnn::Tensor> norm_tensor;
        if (norm) {
            norm_tensor = device_tensor(norm_weight, ttnn::Shape({Q, W}), DataType::FLOAT32);
        }
        const auto outs = ttnn::experimental::loop_mix(
            device_tensor(dest, ttnn::Shape({Qd, T, W}), DataType::FLOAT32),
            device_tensor(msg, ttnn::Shape({Q, T, W}), DataType::FLOAT32),
            device_tensor(down, ttnn::Shape({Q, 2 * W, R}), DataType::BFLOAT16),
            device_tensor(gain, ttnn::Shape({Q, R, W}), DataType::BFLOAT16),
            device_tensor(forget, ttnn::Shape({Q, R, W}), DataType::BFLOAT16),
            eps,
            norm_tensor,
            eps);
        // [Q, T, W] FP32, or with a norm Q outputs [T, W] BF16, gathered here
        // as one [Q, T, W].
        std::vector<float> got;
        if (norm) {
            EXPECT_EQ(outs.size(), Q);
            for (const auto& out : outs) {
                EXPECT_EQ(out.logical_shape(), ttnn::Shape({T, W}));
                const auto part = out.cpu().to_vector<float>();
                got.insert(got.end(), part.begin(), part.end());
            }
        } else {
            EXPECT_EQ(outs.size(), 1u);
            EXPECT_EQ(outs.at(0).logical_shape(), ttnn::Shape({Q, T, W}));
            got = outs.at(0).cpu().to_vector<float>();
        }

        size_t wrong = 0, flips = 0;
        auto rms_bf16 = [&](const float* x) {
            double sum = 0;
            for (uint32_t c = 0; c < W; ++c) {
                sum += double(x[c]) * x[c];
            }
            const double r = 1.0 / std::sqrt(sum / W + eps);
            std::vector<double> y(W);
            for (uint32_t c = 0; c < W; ++c) {
                y[c] = bf16(x[c] * r);
            }
            return y;
        };
        for (uint32_t q = 0; q < Q; ++q) {
            const uint32_t qd = Qd == 1 ? 0 : q;
            for (uint32_t t = 0; t < T; ++t) {
                const float* d = &dest[(qd * T + t) * W];
                const float* m = &msg[(q * T + t) * W];
                const auto dn = rms_bf16(d), mn = rms_bf16(m);
                std::vector<double> hidden(R);
                for (uint32_t r = 0; r < R; ++r) {
                    double h = 0;
                    for (uint32_t k = 0; k < W; ++k) {
                        h += dn[k] * down[(q * 2 * W + k) * R + r] + mn[k] * down[(q * 2 * W + W + k) * R + r];
                    }
                    hidden[r] = bf16(h);
                }
                std::vector<double> want_row(W);
                for (uint32_t c = 0; c < W; ++c) {
                    double g = 0, f = 0;
                    for (uint32_t r = 0; r < R; ++r) {
                        g += hidden[r] * gain[(q * R + r) * W + c];
                        f += hidden[r] * forget[(q * R + r) * W + c];
                    }
                    want_row[c] = d[c] + (std::tanh(bf16(g)) * m[c] - std::tanh(bf16(f)) * d[c]);
                }
                double scale = 0;
                if (norm) {
                    double sum = 0;
                    for (double v : want_row) {
                        sum += v * v;
                    }
                    const double rstd = 1.0 / std::sqrt(sum / W + eps);
                    for (uint32_t c = 0; c < W; ++c) {
                        want_row[c] *= rstd * norm_weight[q * W + c];
                        scale = std::max(scale, std::fabs(want_row[c]));
                    }
                } else {
                    for (uint32_t c = 0; c < W; ++c) {
                        scale = std::max(scale, std::fabs(double(d[c])) + std::fabs(double(m[c])));
                    }
                }
                for (uint32_t c = 0; c < W; ++c) {
                    const double want = want_row[c];
                    const double value = got[(q * T + t) * W + c];
                    const double error = std::fabs(value - want);
                    // A BF16 output also rounds by up to 2^-8 of itself.
                    flips += error > 2e-3 * scale + (norm ? std::ldexp(std::fabs(want), -7) : 0.0);
                    if (!(error <= 1e-2 * scale) && wrong++ < 8) {
                        ADD_FAILURE() << "Q " << Q << " Qd " << Qd << " T " << T << " [" << q << "][" << t << "]["
                                      << c << "] " << value << " != " << want;
                    }
                }
            }
        }
        EXPECT_LE(flips, got.size() / 200) << "Q " << Q << " Qd " << Qd << " T " << T;
        return wrong;
    }

    std::shared_ptr<MeshDevice> mesh_device_;
};

TEST_F(LoopMixTest, MatchesReference) {
    // Small widths first: one and two cores, R = 32.
    EXPECT_EQ(check(2, 2, 2, 64, 32, 5), 0u);
    EXPECT_EQ(check(1, 1, 3, 128, 64, 6), 0u);
    EXPECT_EQ(check(3, 1, 1, 2304, 64, 1), 0u);
    EXPECT_EQ(check(3, 3, 1, 2304, 64, 2), 0u);
    EXPECT_EQ(check(1, 1, 1, 2304, 64, 3), 0u);
    EXPECT_EQ(check(3, 1, 4, 2304, 64, 4), 0u);
    // With the norms that follow: per-destination BF16 outputs.
    EXPECT_EQ(check(3, 1, 1, 2304, 64, 7, true), 0u);
    EXPECT_EQ(check(1, 1, 1, 2304, 64, 8, true), 0u);
    EXPECT_EQ(check(3, 3, 2, 2304, 64, 9, true), 0u);
    EXPECT_EQ(check(2, 2, 2, 64, 32, 10, true), 0u);
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
