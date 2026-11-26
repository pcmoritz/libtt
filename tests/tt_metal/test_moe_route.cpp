// SPDX-License-Identifier: Apache-2.0

// ttnn::experimental::moe_route: a router's top-k selection (sigmoid scores,
// top k of score + bias, the selected scores renormalized times a scale)
// against a double reference: the ids exactly (best first, the lower expert
// first among equal keys), the weights within 1e-5 relative; rows from
// num_valid on get weight 0 and id -1.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "ttnn/operations/experimental/moe_route/moe_route.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

class MoeRouteTest : public ::testing::Test {
protected:
    void SetUp() override { mesh_device_ = MeshDevice::create(MeshDeviceConfig(MeshShape{1, 1})); }

    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
    }

    template <typename T>
    ttnn::Tensor device_tensor(const std::vector<T>& values, const ttnn::Shape& shape, DataType dtype) {
        const TensorSpec spec(shape, TensorLayout(dtype, PageConfig(Layout::TILE), {}));
        return ttnn::Tensor::from_vector(values, spec).to_device(mesh_device_.get());
    }

    size_t check(uint32_t T, uint32_t E, uint32_t k, DataType bias_dtype, std::optional<int32_t> valid, bool ties,
                 uint32_t seed) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        std::vector<float> logits(T * E), bias(E);
        for (float& v : logits) {
            v = 3.0f * normal(rng);
        }
        for (float& v : bias) {
            v = 0.1f * normal(rng);
            if (bias_dtype == DataType::BFLOAT16) {
                v = static_cast<float>(bfloat16(v));
            }
        }
        if (ties) {
            // Experts 5 and 9, and 40 and 41, tie on every row (and win).
            for (uint32_t t = 0; t < T; ++t) {
                logits[t * E + 5] = logits[t * E + 9] = 8.0f;
                logits[t * E + 40] = logits[t * E + 41] = 7.5f;
            }
            bias[5] = bias[9] = 0.0f;
            bias[40] = bias[41] = 0.0f;
        }
        const float scale = 2.5f;
        std::optional<ttnn::Tensor> valid_tensor;
        if (valid.has_value()) {
            valid_tensor = device_tensor(std::vector<int32_t>{*valid}, ttnn::Shape({1}), DataType::INT32);
        }
        const auto out = ttnn::experimental::moe_route(
            device_tensor(logits, ttnn::Shape({T, E}), DataType::FLOAT32),
            device_tensor(bias, ttnn::Shape({E}), bias_dtype),
            k,
            scale,
            valid_tensor);
        const auto weights = out.at(0).cpu().to_vector<float>();
        const auto ids = out.at(1).cpu().to_vector<int32_t>();
        size_t wrong = 0;
        for (uint32_t t = 0; t < T; ++t) {
            const bool pad = valid.has_value() && static_cast<int32_t>(t) >= *valid;
            std::vector<double> s(E), key(E);
            for (uint32_t e = 0; e < E; ++e) {
                s[e] = 1.0 / (1.0 + std::exp(-static_cast<double>(logits[t * E + e])));
                // Keys as the device rounds them (an FP32 sum), so equal keys tie.
                key[e] = static_cast<float>(static_cast<float>(s[e]) + bias[e]);
            }
            std::vector<uint32_t> order(E);
            std::iota(order.begin(), order.end(), 0u);
            std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return key[a] > key[b]; });
            double total = 0;
            for (uint32_t i = 0; i < k; ++i) {
                total += s[order[i]];
            }
            for (uint32_t i = 0; i < k; ++i) {
                const int32_t want_id = pad ? -1 : static_cast<int32_t>(order[i]);
                const double want_w = pad ? 0.0 : s[order[i]] / total * scale;
                const int32_t got_id = ids[t * k + i];
                const double got_w = weights[t * k + i];
                if ((got_id != want_id || !(std::fabs(got_w - want_w) <= 1e-5 * std::fabs(want_w) + 1e-7)) &&
                    wrong++ < 8) {
                    ADD_FAILURE() << "T " << T << " E " << E << " k " << k << " [" << t << "][" << i << "] id "
                                  << got_id << " (want " << want_id << ") weight " << got_w << " (want " << want_w
                                  << ")";
                }
            }
        }
        return wrong;
    }

    std::shared_ptr<MeshDevice> mesh_device_;
};

TEST_F(MoeRouteTest, MatchesReference) {
    // Inkling-2-Mini: 128 experts, top 8, one and four tokens.
    EXPECT_EQ(check(1, 128, 8, DataType::FLOAT32, std::nullopt, false, 1), 0u);
    EXPECT_EQ(check(4, 128, 8, DataType::FLOAT32, std::nullopt, false, 2), 0u);
    // Padding rows, ties, a BF16 bias, other sizes.
    EXPECT_EQ(check(4, 128, 8, DataType::FLOAT32, 2, false, 3), 0u);
    EXPECT_EQ(check(1, 128, 8, DataType::FLOAT32, 1, true, 4), 0u);
    EXPECT_EQ(check(3, 64, 4, DataType::BFLOAT16, std::nullopt, false, 5), 0u);
    EXPECT_EQ(check(2, 256, 16, DataType::FLOAT32, std::nullopt, false, 6), 0u);
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
