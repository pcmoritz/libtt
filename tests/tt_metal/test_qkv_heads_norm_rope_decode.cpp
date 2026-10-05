// SPDX-License-Identifier: Apache-2.0

// nlp_create_qkv_heads_decode_norm_rope splits a decode step's fused QKV
// projection into heads, normalizes the q and k heads (a weight row per head)
// and applies the rotary embedding in one program, also with an output gate
// after every q head and with a rotary embedding of part of each head, as in
// Qwen3.5. Its outputs must match the separate steps.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include <tt-metalium/mesh_device.hpp>

#include "ttnn/operations/transformer/qkv_heads_norm_rope_decode/qkv_heads_norm_rope_decode.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

struct Case {
    uint32_t users;
    uint32_t num_q_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim;
    // 0 rotates the whole head.
    uint32_t rotary_dim = 0;
    bool gated_query = false;
};

class QKVHeadsNormRopeDecodeTest : public ::testing::TestWithParam<Case> {
protected:
    void SetUp() override { mesh_device_ = MeshDevice::create(MeshDeviceConfig(MeshShape{1, 1})); }

    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
    }

    // A BF16 TILE tensor in DRAM with `values` rounded to BF16, which are
    // returned rounded so the reference sees what the device does.
    ttnn::Tensor to_device(std::vector<float>& values, const ttnn::Shape& shape) {
        const TensorSpec spec(shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), {}));
        ttnn::Tensor host = ttnn::Tensor::from_vector(values, spec);
        values = host.to_vector<float>();
        return host.to_device(mesh_device_.get());
    }

    std::shared_ptr<MeshDevice> mesh_device_;
};

std::vector<float> random_values(size_t count, float scale, std::mt19937& rng) {
    std::normal_distribution<float> normal(0.0f, scale);
    std::vector<float> values(count);
    for (float& value : values) {
        value = normal(rng);
    }
    return values;
}

TEST_P(QKVHeadsNormRopeDecodeTest, MatchesSeparateSteps) {
    const auto [users, num_q_heads, num_kv_heads, head_dim, rotary_dim_param, gated_query] = GetParam();
    const uint32_t rotary_dim = rotary_dim_param ? rotary_dim_param : head_dim;
    const uint32_t q_stride = (gated_query ? 2 : 1) * head_dim;
    const uint32_t width = num_q_heads * q_stride + 2 * num_kv_heads * head_dim;
    // The column of q head h, of k head h at num_q_heads + h, and of v head h
    // at num_q_heads + num_kv_heads + h.
    auto column = [&](uint32_t head) {
        return head < num_q_heads ? head * q_stride : num_q_heads * q_stride + (head - num_q_heads) * head_dim;
    };
    const float epsilon = 1e-6f;
    std::mt19937 rng(users * 1000 + num_q_heads * 10 + head_dim);

    // Two runs: the second reuses the cached program with tensors at new addresses.
    for (int run = 0; run < 2; ++run) {
        std::vector<float> qkv = random_values(static_cast<size_t>(users) * width, 2.0f, rng);
        // A different weight row per q and k head, q heads first.
        std::vector<float> weight = random_values(static_cast<size_t>(num_q_heads + num_kv_heads) * head_dim, 1.0f, rng);
        std::vector<float> cos(rotary_dim), sin(rotary_dim);
        for (uint32_t i = 0; i < rotary_dim; ++i) {
            const float angle = 0.37f * static_cast<float>(i % (rotary_dim / 2) + 1) * static_cast<float>(run + 3);
            cos[i] = std::cos(angle);
            sin[i] = std::sin(angle);
        }
        const ttnn::Tensor qkv_device = to_device(qkv, ttnn::Shape({1, 1, users, width}));
        const ttnn::Tensor weight_device = to_device(weight, ttnn::Shape({num_q_heads + num_kv_heads, head_dim}));
        const ttnn::Tensor cos_device = to_device(cos, ttnn::Shape({1, 1, 1, rotary_dim}));
        const ttnn::Tensor sin_device = to_device(sin, ttnn::Shape({1, 1, 1, rotary_dim}));

        const std::vector<ttnn::Tensor> outputs = ttnn::transformer::nlp_create_qkv_heads_decode_norm_rope(
            qkv_device, weight_device, cos_device, sin_device, num_q_heads, num_kv_heads, epsilon, rotary_dim_param,
            gated_query);
        ASSERT_EQ(outputs.size(), 3u);

        // Head `head` of user `user`; q and k heads use weight row `weight_row`.
        auto reference = [&](uint32_t user, uint32_t head, uint32_t weight_row, bool rotate) {
            const float* x = qkv.data() + static_cast<size_t>(user) * width + column(head);
            std::vector<float> out(x, x + head_dim);
            if (!rotate) {
                return out;
            }
            double sum = 0.0;
            for (uint32_t i = 0; i < head_dim; ++i) {
                sum += static_cast<double>(x[i]) * x[i];
            }
            const float rstd = 1.0f / std::sqrt(static_cast<float>(sum / head_dim) + epsilon);
            std::vector<float> normed(head_dim);
            for (uint32_t i = 0; i < head_dim; ++i) {
                normed[i] = x[i] * rstd * weight[static_cast<size_t>(weight_row) * head_dim + i];
            }
            const uint32_t half = rotary_dim / 2;
            for (uint32_t i = 0; i < head_dim; ++i) {
                if (i >= rotary_dim) {
                    out[i] = normed[i];
                    continue;
                }
                const float rotated = i < half ? -normed[i + half] : normed[i - half];
                out[i] = normed[i] * cos[i] + rotated * sin[i];
            }
            return out;
        };
        auto check = [&](const ttnn::Tensor& output, uint32_t heads, uint32_t first_head, uint32_t first_weight_row,
                         bool rotate, const char* name) {
            EXPECT_EQ(output.logical_shape(), ttnn::Shape({1, users, heads, head_dim})) << name;
            EXPECT_EQ(output.memory_config().memory_layout(), TensorMemoryLayout::HEIGHT_SHARDED) << name;
            const std::vector<float> values = output.cpu().to_vector<float>();
            ASSERT_EQ(values.size(), static_cast<size_t>(users) * heads * head_dim) << name;
            for (uint32_t user = 0; user < users; ++user) {
                for (uint32_t head = 0; head < heads; ++head) {
                    const std::vector<float> expected =
                        reference(user, first_head + head, first_weight_row + head, rotate);
                    for (uint32_t i = 0; i < head_dim; ++i) {
                        const float actual = values[(static_cast<size_t>(user) * heads + head) * head_dim + i];
                        // BF16 output: 8 bits of mantissa, on values up to a few.
                        ASSERT_NEAR(actual, expected[i], rotate ? 0.04f + 0.01f * std::abs(expected[i]) : 0.0f)
                            << name << " user " << user << " head " << head << " element " << i << " run " << run;
                    }
                }
            }
        };
        check(outputs[0], num_q_heads, 0, 0, true, "q");
        check(outputs[1], num_kv_heads, num_q_heads, num_q_heads, true, "k");
        check(outputs[2], num_kv_heads, num_q_heads + num_kv_heads, 0, false, "v");
    }
}

INSTANTIATE_TEST_SUITE_P(
    Shapes,
    QKVHeadsNormRopeDecodeTest,
    ::testing::Values(
        // Qwen3-8B at TP4, TP2 and TP1, Qwen3-32B at TP4, and Qwen3-14B at TP1
        // (q heads over two tile rows, k heads in the second).
        Case{1, 8, 2, 128},
        Case{1, 16, 4, 128},
        Case{1, 32, 8, 128},
        Case{1, 16, 2, 128},
        Case{1, 40, 8, 128},
        // k heads across the boundary of two tile rows.
        Case{1, 24, 16, 128},
        // Several users, users in the lower faces of the tiles, and wider heads.
        Case{3, 8, 2, 128},
        Case{20, 8, 2, 128},
        Case{2, 4, 1, 256},
        // Qwen3.5-9B at TP4 and TP1 and Qwen3.8-27B at TP4: a gate after
        // every q head and a rotary embedding of the first 64 elements.
        Case{1, 4, 1, 256, 64, true},
        Case{1, 16, 4, 256, 64, true},
        Case{1, 6, 1, 256, 64, true},
        // Each on its own, with several users.
        Case{3, 4, 1, 256, 0, true},
        Case{3, 8, 2, 128, 64, false},
        Case{20, 4, 1, 256, 128, true}),
    [](const ::testing::TestParamInfo<Case>& info) {
        const Case& c = info.param;
        return "users" + std::to_string(c.users) + "_q" + std::to_string(c.num_q_heads) + "_kv" +
               std::to_string(c.num_kv_heads) + "_d" + std::to_string(c.head_dim) +
               (c.rotary_dim ? "_rot" + std::to_string(c.rotary_dim) : "") + (c.gated_query ? "_gated" : "");
    });

}  // namespace
}  // namespace tt::tt_metal::distributed::test
