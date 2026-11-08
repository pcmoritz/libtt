// SPDX-License-Identifier: Apache-2.0

// split_qkv_norm_rope_decode runs a decode step's attention prologue as
// PyTorch models trace it in one program: split the fused QKV projection into
// heads (q heads with their gates in gated attention), rms_norm the q and k
// heads, rotate them (or their leading dims), pad q and put k and v of each
// token side by side. Its outputs must be bit for bit those of the separate
// ttnn ops, as SGLang-TorchTPU's decode programs run them.

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <vector>

#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/experimental/transformer/rotary_embedding/rotary_embedding.hpp"
#include "ttnn/operations/normalization/rmsnorm/rmsnorm.hpp"
#include "ttnn/operations/transformer/split_qkv_norm_rope_decode/split_qkv_norm_rope_decode.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

struct Case {
    uint32_t tokens;
    uint32_t num_q_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim;
    uint32_t q_width;
    // Gated attention: each q head followed by its gate head.
    bool gated = false;
    // The rotated dims, all of head_dim if 0.
    uint32_t rotary_dim = 0;
    // FLOAT32 norm weights, as 1 + w of a zero-centered RMSNorm is.
    bool float32_weights = false;
};

class SplitQKVNormRopeDecodeTest : public ::testing::TestWithParam<Case> {
protected:
    void SetUp() override { mesh_device_ = MeshDevice::create(MeshDeviceConfig(MeshShape{1, 1})); }

    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
    }

    ttnn::Tensor to_device(
        const std::vector<float>& values, const ttnn::Shape& shape, DataType dtype = DataType::BFLOAT16) {
        const TensorSpec spec(shape, TensorLayout(dtype, PageConfig(Layout::TILE), {}));
        return ttnn::Tensor::from_vector(values, spec).to_device(mesh_device_.get());
    }

    std::shared_ptr<MeshDevice> mesh_device_;
};

// Values of very different sizes, so that adds and the row sums lose bits.
std::vector<float> random_values(size_t count, float scale, std::mt19937& rng) {
    std::normal_distribution<float> normal(0.0f, scale);
    std::uniform_int_distribution<int> exponent(-6, 6);
    std::vector<float> values(count);
    for (float& value : values) {
        value = std::ldexp(normal(rng), exponent(rng));
    }
    return values;
}

std::vector<uint32_t> bits(const ttnn::Tensor& tensor) {
    std::vector<uint32_t> out;
    for (float value : tensor.cpu().to_vector<float>()) {
        out.push_back(std::bit_cast<uint32_t>(value));
    }
    return out;
}

TEST_P(SplitQKVNormRopeDecodeTest, MatchesSeparateOps) {
    const auto [tokens, num_q_heads, num_kv_heads, head_dim, q_width, gated, rotary, float32_weights] = GetParam();
    const uint32_t rotary_dim = rotary == 0 ? head_dim : rotary;
    const uint32_t q_stride = (gated ? 2 : 1) * head_dim;
    const uint32_t width = num_q_heads * q_stride + 2 * num_kv_heads * head_dim;
    const float epsilon = 1e-6f;
    std::mt19937 rng(tokens * 1000 + num_q_heads * 10 + head_dim + rotary_dim + gated);

    // As SGLang-TorchTPU's decode programs run rms_norm and rotary_embedding.
    const ttnn::DeviceComputeKernelConfig norm_config = ttnn::ComputeKernelConfig{
        .math_fidelity = MathFidelity::HiFi4,
        .math_approx_mode = false,
        .fp32_dest_acc_en = true,
        .packer_l1_acc = true};
    const ttnn::DeviceComputeKernelConfig rope_config = ttnn::ComputeKernelConfig{
        .math_fidelity = MathFidelity::HiFi4, .math_approx_mode = true, .fp32_dest_acc_en = true};
    const MemoryConfig dram{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    // A tile row of each token's k and v on its own core, as the paged cache update reads them.
    const MemoryConfig kv_config{
        TensorMemoryLayout::HEIGHT_SHARDED,
        BufferType::L1,
        ShardSpec(
            num_cores_to_corerangeset(tokens, mesh_device_->compute_with_storage_grid_size(), /*row_wise=*/true),
            {32, 2 * head_dim},
            ShardOrientation::ROW_MAJOR)};
    auto slice = [](const ttnn::Tensor& x, ttsl::SmallVector<uint32_t> begins, ttsl::SmallVector<uint32_t> ends) {
        return ttnn::slice(x, begins, ends, ttsl::SmallVector<uint32_t>(begins.size(), 1));
    };

    // Two runs: the second reuses the cached program with tensors at new addresses.
    for (int run = 0; run < 2; ++run) {
        const std::vector<float> qkv = random_values(static_cast<size_t>(tokens) * width, 2.0f, rng);
        std::vector<float> q_weight = random_values(head_dim, 1.0f, rng);
        std::vector<float> k_weight = random_values(head_dim, 1.0f, rng);
        if (float32_weights) {
            // 1 + w of BFLOAT16 w, which needs more than BFLOAT16's bits.
            for (std::vector<float>* weight : {&q_weight, &k_weight}) {
                for (float& value : *weight) {
                    value = 1.0f + std::bit_cast<float>(std::bit_cast<uint32_t>(value * 0.05f) & 0xFFFF0000u);
                }
            }
        }
        std::vector<float> cos(static_cast<size_t>(tokens) * rotary_dim), sin(cos.size());
        for (uint32_t b = 0; b < tokens; ++b) {
            for (uint32_t i = 0; i < rotary_dim; ++i) {
                const float angle =
                    0.37f * static_cast<float>(i % (rotary_dim / 2) + 1) * static_cast<float>(b + run + 3);
                cos[b * rotary_dim + i] = std::cos(angle);
                sin[b * rotary_dim + i] = std::sin(angle);
            }
        }
        const DataType weight_dtype = float32_weights ? DataType::FLOAT32 : DataType::BFLOAT16;
        const ttnn::Tensor qkv_device = to_device(qkv, ttnn::Shape({tokens, width}));
        const ttnn::Tensor q_weight_device = to_device(q_weight, ttnn::Shape({head_dim}), weight_dtype);
        const ttnn::Tensor k_weight_device = to_device(k_weight, ttnn::Shape({head_dim}), weight_dtype);
        const ttnn::Tensor cos_device = to_device(cos, ttnn::Shape({1, tokens, 1, rotary_dim}));
        const ttnn::Tensor sin_device = to_device(sin, ttnn::Shape({1, tokens, 1, rotary_dim}));

        const std::vector<ttnn::Tensor> fused = ttnn::transformer::split_qkv_norm_rope_decode(
            qkv_device,
            q_weight_device,
            k_weight_device,
            cos_device,
            sin_device,
            num_q_heads,
            num_kv_heads,
            epsilon,
            q_width,
            dram,
            kv_config);
        ASSERT_EQ(fused.size(), 2);

        // The separate ops, on heads [tokens * count, head_dim]. One token rotates in decode mode with
        // token_index 0, several in batched mode.
        const std::optional<uint32_t> token_index = tokens == 1 ? std::optional<uint32_t>(0) : std::nullopt;
        auto norm_rope = [&](ttnn::Tensor x, uint32_t count, const ttnn::Tensor& weight) {
            x = ttnn::rms_norm(
                x, epsilon, weight, std::nullopt, std::nullopt, std::nullopt, std::nullopt, norm_config);
            x = ttnn::reshape(x, ttnn::Shape({1, tokens, count, head_dim}));
            ttnn::Tensor rotated = ttnn::experimental::rotary_embedding(
                slice(x, {0, 0, 0, 0}, {1, tokens, count, rotary_dim}),
                cos_device,
                sin_device,
                token_index,
                std::nullopt,
                rope_config);
            if (rotary_dim < head_dim) {
                rotated = ttnn::concat({rotated, slice(x, {0, 0, 0, rotary_dim}, {1, tokens, count, head_dim})}, 3);
            }
            return ttnn::reshape(rotated, ttnn::Shape({tokens, count, head_dim}));
        };
        const ttnn::Tensor q_heads = slice(
            ttnn::reshape(slice(qkv_device, {0, 0}, {tokens, num_q_heads * q_stride}), ttnn::Shape({tokens, num_q_heads, q_stride})),
            {0, 0, 0},
            {tokens, num_q_heads, head_dim});
        ttnn::Tensor q = norm_rope(ttnn::reshape(q_heads, ttnn::Shape({tokens * num_q_heads, head_dim})), num_q_heads, q_weight_device);
        if (q_width > head_dim) {
            q = ttnn::pad(q, ttsl::SmallVector<std::array<uint32_t, 2>>{{0, 0}, {0, 0}, {0, q_width - head_dim}}, 0.0f);
        }
        q = ttnn::reshape(q, ttnn::Shape({1, tokens, num_q_heads, q_width}));
        const uint32_t k_begin = num_q_heads * q_stride;
        const uint32_t v_begin = k_begin + num_kv_heads * head_dim;
        const ttnn::Tensor k = norm_rope(
            ttnn::reshape(slice(qkv_device, {0, k_begin}, {tokens, v_begin}), ttnn::Shape({tokens * num_kv_heads, head_dim})),
            num_kv_heads,
            k_weight_device);
        const ttnn::Tensor v =
            ttnn::reshape(slice(qkv_device, {0, v_begin}, {tokens, width}), ttnn::Shape({tokens, num_kv_heads, head_dim}));
        const ttnn::Tensor kv =
            ttnn::reshape(ttnn::concat({k, v}, 2), ttnn::Shape({1, tokens, num_kv_heads, 2 * head_dim}));

        EXPECT_EQ(fused[0].logical_shape(), q.logical_shape());
        EXPECT_EQ(bits(fused[0]), bits(q)) << "q, run " << run;
        EXPECT_EQ(fused[1].logical_shape(), kv.logical_shape());
        EXPECT_EQ(fused[1].memory_config(), kv_config);
        EXPECT_EQ(bits(fused[1]), bits(kv)) << "kv, run " << run;
    }
}

INSTANTIATE_TEST_SUITE_P(
    Shapes,
    SplitQKVNormRopeDecodeTest,
    ::testing::Values(
        // Qwen3-8B on one and four chips, one and four tokens, q padded to the k and v width.
        Case{1, 32, 8, 128, 256},
        Case{1, 8, 2, 128, 256},
        Case{4, 32, 8, 128, 256},
        Case{4, 8, 2, 128, 256},
        // Qwen3-14B (q over two tile rows on one chip) and Qwen3-32B on four chips.
        Case{1, 40, 8, 128, 256},
        Case{1, 10, 2, 128, 256},
        Case{1, 16, 2, 128, 256},
        // Tokens in the lower faces, more units than cores, no q padding, wider heads.
        Case{20, 8, 2, 128, 256},
        Case{32, 32, 8, 128, 256},
        Case{3, 8, 2, 128, 128},
        Case{2, 4, 1, 256, 512},
        // Qwen3.8-27B on four chips: gated q, the first 64 of 256 dims rotated, 1 + w weights; one and four
        // tokens. And a partial rotary alone, and FLOAT32 weights alone.
        Case{1, 6, 1, 256, 512, true, 64, true},
        Case{4, 6, 1, 256, 512, true, 64, true},
        Case{4, 8, 2, 128, 256, false, 64, false},
        Case{4, 8, 2, 128, 256, false, 0, true}),
    [](const ::testing::TestParamInfo<Case>& info) {
        const Case& c = info.param;
        return "tokens" + std::to_string(c.tokens) + "_q" + std::to_string(c.num_q_heads) + "_kv" +
               std::to_string(c.num_kv_heads) + "_d" + std::to_string(c.head_dim) + "_w" + std::to_string(c.q_width) +
               (c.gated ? "_gated" : "") + (c.rotary_dim ? "_r" + std::to_string(c.rotary_dim) : "") +
               (c.float32_weights ? "_fp32w" : "");
    });

}  // namespace
}  // namespace tt::tt_metal::distributed::test
