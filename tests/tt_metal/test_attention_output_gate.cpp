// SPDX-License-Identifier: Apache-2.0
// attention_output_gate runs gated attention's output gate (Qwen3.5 and
// Qwen3.8) as PyTorch models trace it in one program: each head of the paged
// SDPA decode output [1, T, H, W] (the value half of the K and V side by
// side) times the sigmoid of its gate head in the QKV projection [T, N]. The
// result must equal the separate ttnn ops, as SGLang-TorchTPU's decode
// programs run them (the slices, a typecast to FP32, sigmoid, a typecast back,
// the multiply), bit for bit.
#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include <tt-metalium/mesh_device.hpp>

#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/transformer/attention_output_gate/attention_output_gate.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

struct Case {
    uint32_t tokens;
    uint32_t heads;
    uint32_t head_dim;
    // The attention output's width and where each head's values start in it.
    uint32_t width;
    uint32_t value_offset;
    // The projection's gate heads: the first at gate_offset, each gate_stride
    // after the last, in a row of gate_width.
    uint32_t gate_offset;
    uint32_t gate_stride;
    uint32_t gate_width;
};

class AttentionOutputGateTest : public ::testing::TestWithParam<Case> {
protected:
    void SetUp() override { mesh_device_ = MeshDevice::create(MeshDeviceConfig(MeshShape{1, 1})); }

    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
    }

    // A BF16 TILE tensor in DRAM of normal values times `scale`.
    ttnn::Tensor random(const ttnn::Shape& shape, uint32_t seed, float scale) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> normal(0.0f, scale);
        std::vector<float> values(shape.volume());
        for (float& v : values) {
            v = normal(rng);
        }
        const TensorSpec spec(shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), {}));
        return ttnn::Tensor::from_vector(values, spec).to_device(mesh_device_.get());
    }

    std::shared_ptr<MeshDevice> mesh_device_;
};

std::vector<uint32_t> bits(const ttnn::Tensor& tensor) {
    std::vector<uint32_t> out;
    for (float value : tensor.cpu().to_vector<float>()) {
        out.push_back(std::bit_cast<uint32_t>(value));
    }
    return out;
}

TEST_P(AttentionOutputGateTest, MatchesSeparateOps) {
    const auto [T, H, D, W, value_offset, gate_offset, gate_stride, gate_width] = GetParam();
    using Index = ttsl::SmallVector<uint32_t>;
    const MemoryConfig dram{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    // Two runs: the second reuses the cached program with tensors at new addresses.
    for (uint32_t run = 0; run < 2; ++run) {
        const uint32_t seed = T * 1000 + H * 10 + D + run;
        const ttnn::Tensor attention = random(ttnn::Shape({1, T, H, W}), seed, 1.0f);
        // Gate logits of all sizes, so the sigmoid saturates and rounds both ways.
        const ttnn::Tensor qkv = random(ttnn::Shape({T, gate_width}), seed + 1, 4.0f);

        const ttnn::Tensor fused = ttnn::transformer::attention_output_gate(
            attention, qkv, D, value_offset, gate_offset, gate_stride, dram);

        const ttnn::Tensor values = ttnn::reshape(
            ttnn::slice(attention, Index{0, 0, 0, value_offset}, Index{1, T, H, value_offset + D}, Index{1, 1, 1, 1}),
            ttnn::Shape({T, H * D}));
        const ttnn::Tensor gate_heads = ttnn::slice(
            ttnn::reshape(qkv, ttnn::Shape({T, gate_width / gate_stride, gate_stride})),
            Index{0, 0, gate_offset},
            Index{T, H, gate_offset + D},
            Index{1, 1, 1});
        // As the runtime runs ttnn.sigmoid.
        const ttnn::Tensor sigmoid = ttnn::sigmoid(
            ttnn::typecast(gate_heads, DataType::FLOAT32),
            static_cast<int>(ttnn::operations::unary::VecMode::RC),
            ttnn::operations::unary::SigmoidMode::ACCURATE,
            std::nullopt,
            std::nullopt,
            std::nullopt);
        const ttnn::Tensor gate =
            ttnn::reshape(ttnn::typecast(sigmoid, DataType::BFLOAT16), ttnn::Shape({T, H * D}));
        const ttnn::Tensor unfused = ttnn::multiply(values, gate, DataType::BFLOAT16);

        EXPECT_EQ(fused.logical_shape(), ttnn::Shape({T, H * D}));
        EXPECT_EQ(fused.dtype(), DataType::BFLOAT16);
        EXPECT_EQ(bits(fused), bits(unfused)) << "run " << run;
    }
}

INSTANTIATE_TEST_SUITE_P(
    Shapes,
    AttentionOutputGateTest,
    ::testing::Values(
        // Qwen3.8-27B on four chips (6 q heads of 256, each followed by its
        // gate head, then the k and v heads): one token and SGLang's batch of
        // four.
        Case{1, 6, 256, 512, 256, 256, 512, 3584},
        Case{4, 6, 256, 512, 256, 256, 512, 3584},
        // Tokens past the top faces of a tile.
        Case{17, 6, 256, 512, 256, 256, 512, 3584},
        // More heads than a tile's rows, narrower heads, values at the start.
        Case{2, 40, 64, 128, 0, 64, 128, 5376}));

}  // namespace
}  // namespace tt::tt_metal::distributed::test
