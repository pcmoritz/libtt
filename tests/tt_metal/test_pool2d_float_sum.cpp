// SPDX-License-Identifier: Apache-2.0
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/pool/generic/generic_pools.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <tt-metalium/mesh_device.hpp>
#include <vector>

namespace tt::tt_metal::distributed::test {
TEST(NativeFloatSumPool, PreservesInputPrecision) {
    auto device = MeshDevice::create(MeshDeviceConfig(MeshShape{1, 1}), 1 << 16);
    device->enable_program_cache();
    constexpr uint32_t H = 17, W = 19, K = 3, S = 2, P = 1, OH = 9, OW = 10;
    for (auto dtype : {DataType::FLOAT32, DataType::FLOAT16}) {
        for (uint32_t C : {1u, 16u, 32u, 48u, 144u}) {
            for (auto scheme : {TensorMemoryLayout::HEIGHT_SHARDED, TensorMemoryLayout::WIDTH_SHARDED,
                                TensorMemoryLayout::BLOCK_SHARDED}) {
                for (int trial = 0; trial < 2; ++trial) {
                    SCOPED_TRACE(::testing::Message() << "dtype=" << int(dtype) << " channels=" << C
                                                      << " scheme=" << int(scheme) << " trial=" << trial);
                    std::vector<float> values(H * W * C);
                    for (size_t i = 0; i < values.size(); ++i) {
                        values[i] = std::sin(float(i)) + (trial + 1) * 0.00017f;
                        if (dtype == DataType::FLOAT16)
                            values[i] = float(_Float16(values[i]));
                    }
                    const TensorSpec spec(ttnn::Shape({1, 1, H * W, C}),
                                          TensorLayout(dtype, PageConfig(Layout::ROW_MAJOR), {}));
                    auto input = ttnn::Tensor::from_vector(values, spec).to_device(device.get());
                    auto out =
                        ttnn::avg_pool2d(input, 1, H, W, C, {K, K}, {S, S}, std::array<uint32_t, 2>{P, P}, false, true,
                                         1, std::nullopt, std::nullopt, scheme, std::nullopt, false, true, dtype,
                                         Layout::ROW_MAJOR, scheme != TensorMemoryLayout::HEIGHT_SHARDED);
                    auto actual = ttnn::typecast(out, DataType::FLOAT32).cpu().to_vector<float>();
                    ASSERT_EQ(actual.size(), OH * OW * C);
                    for (uint32_t y = 0; y < OH; ++y)
                        for (uint32_t x = 0; x < OW; ++x) {
                            for (uint32_t c = 0; c < C; ++c) {
                                float expected = 0;
                                for (int dy = 0; dy < int(K); ++dy)
                                    for (int dx = 0; dx < int(K); ++dx) {
                                        const int iy = int(y * S) + dy - P, ix = int(x * S) + dx - P;
                                        if (iy >= 0 && iy < int(H) && ix >= 0 && ix < int(W))
                                            expected += values[(iy * W + ix) * C + c];
                                    }
                                if (dtype == DataType::FLOAT16)
                                    expected = float(_Float16(expected));
                                ASSERT_NEAR(actual[(y * OW + x) * C + c], expected, 1e-5f)
                                    << "y=" << y << " x=" << x << " c=" << c;
                            }
                        }
                }
            }
        }
    }
    device->close();
}
} // namespace tt::tt_metal::distributed::test
