// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
// ttnn::experimental::masked_reduce counts, or sums, the elements of each row
// that compare true against the row's threshold, in one pass over the input.
#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/mesh_device.hpp>
#include "ttnn/operations/experimental/masked_reduce/masked_reduce.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

using ttnn::experimental::MaskedCompare;
using ttnn::experimental::MaskedMode;

class MaskedReduceTest : public ::testing::Test {
protected:
    void SetUp() override { mesh_device_ = MeshDevice::create(MeshDeviceConfig(MeshShape{1, 1})); }
    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
    }

    template <typename T>
    ttnn::Tensor to_device(const std::vector<T>& values, const std::vector<uint32_t>& shape, DataType dtype) {
        const TensorSpec spec(ttnn::Shape(shape), TensorLayout(dtype, PageConfig(Layout::TILE), MemoryConfig{}));
        return ttnn::Tensor::from_vector(values, spec).to_device(mesh_device_.get());
    }

    static bool holds(MaskedCompare compare, float x, float t) {
        switch (compare) {
            case MaskedCompare::GT: return x > t;
            case MaskedCompare::GE: return x >= t;
            case MaskedCompare::LT: return x < t;
            case MaskedCompare::LE: return x <= t;
        }
        return false;
    }

    struct Options {
        bool timed = false;
        // Every seventh value is -inf: a masked one must add nothing to a
        // sum, an unmasked one makes the sum -inf.
        bool infinities = false;
        // Leading dimension of a [batch, rows, width] input.
        uint32_t batch = 1;
        // A [..., rows] threshold and result rather than [..., rows, 1].
        bool row_form = false;
        // Counts as INT32.
        bool int_counts = false;
    };

    // rows x width values, one threshold per row; checks every compare and mode.
    template <typename T>
    void check(uint32_t rows, uint32_t width, DataType dtype, Options options = {}) {
        const bool timed = options.timed;
        const bool infinities = options.infinities;
        std::mt19937 rng(7);
        std::normal_distribution<float> dist(0.0f, 4.0f);
        rows *= options.batch;
        std::vector<T> values(static_cast<size_t>(rows) * width);
        std::vector<float> ref(values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = T(infinities && i % 7 == 3 ? -std::numeric_limits<float>::infinity() : dist(rng));
            ref[i] = static_cast<float>(values[i]);
        }
        std::vector<T> thresholds(rows);
        std::vector<float> tref(rows);
        for (uint32_t r = 0; r < rows; ++r) {
            // A value of the row, so that ties exercise GT against GE.
            thresholds[r] = values[static_cast<size_t>(r) * width + (r * 97) % width];
            if (std::isinf(static_cast<float>(thresholds[r]))) {
                thresholds[r] = values[static_cast<size_t>(r) * width + (r * 97 + 1) % width];
            }
            tref[r] = static_cast<float>(thresholds[r]);
        }
        std::vector<uint32_t> in_shape = {rows, width};
        std::vector<uint32_t> t_shape = options.row_form ? std::vector<uint32_t>{rows} : std::vector<uint32_t>{rows, 1};
        if (options.batch > 1) {
            in_shape = {options.batch, rows / options.batch, width};
            t_shape = options.row_form ? std::vector<uint32_t>{options.batch, rows / options.batch}
                                       : std::vector<uint32_t>{options.batch, rows / options.batch, 1};
        }
        auto input = to_device(values, in_shape, dtype);
        auto threshold = to_device(thresholds, t_shape, dtype);
        for (MaskedCompare compare : {MaskedCompare::GT, MaskedCompare::GE, MaskedCompare::LT, MaskedCompare::LE}) {
            for (MaskedMode mode : {MaskedMode::Count, MaskedMode::Sum}) {
                const bool ints = options.int_counts && mode == MaskedMode::Count;
                auto out = ttnn::experimental::masked_reduce(
                    input, threshold, compare, mode, ints ? DataType::INT32 : DataType::FLOAT32);
                EXPECT_EQ(out.logical_shape(), ttnn::Shape(t_shape));
                std::vector<float> got;
                if (ints) {
                    for (int32_t v : out.cpu().template to_vector<int32_t>()) {
                        got.push_back(static_cast<float>(v));
                    }
                } else {
                    got = out.cpu().template to_vector<float>();
                }
                ASSERT_EQ(got.size(), rows);
                for (uint32_t r = 0; r < rows; ++r) {
                    double want = 0;
                    for (uint32_t w = 0; w < width; ++w) {
                        const float x = ref[static_cast<size_t>(r) * width + w];
                        if (holds(compare, x, tref[r])) {
                            want += mode == MaskedMode::Count ? 1.0 : x;
                        }
                    }
                    const double tolerance = mode == MaskedMode::Count ? 0.0 : 1e-4 * std::abs(want) + 1e-2;
                    if (std::isinf(want)) {
                        EXPECT_EQ(got[r], want) << "row " << r << " compare " << static_cast<int>(compare)
                                                << " mode " << static_cast<int>(mode);
                    } else {
                        EXPECT_NEAR(got[r], want, tolerance)
                            << "row " << r << " compare " << static_cast<int>(compare) << " mode "
                            << static_cast<int>(mode);
                    }
                }
            }
        }
        if (timed) {
            constexpr int reps = 20;
            auto out = ttnn::experimental::masked_reduce(input, threshold, MaskedCompare::GT, MaskedMode::Count);
            out.cpu();
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < reps; ++i) {
                out = ttnn::experimental::masked_reduce(input, threshold, MaskedCompare::GT, MaskedMode::Count);
            }
            out.cpu();
            const double ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / reps;
            printf("masked_reduce count [%u x %u] %s: %.3f ms per call (host-timed, 20 calls)\n",
                   rows, width, dtype == DataType::FLOAT32 ? "f32" : "bf16", ms);
        }
    }

    std::shared_ptr<MeshDevice> mesh_device_;
};

TEST_F(MaskedReduceTest, F32VocabularyRows) { check<float>(2, 151936, DataType::FLOAT32, {.timed = true}); }

TEST_F(MaskedReduceTest, Bf16Rows) { check<bfloat16>(4, 4096, DataType::BFLOAT16); }

TEST_F(MaskedReduceTest, PartialTileWidthAndManyRows) {
    // 40 rows span two tile rows; 1000 is not a multiple of a tile.
    check<float>(40, 1000, DataType::FLOAT32);
}

TEST_F(MaskedReduceTest, DenseRowPacking) {
    // 3 rows pack 10 input tiles per dense tile and straddle the face boundary;
    // 16 rows pack two; 17 rows read whole tiles; one row packs 32; a width of
    // one tile leaves most of the dense tile empty.
    check<float>(3, 1000, DataType::FLOAT32);
    check<float>(16, 4096, DataType::FLOAT32);
    check<float>(17, 2048, DataType::FLOAT32);
    check<bfloat16>(1, 100000, DataType::BFLOAT16);
    check<float>(2, 16, DataType::FLOAT32);
}

TEST_F(MaskedReduceTest, MaskedInfinitiesAddNothing) {
    check<float>(4, 3000, DataType::FLOAT32, {.infinities = true});
    check<bfloat16>(4, 3000, DataType::BFLOAT16, {.infinities = true});
}

TEST_F(MaskedReduceTest, RowThresholdsAndIntCounts) {
    // The sampler's form: [2] thresholds, [2] results, counts as int32; then
    // batches whose thresholds share a tile (rows 0 and 1 of it, the odd row
    // off the 64-byte alignment for bf16), and a many-row batch.
    check<float>(2, 151936, DataType::FLOAT32, {.row_form = true, .int_counts = true});
    check<float>(2, 1000, DataType::FLOAT32, {.batch = 3, .row_form = true, .int_counts = true});
    check<bfloat16>(2, 1000, DataType::BFLOAT16, {.batch = 3, .row_form = true});
    check<float>(40, 1000, DataType::FLOAT32, {.batch = 2, .row_form = true});
    check<float>(40, 1000, DataType::FLOAT32, {.batch = 2, .int_counts = true});
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
