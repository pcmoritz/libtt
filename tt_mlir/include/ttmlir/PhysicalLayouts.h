// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

// The shapes large tensors are stored in on device, for both
// ttir-physical-layouts (public functions' arguments and results) and the
// runtime (the buffers it stores, tt_mlir_physical_layouts.patch).
//
// In tiles, a rank-1 array of N elements is one row of N / 32 tiles, each
// padded to 32 rows: 32 times its size. A matrix of W < 32 rows takes 32 / W
// times its size, and a [T, 1, V] tensor pads each of its T rows to 32. Large
// ones hold the same elements in row-major order in whole tiles instead.

#ifndef TTMLIR_PHYSICALLAYOUTS_H
#define TTMLIR_PHYSICALLAYOUTS_H

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <optional>
#include <vector>

namespace ttmlir::physical_layouts {

constexpr int64_t kFoldWidth = 32;
// Rank-1 arrays and [T, 1, V] tensors of at least 2^20 elements, and matrices
// of at most 8 rows and at least 2^22 elements.
constexpr int64_t kMinElements = int64_t{1} << 20;
constexpr int64_t kMaxMatrixRows = 8;
constexpr int64_t kMinMatrixElements = int64_t{1} << 22;

// LIBTT_PHYSICAL_LAYOUTS=0 turns physical layouts off: the compiler's compact
// shapes, and the runtime's for new buffers. Both read it each time.
inline bool enabled() {
  const char *value = std::getenv("LIBTT_PHYSICAL_LAYOUTS");
  return value == nullptr || std::strcmp(value, "0") != 0;
}

// The shape a tensor of shape `dims` is stored in, if not its own: rank-1
// arrays and few-row matrices of a multiple of 32 elements folded to
// [N / 32, 32], and [T, 1, V] tensors as [T, V], folded too when that is a
// few-row matrix. A compact shape is stored in its own shape.
template <typename Dims>
std::optional<std::array<int64_t, 2>> compactShape(const Dims &dims) {
  const std::vector<int64_t> shape(std::begin(dims), std::end(dims));
  int64_t n = 1;
  for (int64_t d : shape) {
    n *= d;
  }
  const bool rank1 = shape.size() == 1 && n >= kMinElements;
  const bool fewRows = shape.size() == 2 && shape[0] <= kMaxMatrixRows &&
                       n >= kMinMatrixElements;
  if ((rank1 || fewRows) && n % kFoldWidth == 0) {
    return std::array<int64_t, 2>{n / kFoldWidth, kFoldWidth};
  }
  if (shape.size() == 3 && shape[1] == 1 && shape[0] > 1 && shape[2] > 1 &&
      n >= kMinElements) {
    const std::array<int64_t, 2> rows{shape[0], shape[2]};
    return compactShape(rows).value_or(rows);
  }
  return std::nullopt;
}

} // namespace ttmlir::physical_layouts

#endif // TTMLIR_PHYSICALLAYOUTS_H
