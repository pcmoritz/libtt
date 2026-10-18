// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::transformer {

// Copies slot sources[i] of a recurrent state pool to slot targets[i], in
// place, for each of the N <= 32 pairs. A pair with slot 0 on either side is
// skipped. The sources and targets must be disjoint, as when a speculative
// verify keeps each request's state after its accepted draft tokens.
void state_pool_copy(const Tensor& state_pool, const Tensor& sources, const Tensor& targets);

}  // namespace ttnn::transformer
