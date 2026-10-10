// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct MatmulFp32Params {
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct MatmulFp32Inputs {
    Tensor a;
    Tensor b;
};

}  // namespace ttnn::experimental::prim
