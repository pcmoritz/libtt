// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Each core computes a contiguous range of output tiles (row-major over batch,
// rows and columns) in runs of up to `run_width` consecutive tiles of one
// output row. The tiles of a run share their A row, which is read once per run
// and kept in in0 for the whole run; B streams one tile per k.
inline uint32_t matmul_fp32_run_cols(uint32_t tile, uint32_t end, uint32_t Nt, uint32_t run_width) {
    uint32_t cols = Nt - tile % Nt;
    cols = cols < run_width ? cols : run_width;
    return cols < end - tile ? cols : end - tile;
}
