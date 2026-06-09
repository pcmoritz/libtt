#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/tile_move_copy.h"

namespace NAMESPACE {

constexpr uint32_t cb_input = tt::CBIndex::c_0;
constexpr uint32_t cb_pair = tt::CBIndex::c_1;
constexpr uint32_t cb_cos = tt::CBIndex::c_2;
constexpr uint32_t cb_sin = tt::CBIndex::c_3;
constexpr uint32_t cb_output = tt::CBIndex::c_16;
constexpr uint32_t cb_x_cos = tt::CBIndex::c_24;
constexpr uint32_t cb_pair_sin = tt::CBIndex::c_25;
constexpr uint32_t TILE_R = 32;
constexpr uint32_t tiles_per_row = ROPE_TILES_PER_ROW;
constexpr uint32_t output_tile_rows = ROPE_OUTPUT_TILE_ROWS;
constexpr uint32_t half_tiles = ROPE_HALF_TILES;

void multiply_tile(uint32_t lhs_cb, uint32_t rhs_cb, uint32_t output_cb,
                   uint32_t bcast_row) {
  mul_bcast_rows_init_short(lhs_cb, rhs_cb);
  cb_wait_front(lhs_cb, 1);
  cb_wait_front(rhs_cb, 1);
  cb_reserve_back(output_cb, 1);

  invalidate_l1_cache();
  tile_regs_acquire();
  mul_tiles_bcast_rows(lhs_cb, rhs_cb, 0, 0, 0, bcast_row);
  tile_regs_commit();

  tile_regs_wait();
  pack_tile(0, output_cb);
  tile_regs_release();

  cb_pop_front(lhs_cb, 1);
  cb_pop_front(rhs_cb, 1);
  cb_push_back(output_cb, 1);
}

template <bool Subtract>
void combine_tile() {
  if constexpr (Subtract) {
    sub_tiles_init(cb_x_cos, cb_pair_sin);
  } else {
    add_tiles_init(cb_x_cos, cb_pair_sin);
  }
  cb_wait_front(cb_x_cos, 1);
  cb_wait_front(cb_pair_sin, 1);
  cb_reserve_back(cb_output, 1);

  invalidate_l1_cache();
  tile_regs_acquire();
  if constexpr (Subtract) {
    sub_tiles(cb_x_cos, cb_pair_sin, 0, 0, 0);
  } else {
    add_tiles(cb_x_cos, cb_pair_sin, 0, 0, 0);
  }
  tile_regs_commit();

  tile_regs_wait();
  pack_tile(0, cb_output);
  tile_regs_release();

  cb_pop_front(cb_x_cos, 1);
  cb_pop_front(cb_pair_sin, 1);
  cb_push_back(cb_output, 1);
}

void MAIN {
  uint32_t tile_offset = get_arg_val<uint32_t>(0);
  uint32_t n_tiles = get_arg_val<uint32_t>(1);

  unary_op_init_common(cb_input, cb_output);
  for (uint32_t i = 0; i < n_tiles; ++i) {
    uint32_t output_tile = tile_offset + i;
    uint32_t col_tile = output_tile % tiles_per_row;
    uint32_t row_major = output_tile / tiles_per_row;
    uint32_t batch = row_major / output_tile_rows;
    uint32_t bcast_row = batch % TILE_R;
    multiply_tile(cb_input, cb_cos, cb_x_cos, bcast_row);
    multiply_tile(cb_pair, cb_sin, cb_pair_sin, bcast_row);
    if (col_tile < half_tiles) {
      combine_tile<true>();
    } else {
      combine_tile<false>();
    }
  }
}

}  // namespace NAMESPACE
