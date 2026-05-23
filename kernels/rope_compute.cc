#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/negative.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api.h"

namespace NAMESPACE {
void MAIN {
  uint32_t tile_offset = get_arg_val<uint32_t>(0);
  uint32_t n_tiles = get_arg_val<uint32_t>(1);
  constexpr uint32_t cb_input = tt::CBIndex::c_0;
  constexpr uint32_t cb_rotated = tt::CBIndex::c_1;
  constexpr uint32_t cb_cos = tt::CBIndex::c_2;
  constexpr uint32_t cb_sin = tt::CBIndex::c_3;
  constexpr uint32_t cb_out = tt::CBIndex::c_16;
  constexpr uint32_t width_tiles = ROPE_WIDTH_TILES;
  constexpr uint32_t half_width_tiles = width_tiles / 2;

  unary_op_init_common(cb_input, cb_out);
  mul_binary_tile_init();
  add_binary_tile_init();
  negative_tile_init();

  for (uint32_t i = 0; i < n_tiles; ++i) {
    uint32_t tile_col = (tile_offset + i) % width_tiles;
    cb_wait_front(cb_input, 1);
    cb_wait_front(cb_rotated, 1);
    cb_wait_front(cb_cos, 1);
    cb_wait_front(cb_sin, 1);
    cb_reserve_back(cb_out, 1);

    tile_regs_acquire();
    copy_tile_to_dst_init_short_with_dt(cb_cos, cb_input);
    copy_tile(cb_input, 0, 0);
    copy_tile_to_dst_init_short_with_dt(cb_input, cb_cos);
    copy_tile(cb_cos, 0, 1);
    mul_binary_tile(0, 1, 0);

    copy_tile_to_dst_init_short_with_dt(cb_sin, cb_rotated);
    copy_tile(cb_rotated, 0, 1);
    if (tile_col < half_width_tiles) {
      negative_tile(1);
    }
    copy_tile_to_dst_init_short_with_dt(cb_rotated, cb_sin);
    copy_tile(cb_sin, 0, 2);
    mul_binary_tile(1, 2, 1);
    add_binary_tile(0, 1, 0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    cb_pop_front(cb_input, 1);
    cb_pop_front(cb_rotated, 1);
    cb_pop_front(cb_cos, 1);
    cb_pop_front(cb_sin, 1);
    cb_push_back(cb_out, 1);
  }
}
}  // namespace NAMESPACE
