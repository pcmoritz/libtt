#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api.h"

namespace NAMESPACE {

void multiply_tile(uint32_t lhs_cb, uint32_t rhs_cb, uint32_t output_cb) {
  cb_wait_front(lhs_cb, 1);
  cb_wait_front(rhs_cb, 1);
  mul_binary_tile_init();
  cb_reserve_back(output_cb, 1);
  tile_regs_acquire();
  copy_tile_to_dst_init_short(lhs_cb);
  copy_tile(lhs_cb, 0, 0);
  copy_tile_to_dst_init_short(rhs_cb);
  copy_tile(rhs_cb, 0, 1);
  mul_binary_tile(0, 1, 0);
  tile_regs_commit();
  tile_regs_wait();
  pack_tile(0, output_cb);
  tile_regs_release();
  cb_pop_front(lhs_cb, 1);
  cb_pop_front(rhs_cb, 1);
  cb_push_back(output_cb, 1);
}

void MAIN {
  uint32_t n_tiles = get_arg_val<uint32_t>(0);

  constexpr uint32_t cb_x_a = tt::CBIndex::c_0;
  constexpr uint32_t cb_cos = tt::CBIndex::c_1;
  constexpr uint32_t cb_x_b = tt::CBIndex::c_2;
  constexpr uint32_t cb_sin = tt::CBIndex::c_3;
  constexpr uint32_t cb_a_cos = tt::CBIndex::c_4;
  constexpr uint32_t cb_b_sin = tt::CBIndex::c_5;
  constexpr uint32_t cb_out = tt::CBIndex::c_16;

  unary_op_init_common(cb_x_a, cb_out);

  for (uint32_t i = 0; i < n_tiles; ++i) {
    multiply_tile(cb_x_a, cb_cos, cb_a_cos);
    multiply_tile(cb_x_b, cb_sin, cb_b_sin);

    cb_wait_front(cb_a_cos, 1);
    cb_wait_front(cb_b_sin, 1);
    add_binary_tile_init();
    cb_reserve_back(cb_out, 1);
    tile_regs_acquire();
    copy_tile_to_dst_init_short(cb_a_cos);
    copy_tile(cb_a_cos, 0, 0);
    copy_tile_to_dst_init_short(cb_b_sin);
    copy_tile(cb_b_sin, 0, 1);
    add_binary_tile(0, 1, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_pop_front(cb_a_cos, 1);
    cb_pop_front(cb_b_sin, 1);
    cb_push_back(cb_out, 1);
  }
}

}  // namespace NAMESPACE
