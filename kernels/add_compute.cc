#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api.h"

namespace NAMESPACE {
void MAIN {
  uint32_t n_tiles = get_arg_val<uint32_t>(0);
  constexpr uint32_t cb_lhs = tt::CBIndex::c_0;
  constexpr uint32_t cb_rhs = tt::CBIndex::c_1;
  constexpr uint32_t cb_out = tt::CBIndex::c_16;
  constexpr uint32_t dst_reg_idx = 0;

  binary_op_init_common(cb_lhs, cb_rhs, cb_out);
  add_tiles_init(cb_lhs, cb_rhs);

  for (uint32_t i = 0; i < n_tiles; ++i) {
    cb_wait_front(cb_lhs, 1);
    cb_wait_front(cb_rhs, 1);
    tile_regs_acquire();
    add_tiles(cb_lhs, cb_rhs, 0, 0, dst_reg_idx);
    cb_pop_front(cb_lhs, 1);
    cb_pop_front(cb_rhs, 1);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_out, 1);
    pack_tile(dst_reg_idx, cb_out);
    cb_push_back(cb_out, 1);
    tile_regs_release();
  }
}
}  // namespace NAMESPACE
