#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "UNARY_OP_HEADER"
#include "compute_kernel_api.h"

namespace NAMESPACE {
void MAIN {
  uint32_t n_tiles = get_arg_val<uint32_t>(0);
  constexpr uint32_t cb_input = tt::CBIndex::c_0;
  constexpr uint32_t cb_out = tt::CBIndex::c_16;

  init_sfpu(cb_input, cb_out);
  UNARY_OP_INIT;

  for (uint32_t i = 0; i < n_tiles; ++i) {
    tile_regs_acquire();
    cb_wait_front(cb_input, 1);
    copy_tile(cb_input, 0, 0);
    UNARY_OP_TILE;
    tile_regs_commit();

    tile_regs_wait();
    cb_reserve_back(cb_out, 1);
    pack_tile(0, cb_out);
    cb_pop_front(cb_input, 1);
    tile_regs_release();
    cb_push_back(cb_out, 1);
  }
}
}  // namespace NAMESPACE
