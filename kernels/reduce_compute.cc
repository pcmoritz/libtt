#include <cstdint>
#define REDUCE_OP REDUCE_POOL_TYPE
#define REDUCE_DIM ckernel::REDUCE_ROW
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api.h"

namespace NAMESPACE {
void MAIN {
  uint32_t reduce_groups = get_arg_val<uint32_t>(0);
  uint32_t width_tiles = get_arg_val<uint32_t>(1);

  constexpr uint32_t cb_input = tt::CBIndex::c_0;
  constexpr uint32_t cb_scaler = tt::CBIndex::c_2;
  constexpr uint32_t cb_output = tt::CBIndex::c_16;
  constexpr uint32_t onetile = 1;

  ckernel::reduce_init<REDUCE_POOL_TYPE, ckernel::REDUCE_ROW>(cb_input, cb_scaler, cb_output);
  cb_wait_front(cb_scaler, onetile);

  for (uint32_t group = 0; group < reduce_groups; ++group) {
    int reduce_dst_idx = 0;
    tile_regs_acquire();
    for (uint32_t wt = 0; wt < width_tiles; ++wt) {
      cb_wait_front(cb_input, onetile);
      ckernel::reduce_tile<REDUCE_POOL_TYPE, ckernel::REDUCE_ROW>(
          cb_input, cb_scaler, 0, 0, reduce_dst_idx);
      cb_pop_front(cb_input, onetile);
    }
    cb_reserve_back(cb_output, onetile);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(reduce_dst_idx, cb_output);
    tile_regs_release();
    cb_push_back(cb_output, onetile);
  }

  cb_pop_front(cb_scaler, onetile);
  ckernel::reduce_uninit();
}
}  // namespace NAMESPACE
