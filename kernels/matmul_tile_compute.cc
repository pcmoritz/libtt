#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api.h"
#define A(n) get_arg_val<uint32_t>(n)
namespace NAMESPACE {
void MAIN {
  constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
  constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
  constexpr uint32_t cb_out = tt::CBIndex::c_16;
  constexpr uint32_t one_tile = 1;
  const uint32_t kt = A(0);
  const uint32_t output_tiles = A(1);
  mm_init(cb_in0, cb_in1, cb_out);
  for (uint32_t tile = 0; tile < output_tiles; ++tile) {
    tile_regs_acquire();
    for (uint32_t k = 0; k < kt; ++k) {
      cb_wait_front(cb_in0, one_tile);
      cb_wait_front(cb_in1, one_tile);
      matmul_tiles(cb_in0, cb_in1, 0, 0, 0);
      cb_pop_front(cb_in0, one_tile);
      cb_pop_front(cb_in1, one_tile);
    }
    cb_reserve_back(cb_out, one_tile);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, one_tile);
  }
}
}  // namespace NAMESPACE
