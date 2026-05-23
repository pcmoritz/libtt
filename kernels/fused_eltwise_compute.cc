#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
FUSED_HEADERS
#include "compute_kernel_api.h"

namespace NAMESPACE {

FUSED_HELPERS

void MAIN {
  uint32_t n_tiles = get_arg_val<uint32_t>(0);
  constexpr uint32_t cb_out = tt::CBIndex::c_16;

  unary_op_init_common(tt::CBIndex::c_0, cb_out);
FUSED_TYPECAST_INITS

  for (uint32_t i = 0; i < n_tiles; ++i) {
FUSED_STEPS
  }
}

}  // namespace NAMESPACE
