#include <cstdint>
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api.h"

namespace NAMESPACE {
enum class BroadcastMode {
  Copy,
  Scalar,
  Row,
  Col,
  Transpose,
};

void MAIN {
  uint32_t n_tiles = get_arg_val<uint32_t>(0);

  constexpr uint32_t cb_input = tt::CBIndex::c_0;
  constexpr uint32_t cb_output = tt::CBIndex::c_16;
  constexpr BroadcastMode mode = BroadcastMode::BROADCAST_MODE;

  if constexpr (mode == BroadcastMode::Copy) {
    unary_bcast_init<BroadcastType::NONE>(cb_input, cb_output);
  } else if constexpr (mode == BroadcastMode::Row) {
    unary_bcast_init<BroadcastType::ROW>(cb_input, cb_output);
  } else if constexpr (mode == BroadcastMode::Col) {
    unary_bcast_init<BroadcastType::COL>(cb_input, cb_output);
  } else if constexpr (mode == BroadcastMode::Scalar) {
    unary_bcast_init<BroadcastType::SCALAR>(cb_input, cb_output);
  } else if constexpr (mode == BroadcastMode::Transpose) {
    transpose_wh_init(cb_input, cb_output);
  }

  for (uint32_t i = 0; i < n_tiles; ++i) {
    cb_wait_front(cb_input, 1);
    cb_reserve_back(cb_output, 1);

    tile_regs_acquire();
    if constexpr (mode == BroadcastMode::Copy) {
      unary_bcast<BroadcastType::NONE>(cb_input, 0, 0);
    } else if constexpr (mode == BroadcastMode::Row) {
      unary_bcast<BroadcastType::ROW>(cb_input, 0, 0);
    } else if constexpr (mode == BroadcastMode::Col) {
      unary_bcast<BroadcastType::COL>(cb_input, 0, 0);
    } else if constexpr (mode == BroadcastMode::Scalar) {
      unary_bcast<BroadcastType::SCALAR>(cb_input, 0, 0);
    } else if constexpr (mode == BroadcastMode::Transpose) {
      transpose_wh_tile(cb_input, 0, 0);
    }
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(0, cb_output);
    tile_regs_release();

    cb_pop_front(cb_input, 1);
    cb_push_back(cb_output, 1);
  }
}
}  // namespace NAMESPACE
