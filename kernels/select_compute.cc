#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/add_int_sfpu.h"
#include "compute_kernel_api.h"

namespace NAMESPACE {
template <DataFormat Format>
ALWI void add_zero_init() {
  if constexpr (Format == DataFormat::Float16 || Format == DataFormat::Float16_b ||
                Format == DataFormat::Float32) {
    add_binary_tile_init();
  } else {
    add_int_tile_init();
  }
}

template <DataFormat Format>
ALWI void add_zero_tile(uint32_t selected, uint32_t zero, uint32_t out) {
  if constexpr (Format == DataFormat::Float16 || Format == DataFormat::Float16_b ||
                Format == DataFormat::Float32) {
    add_binary_tile(selected, zero, out);
  } else if constexpr (Format == DataFormat::Int32) {
    add_int32_tile(selected, zero, out);
  } else if constexpr (Format == DataFormat::UInt32) {
    add_uint32_tile(selected, zero, out);
  } else if constexpr (Format == DataFormat::UInt16) {
    add_uint16_tile(selected, zero, out);
  }
}

constexpr DataFormat selected_format(uint32_t cb_selected, uint32_t cb_out) {
#ifdef UCK_CHLKC_PACK
  return static_cast<DataFormat>((uint)pack_src_format[cb_out]);
#else
  return static_cast<DataFormat>((uint)unpack_src_format[cb_selected]);
#endif
}

void MAIN {
  uint32_t n_tiles = get_arg_val<uint32_t>(0);
  constexpr uint32_t cb_selected = tt::CBIndex::c_3;
  constexpr uint32_t cb_zero = tt::CBIndex::c_4;
  constexpr uint32_t cb_out = tt::CBIndex::c_16;
  constexpr DataFormat format = selected_format(cb_selected, cb_out);

  unary_op_init_common(cb_selected, cb_out);
  add_zero_init<format>();

  for (uint32_t i = 0; i < n_tiles; ++i) {
    cb_wait_front(cb_selected, 1);
    cb_wait_front(cb_zero, 1);
    cb_reserve_back(cb_out, 1);

    tile_regs_acquire();
    copy_tile_to_dst_init_short_with_dt(cb_zero, cb_selected);
    copy_tile(cb_selected, 0, 0);
    copy_tile_to_dst_init_short_with_dt(cb_selected, cb_zero);
    copy_tile(cb_zero, 0, 1);
    add_zero_tile<format>(0, 1, 0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    cb_pop_front(cb_selected, 1);
    cb_pop_front(cb_zero, 1);
    cb_push_back(cb_out, 1);
  }
}
}  // namespace NAMESPACE
