#include <cstdint>

namespace {

enum class BroadcastMode {
  Copy,
  Scalar,
  Row,
  Col,
  Transpose,
};

uint32_t input_tile_id(uint32_t output_tile_id, BroadcastMode mode,
                       uint32_t output_tiles_per_batch,
                       uint32_t output_tiles_per_row,
                       bool broadcast_batch) {
  uint32_t output_batch = output_tile_id / output_tiles_per_batch;
  uint32_t output_tile_in_batch = output_tile_id % output_tiles_per_batch;
  uint32_t output_tile_row = output_tile_in_batch / output_tiles_per_row;
  uint32_t output_tile_col = output_tile_in_batch % output_tiles_per_row;
  uint32_t input_batch = broadcast_batch ? 0 : output_batch;
  uint32_t input_tiles_per_batch = output_tiles_per_batch;
  if (mode == BroadcastMode::Scalar) {
    input_tiles_per_batch = 1;
  } else if (mode == BroadcastMode::Row) {
    input_tiles_per_batch = output_tiles_per_row;
  } else if (mode == BroadcastMode::Col) {
    input_tiles_per_batch = output_tiles_per_batch / output_tiles_per_row;
  }
  uint32_t input_base = input_batch * input_tiles_per_batch;

  if (mode == BroadcastMode::Copy) {
    return input_base + output_tile_in_batch;
  }
  if (mode == BroadcastMode::Scalar) {
    return input_base;
  }
  if (mode == BroadcastMode::Row) {
    return input_base + output_tile_col;
  }
  if (mode == BroadcastMode::Col) {
    return input_base + output_tile_row;
  }
  return input_base + output_tile_row;
}

}  // namespace

void kernel_main() {
  uint32_t input_addr = get_arg_val<uint32_t>(0);
  uint32_t offset = get_arg_val<uint32_t>(1);
  uint32_t n_tiles = get_arg_val<uint32_t>(2);
  uint32_t output_tiles_per_batch = get_arg_val<uint32_t>(3);
  uint32_t output_tiles_per_row = get_arg_val<uint32_t>(4);
  bool broadcast_batch = get_arg_val<uint32_t>(5) != 0;

  constexpr uint32_t cb_input = tt::CBIndex::c_0;
  constexpr BroadcastMode mode = BroadcastMode::BROADCAST_MODE;
  const InterleavedAddrGenFast<true> input = {
      .bank_base_address = input_addr,
      .page_size = get_tile_size(cb_input),
      .data_format = get_dataformat(cb_input),
  };

  for (uint32_t i = 0; i < n_tiles; ++i) {
    uint32_t output_tile_id = offset + i;
    cb_reserve_back(cb_input, 1);
    noc_async_read_tile(
        input_tile_id(output_tile_id, mode, output_tiles_per_batch,
                      output_tiles_per_row, broadcast_batch),
        input,
        get_write_ptr(cb_input));
    noc_async_read_barrier();
    cb_push_back(cb_input, 1);
  }
}
