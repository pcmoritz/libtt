#include <cstdint>

void kernel_main() {
  uint32_t input_addr = get_arg_val<uint32_t>(0);
  uint32_t reduce_groups = get_arg_val<uint32_t>(1);
  uint32_t width_tiles = get_arg_val<uint32_t>(2);

  constexpr uint32_t cb_input = tt::CBIndex::c_0;
  const InterleavedAddrGenFast<true> input = {
      .bank_base_address = input_addr,
      .page_size = get_tile_size(cb_input),
      .data_format = get_dataformat(cb_input),
  };

  for (uint32_t group = 0; group < reduce_groups; ++group) {
    uint32_t tile_base = group * width_tiles;
    for (uint32_t wt = 0; wt < width_tiles; ++wt) {
      cb_reserve_back(cb_input, 1);
      noc_async_read_tile(tile_base + wt, input, get_write_ptr(cb_input));
      noc_async_read_barrier();
      cb_push_back(cb_input, 1);
    }
  }
}
