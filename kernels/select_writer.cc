#include <cstdint>

void kernel_main() {
  uint32_t out_addr = get_arg_val<uint32_t>(0);
  uint32_t offset = get_arg_val<uint32_t>(1);
  uint32_t n_tiles = get_arg_val<uint32_t>(2);

  constexpr uint32_t cb_selected = tt::CBIndex::c_16;
  const InterleavedAddrGenFast<true> out = {
    .bank_base_address = out_addr, .page_size = get_tile_size(cb_selected), .data_format = get_dataformat(cb_selected),
  };

  for (uint32_t i = 0; i < n_tiles; ++i) {
    cb_wait_front(cb_selected, 1);
    noc_async_write_tile(offset + i, out, get_read_ptr(cb_selected));
    noc_async_write_barrier();
    cb_pop_front(cb_selected, 1);
  }
}
