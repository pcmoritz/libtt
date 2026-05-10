#include <cstdint>

void kernel_main() {
  uint32_t output_addr = get_arg_val<uint32_t>(0);
  uint32_t offset = get_arg_val<uint32_t>(1);
  uint32_t n_tiles = get_arg_val<uint32_t>(2);

  constexpr uint32_t cb_output = tt::CBIndex::c_16;
  const InterleavedAddrGenFast<true> output = {
      .bank_base_address = output_addr,
      .page_size = get_tile_size(cb_output),
      .data_format = get_dataformat(cb_output),
  };

  for (uint32_t i = 0; i < n_tiles; ++i) {
    cb_wait_front(cb_output, 1);
    noc_async_write_tile(offset + i, output, get_read_ptr(cb_output));
    noc_async_write_barrier();
    cb_pop_front(cb_output, 1);
  }
}
