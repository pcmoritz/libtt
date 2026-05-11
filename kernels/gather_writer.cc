#include <cstdint>

void kernel_main() {
  uint32_t output_addr = get_arg_val<uint32_t>(0);
  uint32_t output_row_tile_offset = get_arg_val<uint32_t>(1);
  uint32_t output_row_tile_count = get_arg_val<uint32_t>(2);
  uint32_t output_tiles_per_row = get_arg_val<uint32_t>(3);

  constexpr uint32_t cb_output = tt::CBIndex::c_16;
  const InterleavedAddrGenFast<true> output = {
      .bank_base_address = output_addr,
      .page_size = get_tile_size(cb_output),
      .data_format = get_dataformat(cb_output),
  };

  for (uint32_t row_tile = 0; row_tile < output_row_tile_count; ++row_tile) {
    uint32_t output_row_tile = output_row_tile_offset + row_tile;
    for (uint32_t tile_col = 0; tile_col < output_tiles_per_row; ++tile_col) {
      cb_wait_front(cb_output, 1);
      noc_async_write_tile(output_row_tile * output_tiles_per_row + tile_col, output,
                           get_read_ptr(cb_output));
      noc_async_write_barrier();
      cb_pop_front(cb_output, 1);
    }
  }
}
