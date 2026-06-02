#include <cstdint>

void kernel_main() {
  uint32_t output_addr = get_arg_val<uint32_t>(0);
  uint32_t packed_word = get_arg_val<uint32_t>(1);
  uint32_t tile_offset = get_arg_val<uint32_t>(2);
  uint32_t tile_count = get_arg_val<uint32_t>(3);

  constexpr uint32_t cb_output = tt::CBIndex::c_16;
  const InterleavedAddrGenFast<true> output = {
      .bank_base_address = output_addr,
      .page_size = get_tile_size(cb_output),
      .data_format = get_dataformat(cb_output),
  };

  uint32_t words = get_tile_size(cb_output) / sizeof(uint32_t);
  for (uint32_t local_tile = 0; local_tile < tile_count; ++local_tile) {
    cb_reserve_back(cb_output, 1);
    volatile tt_l1_ptr uint32_t *ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t *>(get_write_ptr(cb_output));
    for (uint32_t word = 0; word < words; ++word) {
      ptr[word] = packed_word;
    }
    noc_async_write_tile(tile_offset + local_tile, output, get_write_ptr(cb_output));
    noc_async_write_barrier();
    cb_push_back(cb_output, 1);
    cb_pop_front(cb_output, 1);
  }
}
