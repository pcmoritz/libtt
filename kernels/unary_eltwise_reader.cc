#include <cstdint>

namespace {

void fill_tile(uint32_t cb, uint32_t packed_value) {
  uint32_t l1_addr = get_write_ptr(cb);
  volatile tt_l1_ptr uint32_t *ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(l1_addr);
  uint32_t words = get_tile_size(cb) / sizeof(uint32_t);
  for (uint32_t i = 0; i < words; ++i) {
    ptr[i] = packed_value;
  }
}

}  // namespace

void kernel_main() {
  uint32_t input_addr = get_arg_val<uint32_t>(0);
  uint32_t offset = get_arg_val<uint32_t>(1);
  uint32_t n_tiles = get_arg_val<uint32_t>(2);
  uint32_t input_constant = get_arg_val<uint32_t>(3);

  constexpr uint32_t cb_input = tt::CBIndex::c_0;
  const InterleavedAddrGenFast<true> input = {
    .bank_base_address = input_addr, .page_size = get_tile_size(cb_input), .data_format = get_dataformat(cb_input),
  };

  for (uint32_t i = 0; i < n_tiles; ++i) {
    cb_reserve_back(cb_input, 1);
    if (input_addr == 0) {
      fill_tile(cb_input, input_constant);
    } else {
      noc_async_read_tile(offset + i, input, get_write_ptr(cb_input));
    }
    noc_async_read_barrier();
    cb_push_back(cb_input, 1);
  }
}
