#include <cstdint>
void kernel_main() {
  uint32_t lhs_addr = get_arg_val<uint32_t>(0);
  uint32_t rhs_addr = get_arg_val<uint32_t>(1);
  uint32_t offset = get_arg_val<uint32_t>(2);
  uint32_t n_tiles = get_arg_val<uint32_t>(3);

  constexpr uint32_t cb_lhs = tt::CBIndex::c_0;
  constexpr uint32_t cb_rhs = tt::CBIndex::c_1;
  const InterleavedAddrGenFast<true> lhs = {
    .bank_base_address = lhs_addr, .page_size = get_tile_size(cb_lhs), .data_format = DataFormat::Float16_b,
  };
  const InterleavedAddrGenFast<true> rhs = {
    .bank_base_address = rhs_addr, .page_size = get_tile_size(cb_rhs), .data_format = DataFormat::Float16_b,
  };

  for (uint32_t i = 0; i < n_tiles; ++i) {
    cb_reserve_back(cb_lhs, 1);
    cb_reserve_back(cb_rhs, 1);
    noc_async_read_tile(offset + i, lhs, get_write_ptr(cb_lhs));
    noc_async_read_tile(offset + i, rhs, get_write_ptr(cb_rhs));
    noc_async_read_barrier();
    cb_push_back(cb_lhs, 1);
    cb_push_back(cb_rhs, 1);
  }
}
