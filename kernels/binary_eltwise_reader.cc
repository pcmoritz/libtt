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
  uint32_t lhs_addr = get_arg_val<uint32_t>(0);
  uint32_t rhs_addr = get_arg_val<uint32_t>(1);
  uint32_t offset = get_arg_val<uint32_t>(2);
  uint32_t n_tiles = get_arg_val<uint32_t>(3);
  uint32_t lhs_constant = get_arg_val<uint32_t>(4);
  uint32_t rhs_constant = get_arg_val<uint32_t>(5);

  constexpr uint32_t cb_lhs = tt::CBIndex::c_0;
  constexpr uint32_t cb_rhs = tt::CBIndex::c_1;
  const InterleavedAddrGenFast<true> lhs = {
    .bank_base_address = lhs_addr, .page_size = get_tile_size(cb_lhs), .data_format = get_dataformat(cb_lhs),
  };
  const InterleavedAddrGenFast<true> rhs = {
    .bank_base_address = rhs_addr, .page_size = get_tile_size(cb_rhs), .data_format = get_dataformat(cb_rhs),
  };

  for (uint32_t i = 0; i < n_tiles; ++i) {
    cb_reserve_back(cb_lhs, 1);
    cb_reserve_back(cb_rhs, 1);
    if (lhs_addr == 0) {
      fill_tile(cb_lhs, lhs_constant);
    } else {
      noc_async_read_tile(offset + i, lhs, get_write_ptr(cb_lhs));
    }
    if (rhs_addr == 0) {
      fill_tile(cb_rhs, rhs_constant);
    } else {
      noc_async_read_tile(offset + i, rhs, get_write_ptr(cb_rhs));
    }
    noc_async_read_barrier();
    cb_push_back(cb_lhs, 1);
    cb_push_back(cb_rhs, 1);
  }
}
