#include <cstdint>

namespace {

constexpr uint32_t pred_tile_bytes = 32 * 32;

void fill_tile(uint32_t cb, uint32_t packed_value) {
  uint32_t l1_addr = get_write_ptr(cb);
  volatile tt_l1_ptr uint32_t *ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(l1_addr);
  uint32_t words = get_tile_size(cb) / sizeof(uint32_t);
  for (uint32_t i = 0; i < words; ++i) {
    ptr[i] = packed_value;
  }
}

void select_output_tile(uint32_t cb_out, uint32_t cb_true, uint32_t cb_false) {
  uint32_t out_addr = get_write_ptr(cb_out);
  uint32_t true_addr = get_write_ptr(cb_true);
  uint32_t false_addr = get_write_ptr(cb_false);
  volatile tt_l1_ptr uint8_t *pred = reinterpret_cast<volatile tt_l1_ptr uint8_t *>(out_addr);
  volatile tt_l1_ptr uint8_t *out = reinterpret_cast<volatile tt_l1_ptr uint8_t *>(out_addr);
  volatile tt_l1_ptr uint8_t *on_true = reinterpret_cast<volatile tt_l1_ptr uint8_t *>(true_addr);
  volatile tt_l1_ptr uint8_t *on_false = reinterpret_cast<volatile tt_l1_ptr uint8_t *>(false_addr);
  uint32_t tile_bytes = get_tile_size(cb_out);
  uint32_t bytes_per_element = tile_bytes / pred_tile_bytes;

  for (int32_t i = tile_bytes - 1; i >= 0; --i) {
    out[i] = pred[static_cast<uint32_t>(i) / bytes_per_element] ? on_true[i] : on_false[i];
  }
}

}  // namespace

void kernel_main() {
  uint32_t pred_addr = get_arg_val<uint32_t>(0);
  uint32_t true_addr = get_arg_val<uint32_t>(1);
  uint32_t false_addr = get_arg_val<uint32_t>(2);
  uint32_t offset = get_arg_val<uint32_t>(3);
  uint32_t n_tiles = get_arg_val<uint32_t>(4);
  uint32_t true_constant = get_arg_val<uint32_t>(5);
  uint32_t false_constant = get_arg_val<uint32_t>(6);

  constexpr uint32_t cb_true = tt::CBIndex::c_1;
  constexpr uint32_t cb_false = tt::CBIndex::c_2;
  constexpr uint32_t cb_selected = tt::CBIndex::c_3;
  constexpr uint32_t cb_zero = tt::CBIndex::c_4;

  const InterleavedAddrGenFast<true> pred = {
    .bank_base_address = pred_addr, .page_size = pred_tile_bytes, .data_format = DataFormat::UInt8,
  };
  const InterleavedAddrGenFast<true> on_true = {
    .bank_base_address = true_addr, .page_size = get_tile_size(cb_true), .data_format = get_dataformat(cb_true),
  };
  const InterleavedAddrGenFast<true> on_false = {
    .bank_base_address = false_addr, .page_size = get_tile_size(cb_false), .data_format = get_dataformat(cb_false),
  };

  for (uint32_t i = 0; i < n_tiles; ++i) {
    cb_reserve_back(cb_selected, 1);
    cb_reserve_back(cb_true, 1);
    cb_reserve_back(cb_false, 1);
    cb_reserve_back(cb_zero, 1);

    noc_async_read_tile(offset + i, pred, get_write_ptr(cb_selected));
    if (true_addr == 0) {
      fill_tile(cb_true, true_constant);
    } else {
      noc_async_read_tile(offset + i, on_true, get_write_ptr(cb_true));
    }
    if (false_addr == 0) {
      fill_tile(cb_false, false_constant);
    } else {
      noc_async_read_tile(offset + i, on_false, get_write_ptr(cb_false));
    }

    noc_async_read_barrier();
    select_output_tile(cb_selected, cb_true, cb_false);
    fill_tile(cb_zero, 0);
    cb_push_back(cb_selected, 1);
    cb_push_back(cb_zero, 1);
  }
}
