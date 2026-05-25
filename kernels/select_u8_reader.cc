#include <cstdint>

namespace {

constexpr uint32_t tile_elements = 32 * 32;
constexpr uint32_t pred_tile_bytes = tile_elements;
using Element = SELECT_RAW_ELEMENT_TYPE;

void fill_tile(uint32_t cb, Element value) {
  uint32_t l1_addr = get_write_ptr(cb);
  volatile tt_l1_ptr Element *ptr =
      reinterpret_cast<volatile tt_l1_ptr Element *>(l1_addr);
  for (uint32_t i = 0; i < tile_elements; ++i) {
    ptr[i] = value;
  }
}

}  // namespace

void kernel_main() {
  uint32_t pred_addr = get_arg_val<uint32_t>(0);
  uint32_t true_addr = get_arg_val<uint32_t>(1);
  uint32_t false_addr = get_arg_val<uint32_t>(2);
  uint32_t offset = get_arg_val<uint32_t>(3);
  uint32_t n_tiles = get_arg_val<uint32_t>(4);
  Element true_constant = static_cast<Element>(get_arg_val<uint32_t>(5));
  Element false_constant = static_cast<Element>(get_arg_val<uint32_t>(6));

  constexpr uint32_t cb_pred = tt::CBIndex::c_0;
  constexpr uint32_t cb_true = tt::CBIndex::c_1;
  constexpr uint32_t cb_false = tt::CBIndex::c_2;
  constexpr uint32_t cb_out = tt::CBIndex::c_16;

  const InterleavedAddrGenFast<true> pred = {
      .bank_base_address = pred_addr,
      .page_size = pred_tile_bytes,
      .data_format = DataFormat::UInt8,
  };
  const InterleavedAddrGenFast<true> on_true = {
      .bank_base_address = true_addr,
      .page_size = get_tile_size(cb_true),
      .data_format = get_dataformat(cb_true),
  };
  const InterleavedAddrGenFast<true> on_false = {
      .bank_base_address = false_addr,
      .page_size = get_tile_size(cb_false),
      .data_format = get_dataformat(cb_false),
  };

  for (uint32_t tile = 0; tile < n_tiles; ++tile) {
    cb_reserve_back(cb_pred, 1);
    cb_reserve_back(cb_true, 1);
    cb_reserve_back(cb_false, 1);

    noc_async_read_tile(offset + tile, pred, get_write_ptr(cb_pred));
    if (true_addr == 0) {
      fill_tile(cb_true, true_constant);
    } else {
      noc_async_read_tile(offset + tile, on_true, get_write_ptr(cb_true));
    }
    if (false_addr == 0) {
      fill_tile(cb_false, false_constant);
    } else {
      noc_async_read_tile(offset + tile, on_false, get_write_ptr(cb_false));
    }
    noc_async_read_barrier();

    cb_push_back(cb_pred, 1);
    cb_push_back(cb_true, 1);
    cb_push_back(cb_false, 1);
    cb_wait_front(cb_pred, 1);
    cb_wait_front(cb_true, 1);
    cb_wait_front(cb_false, 1);
    cb_reserve_back(cb_out, 1);

    volatile tt_l1_ptr const uint8_t *pred_ptr =
        reinterpret_cast<volatile tt_l1_ptr const uint8_t *>(get_read_ptr(cb_pred));
    volatile tt_l1_ptr const Element *true_ptr =
        reinterpret_cast<volatile tt_l1_ptr const Element *>(get_read_ptr(cb_true));
    volatile tt_l1_ptr const Element *false_ptr =
        reinterpret_cast<volatile tt_l1_ptr const Element *>(get_read_ptr(cb_false));
    volatile tt_l1_ptr Element *out_ptr =
        reinterpret_cast<volatile tt_l1_ptr Element *>(get_write_ptr(cb_out));
    for (uint32_t i = 0; i < tile_elements; ++i) {
      out_ptr[i] = pred_ptr[i] != 0 ? true_ptr[i] : false_ptr[i];
    }

    cb_pop_front(cb_pred, 1);
    cb_pop_front(cb_true, 1);
    cb_pop_front(cb_false, 1);
    cb_push_back(cb_out, 1);
  }
}
