#include <cstdint>

namespace {

constexpr uint32_t pred_tile_bytes = 32 * 32;

#ifdef SELECT_RAW_ELEMENT_TYPE
using Element = SELECT_RAW_ELEMENT_TYPE;

void fill_tile(uint32_t cb, Element value) {
  uint32_t l1_addr = get_write_ptr(cb);
  volatile tt_l1_ptr Element *ptr =
      reinterpret_cast<volatile tt_l1_ptr Element *>(l1_addr);
  uint32_t elements = get_tile_size(cb) / sizeof(Element);
  for (uint32_t i = 0; i < elements; ++i) {
    ptr[i] = value;
  }
}
#else

void fill_tile(uint32_t cb, uint32_t packed_value) {
  uint32_t l1_addr = get_write_ptr(cb);
  volatile tt_l1_ptr uint32_t *ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(l1_addr);
  uint32_t words = get_tile_size(cb) / sizeof(uint32_t);
  for (uint32_t i = 0; i < words; ++i) {
    ptr[i] = packed_value;
  }
}

#endif

}  // namespace

void kernel_main() {
  uint32_t pred_addr = get_arg_val<uint32_t>(0);
  uint32_t true_addr = get_arg_val<uint32_t>(1);
  uint32_t false_addr = get_arg_val<uint32_t>(2);
  uint32_t offset = get_arg_val<uint32_t>(3);
  uint32_t n_tiles = get_arg_val<uint32_t>(4);
#ifdef SELECT_RAW_ELEMENT_TYPE
  Element true_constant = static_cast<Element>(get_arg_val<uint32_t>(5));
  Element false_constant = static_cast<Element>(get_arg_val<uint32_t>(6));
#else
  uint32_t true_constant = get_arg_val<uint32_t>(5);
  uint32_t false_constant = get_arg_val<uint32_t>(6);
#endif

  constexpr uint32_t cb_pred = tt::CBIndex::c_0;
  constexpr uint32_t cb_true = tt::CBIndex::c_1;
  constexpr uint32_t cb_false = tt::CBIndex::c_2;
#ifdef SELECT_RAW_ELEMENT_TYPE
  constexpr uint32_t cb_out = tt::CBIndex::c_16;
#endif

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
    cb_reserve_back(cb_pred, 1);
    cb_reserve_back(cb_true, 1);
    cb_reserve_back(cb_false, 1);

    noc_async_read_tile(offset + i, pred, get_write_ptr(cb_pred));
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
    cb_push_back(cb_pred, 1);
    cb_push_back(cb_true, 1);
    cb_push_back(cb_false, 1);

#ifdef SELECT_RAW_ELEMENT_TYPE
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
    uint32_t elements = get_tile_size(cb_out) / sizeof(Element);
    for (uint32_t element = 0; element < elements; ++element) {
      out_ptr[element] = pred_ptr[element] != 0 ? true_ptr[element] : false_ptr[element];
    }

    cb_pop_front(cb_pred, 1);
    cb_pop_front(cb_true, 1);
    cb_pop_front(cb_false, 1);
    cb_push_back(cb_out, 1);
#endif
  }
}
