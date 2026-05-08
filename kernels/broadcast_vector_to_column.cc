#include <cstdint>

namespace {

uint32_t tile_element_offset(uint32_t row, uint32_t col) {
  uint32_t face_row = row >> 4;
  uint32_t face_col = col >> 4;
  uint32_t local_row = row & 0xf;
  uint32_t local_col = col & 0xf;
  return ((face_row * 2 + face_col) * 256) + local_row * 16 + local_col;
}

void zero_tile(uint32_t l1_addr, uint32_t tile_bytes) {
  volatile tt_l1_ptr uint32_t *ptr =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(l1_addr);
  for (uint32_t i = 0; i < tile_bytes / sizeof(uint32_t); ++i) {
    ptr[i] = 0;
  }
}

void copy_element(uint32_t src_l1_addr, uint32_t dst_l1_addr, uint32_t src_element,
                  uint32_t dst_element, uint32_t element_bytes) {
  if (element_bytes == sizeof(uint32_t)) {
    volatile tt_l1_ptr uint32_t *src =
        reinterpret_cast<volatile tt_l1_ptr uint32_t *>(src_l1_addr);
    volatile tt_l1_ptr uint32_t *dst =
        reinterpret_cast<volatile tt_l1_ptr uint32_t *>(dst_l1_addr);
    dst[dst_element] = src[src_element];
    return;
  }
  if (element_bytes == sizeof(uint16_t)) {
    volatile tt_l1_ptr uint16_t *src =
        reinterpret_cast<volatile tt_l1_ptr uint16_t *>(src_l1_addr);
    volatile tt_l1_ptr uint16_t *dst =
        reinterpret_cast<volatile tt_l1_ptr uint16_t *>(dst_l1_addr);
    dst[dst_element] = src[src_element];
    return;
  }
  volatile tt_l1_ptr uint8_t *src =
      reinterpret_cast<volatile tt_l1_ptr uint8_t *>(src_l1_addr);
  volatile tt_l1_ptr uint8_t *dst =
      reinterpret_cast<volatile tt_l1_ptr uint8_t *>(dst_l1_addr);
  dst[dst_element] = src[src_element];
}

}  // namespace

void kernel_main() {
  uint32_t input_addr = get_arg_val<uint32_t>(0);
  uint32_t output_addr = get_arg_val<uint32_t>(1);
  uint32_t offset = get_arg_val<uint32_t>(2);
  uint32_t n_tiles = get_arg_val<uint32_t>(3);

  constexpr uint32_t cb_input = tt::CBIndex::c_0;
  constexpr uint32_t cb_output = tt::CBIndex::c_16;
  const InterleavedAddrGenFast<true> input = {
      .bank_base_address = input_addr,
      .page_size = get_tile_size(cb_input),
      .data_format = get_dataformat(cb_input),
  };
  const InterleavedAddrGenFast<true> output = {
      .bank_base_address = output_addr,
      .page_size = get_tile_size(cb_output),
      .data_format = get_dataformat(cb_output),
  };

  uint32_t tile_bytes = get_tile_size(cb_input);
  uint32_t element_bytes = tile_bytes / (32 * 32);

  for (uint32_t i = 0; i < n_tiles; ++i) {
    cb_reserve_back(cb_input, 1);
    cb_reserve_back(cb_output, 1);

    uint32_t tile_id = offset + i;
    uint32_t input_l1 = get_write_ptr(cb_input);
    uint32_t output_l1 = get_write_ptr(cb_output);

    noc_async_read_tile(tile_id, input, input_l1);
    noc_async_read_barrier();
    zero_tile(output_l1, get_tile_size(cb_output));

    for (uint32_t row = 0; row < 32; ++row) {
      uint32_t src_element = tile_element_offset(0, row);
      uint32_t dst_element = tile_element_offset(row, 0);
      copy_element(input_l1, output_l1, src_element, dst_element, element_bytes);
    }

    noc_async_write_tile(tile_id, output, output_l1);
    noc_async_write_barrier();

    cb_push_back(cb_input, 1);
    cb_pop_front(cb_input, 1);
    cb_push_back(cb_output, 1);
    cb_pop_front(cb_output, 1);
  }
}
