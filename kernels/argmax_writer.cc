#include <cstdint>

namespace {

constexpr uint32_t TILE_R = 32;
constexpr uint32_t TILE_C = 32;
constexpr uint32_t FACE_R = 16;
constexpr uint32_t FACE_C = 16;

uint32_t tile_element_index(uint32_t row, uint32_t col) {
  uint32_t face_row = row / FACE_R;
  uint32_t face_col = col / FACE_C;
  uint32_t row_in_face = row % FACE_R;
  uint32_t col_in_face = col % FACE_C;
  return ((face_row * 2 + face_col) * FACE_R * FACE_C) + row_in_face * FACE_C + col_in_face;
}

void zero_tile(uint32_t cb) {
  volatile tt_l1_ptr uint32_t *ptr =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(get_write_ptr(cb));
  uint32_t words = get_tile_size(cb) / sizeof(uint32_t);
  for (uint32_t i = 0; i < words; ++i) {
    ptr[i] = 0;
  }
}

uint32_t ordered_float_key(uint32_t bits) {
  return (bits & 0x80000000u) != 0 ? ~bits : (bits ^ 0x80000000u);
}

uint32_t load_ordered_key(uint32_t tile_l1_addr, uint32_t element) {
#if defined(ARGMAX_DTYPE_BFLOAT16)
  volatile tt_l1_ptr uint16_t *ptr =
      reinterpret_cast<volatile tt_l1_ptr uint16_t *>(tile_l1_addr);
  return ordered_float_key(static_cast<uint32_t>(ptr[element]) << 16);
#elif defined(ARGMAX_DTYPE_FLOAT32)
  volatile tt_l1_ptr uint32_t *ptr =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(tile_l1_addr);
  return ordered_float_key(ptr[element]);
#endif
}

}  // namespace

void kernel_main() {
  uint32_t input_addr = get_arg_val<uint32_t>(0);
  uint32_t output_addr = get_arg_val<uint32_t>(1);
  uint32_t logical_len = get_arg_val<uint32_t>(2);
  uint32_t input_tiles = get_arg_val<uint32_t>(3);

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

  uint32_t best_key = 0;
  uint32_t best_index = 0;
  bool have_best = false;

  for (uint32_t tile_id = 0; tile_id < input_tiles; ++tile_id) {
    uint32_t base_index = tile_id * TILE_C;
    if (base_index >= logical_len) {
      break;
    }
    uint32_t valid_cols = logical_len - base_index;
    if (valid_cols > TILE_C) {
      valid_cols = TILE_C;
    }

    cb_reserve_back(cb_input, 1);
    uint32_t input_l1_addr = get_write_ptr(cb_input);
    noc_async_read_tile(tile_id, input, input_l1_addr);
    noc_async_read_barrier();

    for (uint32_t col = 0; col < valid_cols; ++col) {
      uint32_t key = load_ordered_key(input_l1_addr, tile_element_index(0, col));
      if (!have_best || key > best_key) {
        best_key = key;
        best_index = base_index + col;
        have_best = true;
      }
    }
    cb_push_back(cb_input, 1);
    cb_pop_front(cb_input, 1);
  }

  cb_reserve_back(cb_output, 1);
  zero_tile(cb_output);
  volatile tt_l1_ptr int32_t *out =
      reinterpret_cast<volatile tt_l1_ptr int32_t *>(get_write_ptr(cb_output));
  out[0] = static_cast<int32_t>(best_index);
  noc_async_write_tile(0, output, get_write_ptr(cb_output));
  noc_async_write_barrier();
  cb_push_back(cb_output, 1);
  cb_pop_front(cb_output, 1);
}
