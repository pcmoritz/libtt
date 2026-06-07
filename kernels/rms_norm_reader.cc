#include <cstdint>

namespace {

constexpr uint32_t TILE_R = 32;
constexpr uint32_t TILE_C = 32;
constexpr uint32_t FACE_ELEMENTS = 16 * 16;
constexpr uint32_t BF16_ONE_BITS = 0x3f80u;

uint32_t tile_element_index(uint32_t row, uint32_t col) {
  uint32_t face_row = row / 16;
  uint32_t face_col = col / 16;
  uint32_t row_in_face = row % 16;
  uint32_t col_in_face = col % 16;
  return ((face_row * 2 + face_col) * FACE_ELEMENTS) + row_in_face * 16 +
         col_in_face;
}

void fill_bf16_tile(uint32_t cb, uint32_t value_bits) {
  volatile tt_l1_ptr uint16_t *ptr =
      reinterpret_cast<volatile tt_l1_ptr uint16_t *>(get_write_ptr(cb));
  uint16_t value = static_cast<uint16_t>(value_bits);
  uint32_t elements = get_tile_size(cb) / sizeof(uint16_t);
  for (uint32_t i = 0; i < elements; ++i) {
    ptr[i] = value;
  }
}

void zero_padded_rows(uint32_t tile_addr) {
  static_assert(RMS_NORM_VALID_ROWS > 0 && RMS_NORM_VALID_ROWS <= TILE_R,
                "rms_norm valid rows must fit in one tile");
  if constexpr (RMS_NORM_VALID_ROWS == TILE_R) {
    return;
  } else if constexpr (RMS_NORM_VALID_ROWS == 1) {
    volatile tt_l1_ptr uint32_t *tile =
        reinterpret_cast<volatile tt_l1_ptr uint32_t *>(tile_addr);

    // Tile layout stores 16x16 faces contiguously. For a single valid row, keep
    // row 0 in the two top faces and zero the rest with contiguous 32-bit stores.
    constexpr uint32_t face_words = FACE_ELEMENTS / 2;
    constexpr uint32_t valid_row_words_per_face = 16 / 2;
    for (uint32_t face = 0; face < 2; ++face) {
      uint32_t start = face * face_words + valid_row_words_per_face;
      uint32_t end = (face + 1) * face_words;
      for (uint32_t word = start; word < end; ++word) {
        tile[word] = 0;
      }
    }
    for (uint32_t word = 2 * face_words; word < 4 * face_words; ++word) {
      tile[word] = 0;
    }
  } else {
    volatile tt_l1_ptr uint16_t *tile =
        reinterpret_cast<volatile tt_l1_ptr uint16_t *>(tile_addr);
    for (uint32_t row = RMS_NORM_VALID_ROWS; row < TILE_R; ++row) {
      for (uint32_t col = 0; col < TILE_C; ++col) {
        tile[tile_element_index(row, col)] = 0;
      }
    }
  }
}

}  // namespace

void kernel_main() {
  uint32_t input_addr = get_arg_val<uint32_t>(0);
  uint32_t weight_addr = get_arg_val<uint32_t>(1);
  uint32_t group_offset = get_arg_val<uint32_t>(2);
  uint32_t group_count = get_arg_val<uint32_t>(3);

  constexpr uint32_t cb_input = tt::CBIndex::c_0;
  constexpr uint32_t cb_weight = tt::CBIndex::c_1;
  constexpr uint32_t cb_scaler = tt::CBIndex::c_2;
  constexpr uint32_t width_tiles = RMS_NORM_WIDTH_TILES;

  const InterleavedAddrGenFast<true> input = {
      .bank_base_address = input_addr,
      .page_size = get_tile_size(cb_input),
      .data_format = get_dataformat(cb_input),
  };
  const InterleavedAddrGenFast<true> weight = {
      .bank_base_address = weight_addr,
      .page_size = get_tile_size(cb_weight),
      .data_format = get_dataformat(cb_weight),
  };

  cb_reserve_back(cb_scaler, 1);
  fill_bf16_tile(cb_scaler, BF16_ONE_BITS);
  cb_push_back(cb_scaler, 1);

  for (uint32_t group = 0; group < group_count; ++group) {
    uint32_t base_tile = (group_offset + group) * width_tiles;
    uint32_t input_tile_bytes = get_tile_size(cb_input);
    uint32_t weight_tile_bytes = get_tile_size(cb_weight);

    cb_reserve_back(cb_input, width_tiles);
    uint32_t input_write_ptr = get_write_ptr(cb_input);
    for (uint32_t wt = 0; wt < width_tiles; ++wt) {
      noc_async_read_tile(base_tile + wt, input,
                          input_write_ptr + wt * input_tile_bytes);
    }
    noc_async_read_barrier();
    for (uint32_t wt = 0; wt < width_tiles; ++wt) {
      zero_padded_rows(input_write_ptr + wt * input_tile_bytes);
    }
    cb_push_back(cb_input, width_tiles);

    cb_reserve_back(cb_weight, width_tiles);
    uint32_t weight_write_ptr = get_write_ptr(cb_weight);
    for (uint32_t wt = 0; wt < width_tiles; ++wt) {
      noc_async_read_tile(wt, weight, weight_write_ptr + wt * weight_tile_bytes);
    }
    noc_async_read_barrier();
    cb_push_back(cb_weight, width_tiles);
  }
}
