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

uint32_t float_bits(float value) {
  union Bits {
    float f;
    uint32_t u;
  };
  Bits bits;
  bits.f = value;
  return bits.u;
}

void store_iota_value(uint32_t tile_l1_addr, uint32_t element, uint32_t value) {
#if defined(IOTA_DTYPE_INT32) || defined(IOTA_DTYPE_UINT32)
  volatile tt_l1_ptr uint32_t *ptr =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(tile_l1_addr);
  ptr[element] = value;
#elif defined(IOTA_DTYPE_FLOAT32)
  volatile tt_l1_ptr uint32_t *ptr =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(tile_l1_addr);
  ptr[element] = float_bits(static_cast<float>(value));
#elif defined(IOTA_DTYPE_BFLOAT16)
  volatile tt_l1_ptr uint16_t *ptr =
      reinterpret_cast<volatile tt_l1_ptr uint16_t *>(tile_l1_addr);
  ptr[element] = static_cast<uint16_t>(float_bits(static_cast<float>(value)) >> 16);
#endif
}

}  // namespace

void kernel_main() {
  uint32_t output_addr = get_arg_val<uint32_t>(0);
  uint32_t tile_offset = get_arg_val<uint32_t>(1);
  uint32_t tile_count = get_arg_val<uint32_t>(2);
  uint32_t rank = get_arg_val<uint32_t>(3);
  uint32_t dim0 = get_arg_val<uint32_t>(4);
  uint32_t dim1 = get_arg_val<uint32_t>(5);
  uint32_t iota_dimension = get_arg_val<uint32_t>(6);
  uint32_t output_tiles_per_row = get_arg_val<uint32_t>(7);

  constexpr uint32_t cb_output = tt::CBIndex::c_16;
  const InterleavedAddrGenFast<true> output = {
      .bank_base_address = output_addr,
      .page_size = get_tile_size(cb_output),
      .data_format = get_dataformat(cb_output),
  };

  for (uint32_t i = 0; i < tile_count; ++i) {
    uint32_t tile_id = tile_offset + i;
    uint32_t tile_row = tile_id / output_tiles_per_row;
    uint32_t tile_col = tile_id % output_tiles_per_row;

    cb_reserve_back(cb_output, 1);
    zero_tile(cb_output);
    uint32_t tile_l1_addr = get_write_ptr(cb_output);

    for (uint32_t row = 0; row < TILE_R; ++row) {
      uint32_t logical_row = tile_row * TILE_R + row;
      for (uint32_t col = 0; col < TILE_C; ++col) {
        uint32_t logical_col = tile_col * TILE_C + col;
        bool valid = false;
        uint32_t value = 0;

        if (rank == 1) {
          valid = logical_row == 0 && logical_col < dim0;
          value = logical_col;
        } else {
          valid = logical_row < dim0 && logical_col < dim1;
          value = iota_dimension == 0 ? logical_row : logical_col;
        }

        if (valid) {
          store_iota_value(tile_l1_addr, tile_element_index(row, col), value);
        }
      }
    }

    noc_async_write_tile(tile_id, output, tile_l1_addr);
    noc_async_write_barrier();
  }
}
