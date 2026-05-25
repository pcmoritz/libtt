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

}  // namespace

void kernel_main() {
  uint32_t operand_addr = get_arg_val<uint32_t>(0);
  uint32_t start_indices_addr = get_arg_val<uint32_t>(1);
  uint32_t output_tile_offset = get_arg_val<uint32_t>(2);
  uint32_t output_tile_count = get_arg_val<uint32_t>(3);
  uint32_t logical_output_elements = get_arg_val<uint32_t>(4);
  uint32_t logical_operand_elements = get_arg_val<uint32_t>(5);

  constexpr uint32_t cb_indices = tt::CBIndex::c_0;
  constexpr uint32_t cb_operand = tt::CBIndex::c_1;
  constexpr uint32_t cb_output = tt::CBIndex::c_16;

  const InterleavedAddrGenFast<true> operand = {
      .bank_base_address = operand_addr,
      .page_size = get_tile_size(cb_operand),
      .data_format = get_dataformat(cb_operand),
  };
  const InterleavedAddrGenFast<true> start_indices = {
      .bank_base_address = start_indices_addr,
      .page_size = get_tile_size(cb_indices),
      .data_format = get_dataformat(cb_indices),
  };

  for (uint32_t i = 0; i < output_tile_count; ++i) {
    uint32_t output_tile = output_tile_offset + i;

    cb_reserve_back(cb_indices, 1);
    noc_async_read_tile(output_tile, start_indices, get_write_ptr(cb_indices));
    noc_async_read_barrier();
    cb_push_back(cb_indices, 1);
    cb_wait_front(cb_indices, 1);
    volatile tt_l1_ptr int32_t *indices =
        reinterpret_cast<volatile tt_l1_ptr int32_t *>(get_read_ptr(cb_indices));

    cb_reserve_back(cb_output, 1);
    zero_tile(cb_output);
    volatile tt_l1_ptr int32_t *output =
        reinterpret_cast<volatile tt_l1_ptr int32_t *>(get_write_ptr(cb_output));

    for (uint32_t col = 0; col < TILE_C; ++col) {
      uint32_t logical_output = output_tile * TILE_C + col;
      if (logical_output >= logical_output_elements) {
        continue;
      }

      int32_t index = indices[tile_element_index(col, 0)];
      if (index < 0 || static_cast<uint32_t>(index) >= logical_operand_elements) {
        continue;
      }

      uint32_t operand_col = static_cast<uint32_t>(index);
      uint32_t operand_tile = operand_col / TILE_C;
      uint32_t operand_tile_col = operand_col % TILE_C;

      cb_reserve_back(cb_operand, 1);
      noc_async_read_tile(operand_tile, operand, get_write_ptr(cb_operand));
      noc_async_read_barrier();
      cb_push_back(cb_operand, 1);
      cb_wait_front(cb_operand, 1);
      volatile tt_l1_ptr int32_t *operand_tile_ptr =
          reinterpret_cast<volatile tt_l1_ptr int32_t *>(get_read_ptr(cb_operand));

      output[tile_element_index(0, col)] = operand_tile_ptr[tile_element_index(0, operand_tile_col)];
      cb_pop_front(cb_operand, 1);
    }

    cb_push_back(cb_output, 1);
    cb_pop_front(cb_indices, 1);
  }
}
