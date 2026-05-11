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

void copy_bf16_row(uint32_t source_l1_addr, uint32_t output_l1_addr, uint32_t source_row,
                   uint32_t output_row) {
  volatile tt_l1_ptr uint16_t *source =
      reinterpret_cast<volatile tt_l1_ptr uint16_t *>(source_l1_addr);
  volatile tt_l1_ptr uint16_t *output =
      reinterpret_cast<volatile tt_l1_ptr uint16_t *>(output_l1_addr);
  for (uint32_t col = 0; col < TILE_C; ++col) {
    output[tile_element_index(output_row, col)] = source[tile_element_index(source_row, col)];
  }
}

}  // namespace

void kernel_main() {
  uint32_t operand_addr = get_arg_val<uint32_t>(0);
  uint32_t start_indices_addr = get_arg_val<uint32_t>(1);
  uint32_t output_row_tile_offset = get_arg_val<uint32_t>(2);
  uint32_t output_row_tile_count = get_arg_val<uint32_t>(3);
  uint32_t logical_output_rows = get_arg_val<uint32_t>(4);
  uint32_t operand_tiles_per_row = get_arg_val<uint32_t>(5);
  uint32_t output_tiles_per_row = get_arg_val<uint32_t>(6);
  uint32_t logical_operand_rows = get_arg_val<uint32_t>(7);

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
  for (uint32_t row_tile = 0; row_tile < output_row_tile_count; ++row_tile) {
    uint32_t output_row_tile = output_row_tile_offset + row_tile;

    cb_reserve_back(cb_indices, 1);
    noc_async_read_tile(output_row_tile, start_indices, get_write_ptr(cb_indices));
    noc_async_read_barrier();
    cb_push_back(cb_indices, 1);
    cb_wait_front(cb_indices, 1);
    uint32_t indices_l1_addr = get_read_ptr(cb_indices);
    volatile tt_l1_ptr int32_t *indices =
        reinterpret_cast<volatile tt_l1_ptr int32_t *>(indices_l1_addr);

    for (uint32_t tile_col = 0; tile_col < output_tiles_per_row; ++tile_col) {
      cb_reserve_back(cb_output, 1);
      zero_tile(cb_output);
      uint32_t output_l1_addr = get_write_ptr(cb_output);

      for (uint32_t row = 0; row < TILE_R; ++row) {
        uint32_t logical_row = output_row_tile * TILE_R + row;
        if (logical_row >= logical_output_rows) {
          continue;
        }

        int32_t token = indices[tile_element_index(row, 0)];
        if (token < 0 || static_cast<uint32_t>(token) >= logical_operand_rows) {
          continue;
        }

        uint32_t token_row = static_cast<uint32_t>(token);
        uint32_t operand_tile_row = token_row / TILE_R;
        uint32_t operand_row = token_row % TILE_R;
        uint32_t operand_tile_id = operand_tile_row * operand_tiles_per_row + tile_col;

        cb_reserve_back(cb_operand, 1);
        noc_async_read_tile(operand_tile_id, operand, get_write_ptr(cb_operand));
        noc_async_read_barrier();
        cb_push_back(cb_operand, 1);
        cb_wait_front(cb_operand, 1);
        copy_bf16_row(get_read_ptr(cb_operand), output_l1_addr, operand_row, row);
        cb_pop_front(cb_operand, 1);
      }

      cb_push_back(cb_output, 1);
    }

    cb_pop_front(cb_indices, 1);
  }
}
