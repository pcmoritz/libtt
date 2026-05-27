#include <cstdint>

namespace {

constexpr uint32_t TILE_C = 32;
constexpr uint32_t FACE_R = 16;
constexpr uint32_t FACE_C = 16;
constexpr uint32_t INVALID_TILE = 0xffffffffu;
constexpr uint32_t OPERAND_SIZE = DUS_OPERAND_SIZE;
constexpr uint32_t UPDATE_SIZE = DUS_UPDATE_SIZE;
using Element = DUS_ELEMENT_TYPE;

uint32_t tile_element_index(uint32_t row, uint32_t col) {
  uint32_t face_row = row / FACE_R;
  uint32_t face_col = col / FACE_C;
  uint32_t row_in_face = row % FACE_R;
  uint32_t col_in_face = col % FACE_C;
  return ((face_row * 2 + face_col) * FACE_R * FACE_C) + row_in_face * FACE_C + col_in_face;
}

uint32_t tile_extent(uint32_t logical_dim, uint32_t base, uint32_t tile_dim) {
  if (base >= logical_dim) {
    return 0;
  }
  uint32_t remaining = logical_dim - base;
  return remaining < tile_dim ? remaining : tile_dim;
}

void read_tile_to_cb(const InterleavedAddrGenFast<true> &input, uint32_t tile_id,
                     uint32_t cb) {
  cb_reserve_back(cb, 1);
  noc_async_read_tile(tile_id, input, get_write_ptr(cb));
  noc_async_read_barrier();
  cb_push_back(cb, 1);
  cb_wait_front(cb, 1);
}

void read_operand_tile_to_output(const InterleavedAddrGenFast<true> &operand,
                                 uint32_t tile_id, uint32_t cb_output) {
  cb_reserve_back(cb_output, 1);
  noc_async_read_tile(tile_id, operand, get_write_ptr(cb_output));
  noc_async_read_barrier();
}

int32_t read_start_index(uint32_t start_addr) {
  constexpr uint32_t cb_start = tt::CBIndex::c_1;
  const InterleavedAddrGenFast<true> start_tensor = {
      .bank_base_address = start_addr,
      .page_size = get_tile_size(cb_start),
      .data_format = get_dataformat(cb_start),
  };
  read_tile_to_cb(start_tensor, 0, cb_start);
  volatile tt_l1_ptr int32_t *ptr =
      reinterpret_cast<volatile tt_l1_ptr int32_t *>(get_read_ptr(cb_start));
  int32_t value = ptr[tile_element_index(0, 0)];
  cb_pop_front(cb_start, 1);
  return value;
}

uint32_t clamp_start_index(int32_t raw_start, uint32_t operand_dim,
                           uint32_t update_dim) {
  uint32_t max_start = operand_dim > update_dim ? operand_dim - update_dim : 0;
  if (raw_start <= 0) {
    return 0;
  }
  uint32_t start = static_cast<uint32_t>(raw_start);
  return start > max_start ? max_start : start;
}

void ensure_update_tile(const InterleavedAddrGenFast<true> &updates,
                        uint32_t requested_tile, uint32_t *loaded_tile) {
  constexpr uint32_t cb_updates = tt::CBIndex::c_2;
  if (requested_tile == *loaded_tile) {
    return;
  }
  if (*loaded_tile != INVALID_TILE) {
    cb_pop_front(cb_updates, 1);
  }
  read_tile_to_cb(updates, requested_tile, cb_updates);
  *loaded_tile = requested_tile;
}

void copy_update_element(uint32_t source_row, uint32_t source_col,
                         uint32_t output_row, uint32_t output_col) {
  constexpr uint32_t cb_updates = tt::CBIndex::c_2;
  constexpr uint32_t cb_output = tt::CBIndex::c_16;
  volatile tt_l1_ptr Element *source =
      reinterpret_cast<volatile tt_l1_ptr Element *>(get_read_ptr(cb_updates));
  volatile tt_l1_ptr Element *output =
      reinterpret_cast<volatile tt_l1_ptr Element *>(get_write_ptr(cb_output));
  output[tile_element_index(output_row, output_col)] =
      source[tile_element_index(source_row, source_col)];
}

}  // namespace

void kernel_main() {
  uint32_t operand_addr = get_arg_val<uint32_t>(0);
  uint32_t update_addr = get_arg_val<uint32_t>(1);
  uint32_t start_index_addr = get_arg_val<uint32_t>(2);
  uint32_t output_tile_offset = get_arg_val<uint32_t>(3);
  uint32_t output_tile_count = get_arg_val<uint32_t>(4);

  constexpr uint32_t cb_operand = tt::CBIndex::c_0;
  constexpr uint32_t cb_updates = tt::CBIndex::c_2;
  constexpr uint32_t cb_output = tt::CBIndex::c_16;

  const InterleavedAddrGenFast<true> operand = {
      .bank_base_address = operand_addr,
      .page_size = get_tile_size(cb_operand),
      .data_format = get_dataformat(cb_operand),
  };
  const InterleavedAddrGenFast<true> updates = {
      .bank_base_address = update_addr,
      .page_size = get_tile_size(cb_updates),
      .data_format = get_dataformat(cb_updates),
  };

  uint32_t start = clamp_start_index(read_start_index(start_index_addr),
                                     OPERAND_SIZE, UPDATE_SIZE);

  for (uint32_t tile = 0; tile < output_tile_count; ++tile) {
    uint32_t output_tile_id = output_tile_offset + tile;
    read_operand_tile_to_output(operand, output_tile_id, cb_output);

    uint32_t loaded_update_tile = INVALID_TILE;
    uint32_t output_col_base = output_tile_id * TILE_C;
    uint32_t col_count = tile_extent(OPERAND_SIZE, output_col_base, TILE_C);
    for (uint32_t col = 0; col < col_count; ++col) {
      uint32_t output_col = output_col_base + col;
      if (output_col >= start && output_col - start < UPDATE_SIZE) {
        uint32_t update_col = output_col - start;
        ensure_update_tile(updates, update_col / TILE_C, &loaded_update_tile);
        copy_update_element(0, update_col % TILE_C, 0, col);
      }
    }

    if (loaded_update_tile != INVALID_TILE) {
      cb_pop_front(cb_updates, 1);
    }
    cb_push_back(cb_output, 1);
  }
}
