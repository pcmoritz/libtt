#include <cstdint>

namespace {

constexpr uint32_t TILE_R = 32;
constexpr uint32_t TILE_C = 32;
constexpr uint32_t FACE_R = 16;
constexpr uint32_t FACE_C = 16;
constexpr uint32_t BF16_NEG_INF = 0xff80;

uint32_t tile_element_index(uint32_t row, uint32_t col) {
  uint32_t face_row = row / FACE_R;
  uint32_t face_col = col / FACE_C;
  uint32_t row_in_face = row % FACE_R;
  uint32_t col_in_face = col % FACE_C;
  return ((face_row * 2 + face_col) * FACE_R * FACE_C) + row_in_face * FACE_C + col_in_face;
}

uint16_t value_key(uint16_t bits) {
  return static_cast<uint16_t>(
      (bits & 0x8000u) != 0 ? ~bits : (bits ^ 0x8000u));
}

bool candidate_before(uint16_t lhs_key, uint32_t lhs_index, uint16_t rhs_key,
                      uint32_t rhs_index) {
  return lhs_key > rhs_key || (lhs_key == rhs_key && lhs_index < rhs_index);
}

}  // namespace

void kernel_main() {
  uint32_t input_addr = get_arg_val<uint32_t>(0);
  uint32_t partial_pairs_addr = get_arg_val<uint32_t>(1);
  uint32_t tile_offset = get_arg_val<uint32_t>(2);
  uint32_t tile_count = get_arg_val<uint32_t>(3);
  uint32_t logical_len = get_arg_val<uint32_t>(4);
  uint32_t partial_tile_id = get_arg_val<uint32_t>(5);

  constexpr uint32_t cb_input = tt::CBIndex::c_0;
  constexpr uint32_t cb_pairs = tt::CBIndex::c_16;

  const InterleavedAddrGenFast<true> input = {
      .bank_base_address = input_addr,
      .page_size = get_tile_size(cb_input),
      .data_format = get_dataformat(cb_input),
  };
  const InterleavedAddrGenFast<true> partial_pairs = {
      .bank_base_address = partial_pairs_addr,
      .page_size = get_tile_size(cb_pairs),
      .data_format = get_dataformat(cb_pairs),
  };

  bool have_best = false;
  uint16_t best_key = 0;
  uint32_t best_value = BF16_NEG_INF;
  uint32_t best_index = 0xffffffffu;

  for (uint32_t local_tile = 0; local_tile < tile_count; ++local_tile) {
    uint32_t tile_id = tile_offset + local_tile;
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

    volatile tt_l1_ptr uint16_t *tile =
        reinterpret_cast<volatile tt_l1_ptr uint16_t *>(input_l1_addr);
    for (uint32_t col = 0; col < valid_cols; ++col) {
      uint32_t index = base_index + col;
      uint16_t value = tile[tile_element_index(0, col)];
      uint16_t key = value_key(value);
      if (!have_best || candidate_before(key, index, best_key, best_index)) {
        have_best = true;
        best_key = key;
        best_value = value;
        best_index = index;
      }
    }
    cb_push_back(cb_input, 1);
    cb_pop_front(cb_input, 1);
  }

  cb_reserve_back(cb_pairs, 1);
  volatile tt_l1_ptr uint32_t *pair_ptr =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(get_write_ptr(cb_pairs));
  pair_ptr[0] = best_value;
  pair_ptr[1] = best_index;
  noc_async_write_tile(partial_tile_id, partial_pairs, get_write_ptr(cb_pairs));
  noc_async_write_barrier();
  cb_push_back(cb_pairs, 1);
  cb_pop_front(cb_pairs, 1);
}
