#include <cstdint>

namespace {

constexpr uint32_t TILE_R = 32;
constexpr uint32_t TILE_C = 32;
constexpr uint32_t FACE_R = 16;
constexpr uint32_t FACE_C = 16;
constexpr uint32_t MAX_K = 32;

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

uint32_t load_value_bits(uint32_t tile_l1_addr, uint32_t element) {
#if defined(TOPK_DTYPE_BFLOAT16)
  volatile tt_l1_ptr uint16_t *ptr =
      reinterpret_cast<volatile tt_l1_ptr uint16_t *>(tile_l1_addr);
  return ptr[element];
#elif defined(TOPK_DTYPE_FLOAT32)
  volatile tt_l1_ptr uint32_t *ptr =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(tile_l1_addr);
  return ptr[element];
#endif
}

uint32_t value_key(uint32_t value_bits) {
#if defined(TOPK_DTYPE_BFLOAT16)
  return ordered_float_key(value_bits << 16);
#elif defined(TOPK_DTYPE_FLOAT32)
  return ordered_float_key(value_bits);
#endif
}

bool candidate_before(uint32_t lhs_key, uint32_t lhs_index, uint32_t rhs_key, uint32_t rhs_index) {
  return lhs_key > rhs_key || (lhs_key == rhs_key && lhs_index < rhs_index);
}

void insert_candidate(uint32_t key, uint32_t value_bits, uint32_t index, uint32_t k,
                      uint32_t best_keys[MAX_K], uint32_t best_values[MAX_K],
                      uint32_t best_indices[MAX_K], uint32_t& count) {
  if (count == k && !candidate_before(key, index, best_keys[k - 1], best_indices[k - 1])) {
    return;
  }

  uint32_t pos = count < k ? count : k - 1;
  if (count < k) {
    ++count;
  }
  while (pos > 0 && candidate_before(key, index, best_keys[pos - 1], best_indices[pos - 1])) {
    best_keys[pos] = best_keys[pos - 1];
    best_values[pos] = best_values[pos - 1];
    best_indices[pos] = best_indices[pos - 1];
    --pos;
  }
  best_keys[pos] = key;
  best_values[pos] = value_bits;
  best_indices[pos] = index;
}

void store_value(uint32_t tile_l1_addr, uint32_t element, uint32_t value_bits) {
#if defined(TOPK_DTYPE_BFLOAT16)
  volatile tt_l1_ptr uint16_t *ptr =
      reinterpret_cast<volatile tt_l1_ptr uint16_t *>(tile_l1_addr);
  ptr[element] = static_cast<uint16_t>(value_bits);
#elif defined(TOPK_DTYPE_FLOAT32)
  volatile tt_l1_ptr uint32_t *ptr =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(tile_l1_addr);
  ptr[element] = value_bits;
#endif
}

}  // namespace

void kernel_main() {
  uint32_t input_addr = get_arg_val<uint32_t>(0);
  uint32_t values_addr = get_arg_val<uint32_t>(1);
  uint32_t indices_addr = get_arg_val<uint32_t>(2);
  uint32_t logical_len = get_arg_val<uint32_t>(3);
  uint32_t input_tiles = get_arg_val<uint32_t>(4);
  uint32_t k = get_arg_val<uint32_t>(5);

  constexpr uint32_t cb_input = tt::CBIndex::c_0;
  constexpr uint32_t cb_values = tt::CBIndex::c_16;
  constexpr uint32_t cb_indices = tt::CBIndex::c_17;

  const InterleavedAddrGenFast<true> input = {
      .bank_base_address = input_addr,
      .page_size = get_tile_size(cb_input),
      .data_format = get_dataformat(cb_input),
  };
  const InterleavedAddrGenFast<true> values = {
      .bank_base_address = values_addr,
      .page_size = get_tile_size(cb_values),
      .data_format = get_dataformat(cb_values),
  };
  const InterleavedAddrGenFast<true> indices = {
      .bank_base_address = indices_addr,
      .page_size = get_tile_size(cb_indices),
      .data_format = get_dataformat(cb_indices),
  };

  uint32_t best_keys[MAX_K];
  uint32_t best_values[MAX_K];
  uint32_t best_indices[MAX_K];
  uint32_t count = 0;

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
      uint32_t element = tile_element_index(0, col);
      uint32_t bits = load_value_bits(input_l1_addr, element);
      insert_candidate(value_key(bits), bits, base_index + col, k, best_keys, best_values,
                       best_indices, count);
    }
    cb_push_back(cb_input, 1);
    cb_pop_front(cb_input, 1);
  }

  cb_reserve_back(cb_values, 1);
  zero_tile(cb_values);
  uint32_t values_l1_addr = get_write_ptr(cb_values);
  for (uint32_t i = 0; i < count; ++i) {
    store_value(values_l1_addr, tile_element_index(0, i), best_values[i]);
  }
  noc_async_write_tile(0, values, values_l1_addr);
  noc_async_write_barrier();
  cb_push_back(cb_values, 1);
  cb_pop_front(cb_values, 1);

  cb_reserve_back(cb_indices, 1);
  zero_tile(cb_indices);
  volatile tt_l1_ptr int32_t *indices_ptr =
      reinterpret_cast<volatile tt_l1_ptr int32_t *>(get_write_ptr(cb_indices));
  for (uint32_t i = 0; i < count; ++i) {
    indices_ptr[tile_element_index(0, i)] = static_cast<int32_t>(best_indices[i]);
  }
  noc_async_write_tile(0, indices, get_write_ptr(cb_indices));
  noc_async_write_barrier();
  cb_push_back(cb_indices, 1);
  cb_pop_front(cb_indices, 1);
}
