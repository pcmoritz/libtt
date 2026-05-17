#include <cstdint>

namespace {

constexpr uint32_t TILE_R = 32;
constexpr uint32_t TILE_C = 32;
constexpr uint32_t FACE_R = 16;
constexpr uint32_t FACE_C = 16;
constexpr uint32_t INVALID_TILE = 0xffffffffu;
constexpr uint32_t MAX_RANK = DOT_GENERAL_MAX_RANK;

constexpr uint32_t ARG_LHS_ADDR = 0;
constexpr uint32_t ARG_RHS_ADDR = 1;
constexpr uint32_t ARG_OUTPUT_TILE_OFFSET = 2;
constexpr uint32_t ARG_OUTPUT_TILE_COUNT = 3;
constexpr uint32_t ARG_OUTPUT_RANK = 4;
constexpr uint32_t ARG_LHS_RANK = 5;
constexpr uint32_t ARG_RHS_RANK = 6;
constexpr uint32_t ARG_BATCH_COUNT = 7;
constexpr uint32_t ARG_LHS_OUTER_COUNT = 8;
constexpr uint32_t ARG_RHS_OUTER_COUNT = 9;
constexpr uint32_t ARG_CONTRACT_COUNT = 10;
constexpr uint32_t ARG_CONTRACT_VOLUME = 11;
constexpr uint32_t ARG_LHS_TILE_ROWS = 12;
constexpr uint32_t ARG_LHS_TILES_PER_ROW = 13;
constexpr uint32_t ARG_RHS_TILE_ROWS = 14;
constexpr uint32_t ARG_RHS_TILES_PER_ROW = 15;
constexpr uint32_t ARG_OUTPUT_ROWS = 16;
constexpr uint32_t ARG_OUTPUT_COLS = 17;
constexpr uint32_t ARG_OUTPUT_TILE_ROWS = 18;
constexpr uint32_t ARG_OUTPUT_TILES_PER_ROW = 19;
constexpr uint32_t ARG_OUTPUT_MATRIX_TILES = 20;
constexpr uint32_t ARG_OUTPUT_SHAPE = 21;
constexpr uint32_t ARG_LHS_SHAPE = ARG_OUTPUT_SHAPE + MAX_RANK;
constexpr uint32_t ARG_RHS_SHAPE = ARG_LHS_SHAPE + MAX_RANK;
constexpr uint32_t ARG_LHS_BATCH_DIMS = ARG_RHS_SHAPE + MAX_RANK;
constexpr uint32_t ARG_RHS_BATCH_DIMS = ARG_LHS_BATCH_DIMS + MAX_RANK;
constexpr uint32_t ARG_LHS_CONTRACT_DIMS = ARG_RHS_BATCH_DIMS + MAX_RANK;
constexpr uint32_t ARG_RHS_CONTRACT_DIMS = ARG_LHS_CONTRACT_DIMS + MAX_RANK;
constexpr uint32_t ARG_LHS_OUTER_DIMS = ARG_RHS_CONTRACT_DIMS + MAX_RANK;
constexpr uint32_t ARG_RHS_OUTER_DIMS = ARG_LHS_OUTER_DIMS + MAX_RANK;

uint32_t A(uint32_t index) { return get_arg_val<uint32_t>(index); }

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

void read_input_tile(const InterleavedAddrGenFast<true> &input, uint32_t tile_id, uint32_t cb) {
  cb_reserve_back(cb, 1);
  noc_async_read_tile(tile_id, input, get_write_ptr(cb));
  noc_async_read_barrier();
  cb_push_back(cb, 1);
  cb_wait_front(cb, 1);
}

void ensure_input_tile(const InterleavedAddrGenFast<true> &input, uint32_t requested_tile,
                       uint32_t cb, uint32_t *loaded_tile) {
  if (requested_tile == *loaded_tile) {
    return;
  }
  if (*loaded_tile != INVALID_TILE) {
    cb_pop_front(cb, 1);
  }
  read_input_tile(input, requested_tile, cb);
  *loaded_tile = requested_tile;
}

float bits_to_float(uint32_t bits) {
  union {
    uint32_t u;
    float f;
  } value;
  value.u = bits;
  return value.f;
}

uint32_t float_to_bits(float value) {
  union {
    float f;
    uint32_t u;
  } bits;
  bits.f = value;
  return bits.u;
}

float bf16_to_float(uint16_t value) { return bits_to_float(static_cast<uint32_t>(value) << 16); }

uint16_t float_to_bf16(float value) {
  uint32_t bits = float_to_bits(value);
  uint32_t lsb = (bits >> 16) & 1;
  return static_cast<uint16_t>((bits + 0x7fffu + lsb) >> 16);
}

float read_lhs_value(uint32_t row, uint32_t col) {
  uint32_t index = tile_element_index(row, col);
#if DOT_GENERAL_LHS_BF16
  volatile tt_l1_ptr uint16_t *ptr =
      reinterpret_cast<volatile tt_l1_ptr uint16_t *>(get_read_ptr(tt::CBIndex::c_0));
  return bf16_to_float(ptr[index]);
#else
  volatile tt_l1_ptr uint32_t *ptr =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(get_read_ptr(tt::CBIndex::c_0));
  return bits_to_float(ptr[index]);
#endif
}

float read_rhs_value(uint32_t row, uint32_t col) {
  uint32_t index = tile_element_index(row, col);
#if DOT_GENERAL_RHS_BF16
  volatile tt_l1_ptr uint16_t *ptr =
      reinterpret_cast<volatile tt_l1_ptr uint16_t *>(get_read_ptr(tt::CBIndex::c_1));
  return bf16_to_float(ptr[index]);
#else
  volatile tt_l1_ptr uint32_t *ptr =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(get_read_ptr(tt::CBIndex::c_1));
  return bits_to_float(ptr[index]);
#endif
}

void write_output_value(float value, uint32_t row, uint32_t col) {
  uint32_t index = tile_element_index(row, col);
#if DOT_GENERAL_OUTPUT_BF16
  volatile tt_l1_ptr uint16_t *ptr =
      reinterpret_cast<volatile tt_l1_ptr uint16_t *>(get_write_ptr(tt::CBIndex::c_16));
  ptr[index] = float_to_bf16(value);
#else
  volatile tt_l1_ptr uint32_t *ptr =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(get_write_ptr(tt::CBIndex::c_16));
  ptr[index] = float_to_bits(value);
#endif
}

void load_array(uint32_t base, uint32_t *target) {
  for (uint32_t i = 0; i < MAX_RANK; ++i) {
    target[i] = A(base + i);
  }
}

void decompose_prefix(uint32_t flat, const uint32_t *shape, uint32_t rank,
                      uint32_t *indices) {
  for (int32_t dim = static_cast<int32_t>(rank) - 3; dim >= 0; --dim) {
    uint32_t extent = shape[dim];
    indices[dim] = flat % extent;
    flat /= extent;
  }
}

void decompose_contract(uint32_t flat, const uint32_t *lhs_shape,
                        const uint32_t *lhs_contract_dims, uint32_t contract_count,
                        uint32_t *contract_indices) {
  for (int32_t index = static_cast<int32_t>(contract_count) - 1; index >= 0; --index) {
    uint32_t extent = lhs_shape[lhs_contract_dims[index]];
    contract_indices[index] = flat % extent;
    flat /= extent;
  }
}

uint32_t tile_id_for_indices(const uint32_t *indices, const uint32_t *shape, uint32_t rank,
                             uint32_t tile_rows, uint32_t tiles_per_row,
                             uint32_t *row_in_tile, uint32_t *col_in_tile) {
  uint32_t prefix = 0;
  for (uint32_t dim = 0; dim + 2 < rank; ++dim) {
    prefix = prefix * shape[dim] + indices[dim];
  }
  uint32_t row = indices[rank - 2];
  uint32_t col = indices[rank - 1];
  *row_in_tile = row % TILE_R;
  *col_in_tile = col % TILE_C;
  return (prefix * tile_rows + row / TILE_R) * tiles_per_row + col / TILE_C;
}

}  // namespace

void kernel_main() {
  constexpr uint32_t cb_lhs = tt::CBIndex::c_0;
  constexpr uint32_t cb_rhs = tt::CBIndex::c_1;
  constexpr uint32_t cb_output = tt::CBIndex::c_16;

  const InterleavedAddrGenFast<true> lhs = {
      .bank_base_address = A(ARG_LHS_ADDR),
      .page_size = get_tile_size(cb_lhs),
      .data_format = get_dataformat(cb_lhs),
  };
  const InterleavedAddrGenFast<true> rhs = {
      .bank_base_address = A(ARG_RHS_ADDR),
      .page_size = get_tile_size(cb_rhs),
      .data_format = get_dataformat(cb_rhs),
  };

  uint32_t output_shape[MAX_RANK];
  uint32_t lhs_shape[MAX_RANK];
  uint32_t rhs_shape[MAX_RANK];
  uint32_t lhs_batch_dims[MAX_RANK];
  uint32_t rhs_batch_dims[MAX_RANK];
  uint32_t lhs_contract_dims[MAX_RANK];
  uint32_t rhs_contract_dims[MAX_RANK];
  uint32_t lhs_outer_dims[MAX_RANK];
  uint32_t rhs_outer_dims[MAX_RANK];
  uint32_t output_indices[MAX_RANK];
  uint32_t lhs_indices[MAX_RANK];
  uint32_t rhs_indices[MAX_RANK];
  uint32_t contract_indices[MAX_RANK];

  load_array(ARG_OUTPUT_SHAPE, output_shape);
  load_array(ARG_LHS_SHAPE, lhs_shape);
  load_array(ARG_RHS_SHAPE, rhs_shape);
  load_array(ARG_LHS_BATCH_DIMS, lhs_batch_dims);
  load_array(ARG_RHS_BATCH_DIMS, rhs_batch_dims);
  load_array(ARG_LHS_CONTRACT_DIMS, lhs_contract_dims);
  load_array(ARG_RHS_CONTRACT_DIMS, rhs_contract_dims);
  load_array(ARG_LHS_OUTER_DIMS, lhs_outer_dims);
  load_array(ARG_RHS_OUTER_DIMS, rhs_outer_dims);

  uint32_t output_rank = A(ARG_OUTPUT_RANK);
  uint32_t lhs_rank = A(ARG_LHS_RANK);
  uint32_t rhs_rank = A(ARG_RHS_RANK);
  uint32_t batch_count = A(ARG_BATCH_COUNT);
  uint32_t lhs_outer_count = A(ARG_LHS_OUTER_COUNT);
  uint32_t rhs_outer_count = A(ARG_RHS_OUTER_COUNT);
  uint32_t contract_count = A(ARG_CONTRACT_COUNT);
  uint32_t contract_volume = A(ARG_CONTRACT_VOLUME);
  uint32_t output_matrix_tiles = A(ARG_OUTPUT_MATRIX_TILES);

  for (uint32_t tile = 0; tile < A(ARG_OUTPUT_TILE_COUNT); ++tile) {
    uint32_t output_tile_id = A(ARG_OUTPUT_TILE_OFFSET) + tile;
    uint32_t output_prefix = output_tile_id / output_matrix_tiles;
    uint32_t output_matrix_tile = output_tile_id % output_matrix_tiles;
    uint32_t output_tile_row = output_matrix_tile / A(ARG_OUTPUT_TILES_PER_ROW);
    uint32_t output_tile_col = output_matrix_tile % A(ARG_OUTPUT_TILES_PER_ROW);
    uint32_t output_row_base = output_tile_row * TILE_R;
    uint32_t output_col_base = output_tile_col * TILE_C;
    uint32_t loaded_lhs_tile = INVALID_TILE;
    uint32_t loaded_rhs_tile = INVALID_TILE;

    cb_reserve_back(cb_output, 1);
    zero_tile(cb_output);

    for (uint32_t i = 0; i < MAX_RANK; ++i) {
      output_indices[i] = 0;
    }
    decompose_prefix(output_prefix, output_shape, output_rank, output_indices);

    for (uint32_t row = 0; row < TILE_R; ++row) {
      uint32_t output_row = output_row_base + row;
      if (output_row >= A(ARG_OUTPUT_ROWS)) {
        continue;
      }
      output_indices[output_rank - 2] = output_row;
      for (uint32_t col = 0; col < TILE_C; ++col) {
        uint32_t output_col = output_col_base + col;
        if (output_col >= A(ARG_OUTPUT_COLS)) {
          continue;
        }
        output_indices[output_rank - 1] = output_col;

        for (uint32_t i = 0; i < MAX_RANK; ++i) {
          lhs_indices[i] = 0;
          rhs_indices[i] = 0;
        }
        for (uint32_t i = 0; i < batch_count; ++i) {
          lhs_indices[lhs_batch_dims[i]] = output_indices[i];
          rhs_indices[rhs_batch_dims[i]] = output_indices[i];
        }
        uint32_t output_dim = batch_count;
        for (uint32_t i = 0; i < lhs_outer_count; ++i) {
          lhs_indices[lhs_outer_dims[i]] = output_indices[output_dim++];
        }
        for (uint32_t i = 0; i < rhs_outer_count; ++i) {
          rhs_indices[rhs_outer_dims[i]] = output_indices[output_dim++];
        }

        float acc = 0.0f;
        for (uint32_t contract = 0; contract < contract_volume; ++contract) {
          decompose_contract(contract, lhs_shape, lhs_contract_dims, contract_count,
                             contract_indices);
          for (uint32_t i = 0; i < contract_count; ++i) {
            lhs_indices[lhs_contract_dims[i]] = contract_indices[i];
            rhs_indices[rhs_contract_dims[i]] = contract_indices[i];
          }

          uint32_t lhs_row = 0;
          uint32_t lhs_col = 0;
          uint32_t rhs_row = 0;
          uint32_t rhs_col = 0;
          uint32_t lhs_tile = tile_id_for_indices(
              lhs_indices, lhs_shape, lhs_rank, A(ARG_LHS_TILE_ROWS), A(ARG_LHS_TILES_PER_ROW),
              &lhs_row, &lhs_col);
          uint32_t rhs_tile = tile_id_for_indices(
              rhs_indices, rhs_shape, rhs_rank, A(ARG_RHS_TILE_ROWS), A(ARG_RHS_TILES_PER_ROW),
              &rhs_row, &rhs_col);
          ensure_input_tile(lhs, lhs_tile, cb_lhs, &loaded_lhs_tile);
          ensure_input_tile(rhs, rhs_tile, cb_rhs, &loaded_rhs_tile);
          acc += read_lhs_value(lhs_row, lhs_col) * read_rhs_value(rhs_row, rhs_col);
        }

        write_output_value(acc, row, col);
      }
    }

    if (loaded_lhs_tile != INVALID_TILE) {
      cb_pop_front(cb_lhs, 1);
    }
    if (loaded_rhs_tile != INVALID_TILE) {
      cb_pop_front(cb_rhs, 1);
    }
    cb_push_back(cb_output, 1);
  }
}
