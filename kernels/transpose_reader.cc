#include <cstdint>

namespace {

constexpr uint32_t TILE_R = 32;
constexpr uint32_t TILE_C = 32;
constexpr uint32_t FACE_R = 16;
constexpr uint32_t FACE_C = 16;
constexpr uint32_t INVALID_TILE = 0xffffffffu;
constexpr uint32_t MAX_RANK = TRANSPOSE_MAX_RANK;
constexpr uint32_t RANK = TRANSPOSE_RANK;
constexpr uint32_t OUTPUT_SHAPE[MAX_RANK] = TRANSPOSE_OUTPUT_SHAPE;
constexpr uint32_t INPUT_SHAPE[MAX_RANK] = TRANSPOSE_INPUT_SHAPE;
constexpr uint32_t PERMUTATION[MAX_RANK] = TRANSPOSE_PERMUTATION;
constexpr uint32_t OUTPUT_ROWS = OUTPUT_SHAPE[RANK - 2];
constexpr uint32_t OUTPUT_COLS = OUTPUT_SHAPE[RANK - 1];
constexpr uint32_t INPUT_TILE_ROWS = (INPUT_SHAPE[RANK - 2] + TILE_R - 1) / TILE_R;
constexpr uint32_t INPUT_TILES_PER_ROW = (INPUT_SHAPE[RANK - 1] + TILE_C - 1) / TILE_C;
constexpr uint32_t OUTPUT_TILES_PER_ROW = (OUTPUT_COLS + TILE_C - 1) / TILE_C;
constexpr uint32_t OUTPUT_MATRIX_TILES =
    ((OUTPUT_ROWS + TILE_R - 1) / TILE_R) * OUTPUT_TILES_PER_ROW;
constexpr bool RANK3_LAST_DIM_TO_FRONT =
    RANK == 3 && PERMUTATION[0] == 2 && PERMUTATION[1] == 0 &&
    PERMUTATION[2] == 1 && INPUT_SHAPE[2] == 1 && OUTPUT_SHAPE[0] == 1;
using Element = TRANSPOSE_ELEMENT_TYPE;

uint32_t tile_element_index(uint32_t row, uint32_t col) {
  uint32_t face_row = row / FACE_R;
  uint32_t face_col = col / FACE_C;
  uint32_t row_in_face = row % FACE_R;
  uint32_t col_in_face = col % FACE_C;
  return ((face_row * 2 + face_col) * FACE_R * FACE_C) + row_in_face * FACE_C + col_in_face;
}

uint32_t min_u32(uint32_t lhs, uint32_t rhs) { return lhs < rhs ? lhs : rhs; }

void zero_tile(uint32_t cb) {
  volatile tt_l1_ptr uint32_t *ptr =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(get_write_ptr(cb));
  uint32_t words = get_tile_size(cb) / sizeof(uint32_t);
  for (uint32_t i = 0; i < words; ++i) {
    ptr[i] = 0;
  }
}

uint32_t read_input_tile(const InterleavedAddrGenFast<true> &input, uint32_t tile_id,
                         uint32_t cb) {
  cb_reserve_back(cb, 1);
  uint32_t l1_addr = get_write_ptr(cb);
  noc_async_read_tile(tile_id, input, l1_addr);
  noc_async_read_barrier();
  cb_push_back(cb, 1);
  return l1_addr;
}

void ensure_input_tile(const InterleavedAddrGenFast<true> &input, uint32_t requested_tile,
                       uint32_t *loaded_tile, uint32_t *loaded_l1_addr) {
  constexpr uint32_t cb_input = tt::CBIndex::c_0;
  if (requested_tile == *loaded_tile) {
    return;
  }
  if (*loaded_tile != INVALID_TILE) {
    cb_pop_front(cb_input, 1);
  }
  *loaded_l1_addr = read_input_tile(input, requested_tile, cb_input);
  *loaded_tile = requested_tile;
}

void copy_element(uint32_t input_l1_addr, uint32_t cb_output, uint32_t source_row,
                  uint32_t source_col, uint32_t output_row, uint32_t output_col) {
  volatile tt_l1_ptr Element *source =
      reinterpret_cast<volatile tt_l1_ptr Element *>(input_l1_addr);
  volatile tt_l1_ptr Element *output =
      reinterpret_cast<volatile tt_l1_ptr Element *>(get_write_ptr(cb_output));
  output[tile_element_index(output_row, output_col)] =
      source[tile_element_index(source_row, source_col)];
}

void copy_input_col0_to_output_row(uint32_t input_l1_addr, uint32_t cb_output,
                                   uint32_t output_row, uint32_t valid_cols) {
  volatile tt_l1_ptr Element *source =
      reinterpret_cast<volatile tt_l1_ptr Element *>(input_l1_addr);
  volatile tt_l1_ptr Element *output =
      reinterpret_cast<volatile tt_l1_ptr Element *>(get_write_ptr(cb_output));
  for (uint32_t col = 0; col < valid_cols; ++col) {
    output[tile_element_index(output_row, col)] =
        source[tile_element_index(col, 0)];
  }
}

void decompose_prefix(uint32_t flat, uint32_t *indices) {
  for (int32_t dim = static_cast<int32_t>(RANK) - 3; dim >= 0; --dim) {
    uint32_t extent = OUTPUT_SHAPE[dim];
    indices[dim] = flat % extent;
    flat /= extent;
  }
}

uint32_t tile_id_for_indices(const uint32_t *indices, uint32_t *row_in_tile,
                             uint32_t *col_in_tile) {
  uint32_t prefix = 0;
  for (uint32_t dim = 0; dim + 2 < RANK; ++dim) {
    prefix = prefix * INPUT_SHAPE[dim] + indices[dim];
  }
  uint32_t row = indices[RANK - 2];
  uint32_t col = indices[RANK - 1];
  *row_in_tile = row % TILE_R;
  *col_in_tile = col % TILE_C;
  return (prefix * INPUT_TILE_ROWS + row / TILE_R) * INPUT_TILES_PER_ROW + col / TILE_C;
}

}  // namespace

void kernel_main() {
  uint32_t input_addr = get_arg_val<uint32_t>(0);
  uint32_t output_addr = get_arg_val<uint32_t>(1);
  uint32_t output_tile_offset = get_arg_val<uint32_t>(2);
  uint32_t output_tile_count = get_arg_val<uint32_t>(3);

  constexpr uint32_t cb_input = tt::CBIndex::c_0;
  constexpr uint32_t cb_output = tt::CBIndex::c_1;
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

  if constexpr (RANK3_LAST_DIM_TO_FRONT) {
    for (uint32_t tile = 0; tile < output_tile_count; ++tile) {
      uint32_t output_tile_id = output_tile_offset + tile;
      uint32_t output_tile_row = output_tile_id / OUTPUT_TILES_PER_ROW;
      uint32_t output_tile_col = output_tile_id % OUTPUT_TILES_PER_ROW;
      uint32_t output_row_base = output_tile_row * TILE_R;
      uint32_t output_col_base = output_tile_col * TILE_C;
      uint32_t valid_rows =
          output_row_base < OUTPUT_ROWS ? min_u32(TILE_R, OUTPUT_ROWS - output_row_base) : 0;
      uint32_t valid_cols =
          output_col_base < OUTPUT_COLS ? min_u32(TILE_C, OUTPUT_COLS - output_col_base) : 0;

      cb_reserve_back(cb_output, 1);
      zero_tile(cb_output);

      cb_reserve_back(cb_input, valid_rows);
      uint32_t input_l1_base = get_write_ptr(cb_input);
      uint32_t input_tile_bytes = get_tile_size(cb_input);
      for (uint32_t row = 0; row < valid_rows; ++row) {
        uint32_t input_head = output_row_base + row;
        uint32_t input_tile = input_head * INPUT_TILE_ROWS + output_tile_col;
        uint32_t input_l1_addr = input_l1_base + row * input_tile_bytes;
        noc_async_read_tile(input_tile, input, input_l1_addr);
      }
      noc_async_read_barrier();
      cb_push_back(cb_input, valid_rows);
      for (uint32_t row = 0; row < valid_rows; ++row) {
        uint32_t input_l1_addr = input_l1_base + row * input_tile_bytes;
        copy_input_col0_to_output_row(input_l1_addr, cb_output, row, valid_cols);
      }
      cb_pop_front(cb_input, valid_rows);

      noc_async_write_tile(output_tile_id, output, get_write_ptr(cb_output));
      noc_async_write_barrier();
      cb_push_back(cb_output, 1);
      cb_pop_front(cb_output, 1);
    }
    return;
  }

  uint32_t output_indices[MAX_RANK];
  uint32_t input_indices[MAX_RANK];
  for (uint32_t tile = 0; tile < output_tile_count; ++tile) {
    uint32_t output_tile_id = output_tile_offset + tile;
    uint32_t output_prefix = output_tile_id / OUTPUT_MATRIX_TILES;
    uint32_t output_matrix_tile = output_tile_id % OUTPUT_MATRIX_TILES;
    uint32_t output_tile_row = output_matrix_tile / OUTPUT_TILES_PER_ROW;
    uint32_t output_tile_col = output_matrix_tile % OUTPUT_TILES_PER_ROW;
    uint32_t output_row_base = output_tile_row * TILE_R;
    uint32_t output_col_base = output_tile_col * TILE_C;
    uint32_t loaded_input_tile = INVALID_TILE;
    uint32_t loaded_input_l1_addr = 0;

    cb_reserve_back(cb_output, 1);
    zero_tile(cb_output);

    decompose_prefix(output_prefix, output_indices);
    for (uint32_t row = 0; row < TILE_R; ++row) {
      uint32_t output_row = output_row_base + row;
      if (output_row >= OUTPUT_ROWS) {
        continue;
      }
      output_indices[RANK - 2] = output_row;
      for (uint32_t col = 0; col < TILE_C; ++col) {
        uint32_t output_col = output_col_base + col;
        if (output_col >= OUTPUT_COLS) {
          continue;
        }

        output_indices[RANK - 1] = output_col;
        for (uint32_t dim = 0; dim < RANK; ++dim) {
          input_indices[PERMUTATION[dim]] = output_indices[dim];
        }
        uint32_t input_row = 0;
        uint32_t input_col = 0;
        uint32_t input_tile = tile_id_for_indices(input_indices, &input_row, &input_col);
        ensure_input_tile(input, input_tile, &loaded_input_tile,
                          &loaded_input_l1_addr);
        copy_element(loaded_input_l1_addr, cb_output, input_row, input_col, row, col);
      }
    }

    if (loaded_input_tile != INVALID_TILE) {
      cb_pop_front(cb_input, 1);
    }
    noc_async_write_tile(output_tile_id, output, get_write_ptr(cb_output));
    noc_async_write_barrier();
    cb_push_back(cb_output, 1);
    cb_pop_front(cb_output, 1);
  }
}
