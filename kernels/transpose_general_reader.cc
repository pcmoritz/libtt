#include <cstdint>
namespace {
constexpr uint32_t TILE_R = 32;
constexpr uint32_t TILE_C = 32;
constexpr uint32_t FACE_R = 16;
constexpr uint32_t FACE_C = 16;
constexpr uint32_t INVALID_TILE = 0xffffffffu;
constexpr uint32_t MAX_RANK = TRANSPOSE_GENERAL_MAX_RANK;
constexpr uint32_t RANK = TRANSPOSE_GENERAL_RANK;
constexpr uint32_t INPUT_TILE_ROWS = TRANSPOSE_GENERAL_INPUT_TILE_ROWS;
constexpr uint32_t INPUT_TILES_PER_ROW = TRANSPOSE_GENERAL_INPUT_TILES_PER_ROW;
constexpr uint32_t OUTPUT_ROWS = TRANSPOSE_GENERAL_OUTPUT_ROWS;
constexpr uint32_t OUTPUT_COLS = TRANSPOSE_GENERAL_OUTPUT_COLS;
constexpr uint32_t OUTPUT_TILES_PER_ROW = TRANSPOSE_GENERAL_OUTPUT_TILES_PER_ROW;
constexpr uint32_t OUTPUT_MATRIX_TILES = TRANSPOSE_GENERAL_OUTPUT_MATRIX_TILES;
constexpr uint32_t OUTPUT_SHAPE[MAX_RANK] = TRANSPOSE_GENERAL_OUTPUT_SHAPE;
constexpr uint32_t INPUT_SHAPE[MAX_RANK] = TRANSPOSE_GENERAL_INPUT_SHAPE;
constexpr uint32_t PERMUTATION[MAX_RANK] = TRANSPOSE_GENERAL_PERMUTATION;
using Element = TRANSPOSE_GENERAL_ELEMENT_TYPE;
constexpr uint32_t ARG_INPUT_ADDR = 0;
constexpr uint32_t ARG_OUTPUT_TILE_OFFSET = 1;
constexpr uint32_t ARG_OUTPUT_TILE_COUNT = 2;
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
                       uint32_t *loaded_tile) {
  constexpr uint32_t cb_input = tt::CBIndex::c_0;
  if (requested_tile == *loaded_tile) {
    return;
  }
  if (*loaded_tile != INVALID_TILE) {
    cb_pop_front(cb_input, 1);
  }
  read_input_tile(input, requested_tile, cb_input);
  *loaded_tile = requested_tile;
}
void copy_element(uint32_t source_row, uint32_t source_col, uint32_t output_row,
                  uint32_t output_col) {
  volatile tt_l1_ptr Element *source =
      reinterpret_cast<volatile tt_l1_ptr Element *>(get_read_ptr(tt::CBIndex::c_0));
  volatile tt_l1_ptr Element *output =
      reinterpret_cast<volatile tt_l1_ptr Element *>(get_write_ptr(tt::CBIndex::c_16));
  output[tile_element_index(output_row, output_col)] =
      source[tile_element_index(source_row, source_col)];
}
void decompose_prefix(uint32_t flat, uint32_t *indices) {
  for (int32_t dim = static_cast<int32_t>(RANK) - 3; dim >= 0; --dim) {
    uint32_t extent = OUTPUT_SHAPE[dim];
    indices[dim] = flat % extent;
    flat /= extent;
  }
}
uint32_t tile_id_for_indices(
    const uint32_t *indices,
    uint32_t *row_in_tile,
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
  constexpr uint32_t cb_input = tt::CBIndex::c_0;
  constexpr uint32_t cb_output = tt::CBIndex::c_16;
  const InterleavedAddrGenFast<true> input = {
      .bank_base_address = A(ARG_INPUT_ADDR),
      .page_size = get_tile_size(cb_input),
      .data_format = get_dataformat(cb_input),
  };

  uint32_t output_indices[MAX_RANK];
  uint32_t input_indices[MAX_RANK];
  for (uint32_t tile = 0; tile < A(ARG_OUTPUT_TILE_COUNT); ++tile) {
    uint32_t output_tile_id = A(ARG_OUTPUT_TILE_OFFSET) + tile;
    uint32_t output_prefix = output_tile_id / OUTPUT_MATRIX_TILES;
    uint32_t output_matrix_tile = output_tile_id % OUTPUT_MATRIX_TILES;
    uint32_t output_tile_row = output_matrix_tile / OUTPUT_TILES_PER_ROW;
    uint32_t output_tile_col = output_matrix_tile % OUTPUT_TILES_PER_ROW;
    uint32_t output_row_base = output_tile_row * TILE_R;
    uint32_t output_col_base = output_tile_col * TILE_C;
    uint32_t loaded_input_tile = INVALID_TILE;
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
        ensure_input_tile(input, input_tile, &loaded_input_tile);
        copy_element(input_row, input_col, row, col);
      }
    }
    if (loaded_input_tile != INVALID_TILE) {
      cb_pop_front(cb_input, 1);
    }
    cb_push_back(cb_output, 1);
  }
}
