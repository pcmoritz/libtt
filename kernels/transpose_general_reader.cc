#include <cstdint>
namespace {
constexpr uint32_t TILE_R = 32;
constexpr uint32_t TILE_C = 32;
constexpr uint32_t FACE_R = 16;
constexpr uint32_t FACE_C = 16;
constexpr uint32_t INVALID_TILE = 0xffffffffu;
constexpr uint32_t MAX_RANK = TRANSPOSE_GENERAL_MAX_RANK;
using Element = TRANSPOSE_GENERAL_ELEMENT_TYPE;
constexpr uint32_t ARG_INPUT_ADDR = 0;
constexpr uint32_t ARG_OUTPUT_TILE_OFFSET = 1;
constexpr uint32_t ARG_OUTPUT_TILE_COUNT = 2;
constexpr uint32_t ARG_RANK = 3;
constexpr uint32_t ARG_INPUT_TILE_ROWS = 4;
constexpr uint32_t ARG_INPUT_TILES_PER_ROW = 5;
constexpr uint32_t ARG_OUTPUT_ROWS = 6;
constexpr uint32_t ARG_OUTPUT_COLS = 7;
constexpr uint32_t ARG_OUTPUT_TILES_PER_ROW = 8;
constexpr uint32_t ARG_OUTPUT_MATRIX_TILES = 9;
constexpr uint32_t ARG_OUTPUT_SHAPE = 10;
constexpr uint32_t ARG_INPUT_SHAPE = ARG_OUTPUT_SHAPE + MAX_RANK;
constexpr uint32_t ARG_PERMUTATION = ARG_INPUT_SHAPE + MAX_RANK;
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
  constexpr uint32_t cb_input = tt::CBIndex::c_0;
  constexpr uint32_t cb_output = tt::CBIndex::c_16;
  const InterleavedAddrGenFast<true> input = {
      .bank_base_address = A(ARG_INPUT_ADDR),
      .page_size = get_tile_size(cb_input),
      .data_format = get_dataformat(cb_input),
  };
  uint32_t rank = A(ARG_RANK);
  uint32_t output_shape[MAX_RANK];
  uint32_t input_shape[MAX_RANK];
  uint32_t permutation[MAX_RANK];
  uint32_t output_indices[MAX_RANK];
  uint32_t input_indices[MAX_RANK];
  load_array(ARG_OUTPUT_SHAPE, output_shape);
  load_array(ARG_INPUT_SHAPE, input_shape);
  load_array(ARG_PERMUTATION, permutation);
  for (uint32_t tile = 0; tile < A(ARG_OUTPUT_TILE_COUNT); ++tile) {
    uint32_t output_tile_id = A(ARG_OUTPUT_TILE_OFFSET) + tile;
    uint32_t output_prefix = output_tile_id / A(ARG_OUTPUT_MATRIX_TILES);
    uint32_t output_matrix_tile = output_tile_id % A(ARG_OUTPUT_MATRIX_TILES);
    uint32_t output_tile_row = output_matrix_tile / A(ARG_OUTPUT_TILES_PER_ROW);
    uint32_t output_tile_col = output_matrix_tile % A(ARG_OUTPUT_TILES_PER_ROW);
    uint32_t output_row_base = output_tile_row * TILE_R;
    uint32_t output_col_base = output_tile_col * TILE_C;
    uint32_t loaded_input_tile = INVALID_TILE;
    cb_reserve_back(cb_output, 1);
    zero_tile(cb_output);
    for (uint32_t i = 0; i < MAX_RANK; ++i) {
      output_indices[i] = 0;
      input_indices[i] = 0;
    }
    decompose_prefix(output_prefix, output_shape, rank, output_indices);
    for (uint32_t row = 0; row < TILE_R; ++row) {
      uint32_t output_row = output_row_base + row;
      if (output_row >= A(ARG_OUTPUT_ROWS)) {
        continue;
      }
      output_indices[rank - 2] = output_row;
      for (uint32_t col = 0; col < TILE_C; ++col) {
        uint32_t output_col = output_col_base + col;
        if (output_col >= A(ARG_OUTPUT_COLS)) {
          continue;
        }
        output_indices[rank - 1] = output_col;
        for (uint32_t dim = 0; dim < rank; ++dim) {
          input_indices[permutation[dim]] = output_indices[dim];
        }
        uint32_t input_row = 0;
        uint32_t input_col = 0;
        uint32_t input_tile = tile_id_for_indices(
            input_indices, input_shape, rank, A(ARG_INPUT_TILE_ROWS),
            A(ARG_INPUT_TILES_PER_ROW), &input_row, &input_col);
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
