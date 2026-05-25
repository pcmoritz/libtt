#include <cstdint>

namespace {

constexpr uint32_t ARG_LHS_VIEW_KIND = 12;
constexpr uint32_t ARG_RHS_VIEW_KIND_F32 = ARG_LHS_VIEW_KIND + VIEW_ARG_COUNT;
constexpr uint32_t ARG_OUTPUT_VIEW_KIND_F32 = ARG_RHS_VIEW_KIND_F32 + VIEW_ARG_COUNT;

uint32_t tile_extent_f32(uint32_t logical_dim, uint32_t base, uint32_t tile_dim) {
  if (base >= logical_dim) {
    return 0;
  }
  uint32_t remaining = logical_dim - base;
  return remaining < tile_dim ? remaining : tile_dim;
}

void fill_f32_tile(uint32_t cb, float value) {
  volatile tt_l1_ptr float *ptr =
      reinterpret_cast<volatile tt_l1_ptr float *>(get_write_ptr(cb));
  uint32_t elements = get_tile_size(cb) / sizeof(float);
  for (uint32_t i = 0; i < elements; ++i) {
    ptr[i] = value;
  }
}

float read_view_f32(const InterleavedAddrGenFast<true> &input, const View &view,
                    uint32_t batch, uint32_t logical_row, uint32_t logical_col,
                    uint32_t cb, uint32_t *loaded_tile) {
  uint32_t indices[MAX_RANK] = {};
  decompose_into_dims(batch, view.batch_dims, view.batch_rank, view.shape, indices);
  decompose_into_dims(logical_row, view.row_dims, view.row_rank, view.shape, indices);
  decompose_into_dims(logical_col, view.col_dims, view.col_rank, view.shape, indices);
  uint32_t source_row = 0;
  uint32_t source_col = 0;
  uint32_t source_tile = tile_id_for_indices(view, indices, &source_row, &source_col);
  ensure_source_tile(input, source_tile, cb, loaded_tile);
  volatile tt_l1_ptr float *ptr =
      reinterpret_cast<volatile tt_l1_ptr float *>(get_read_ptr(cb));
  return ptr[tile_element_index(source_row, source_col)];
}

void write_output_f32(uint32_t row, uint32_t col, float value) {
  constexpr uint32_t cb_output = tt::CBIndex::c_16;
  volatile tt_l1_ptr float *ptr =
      reinterpret_cast<volatile tt_l1_ptr float *>(get_write_ptr(cb_output));
  ptr[tile_element_index(row, col)] = value;
}

}  // namespace

void kernel_main() {
  uint32_t lhs_addr = get_arg_val<uint32_t>(0);
  uint32_t rhs_addr = get_arg_val<uint32_t>(1);
  uint32_t output_addr = get_arg_val<uint32_t>(2);
  uint32_t output_tile_offset = get_arg_val<uint32_t>(3);
  uint32_t output_tile_count = get_arg_val<uint32_t>(4);
  uint32_t batch_count = get_arg_val<uint32_t>(5);
  uint32_t m = get_arg_val<uint32_t>(6);
  uint32_t k = get_arg_val<uint32_t>(7);
  uint32_t n = get_arg_val<uint32_t>(8);
  uint32_t output_tile_rows = get_arg_val<uint32_t>(9);
  uint32_t output_tiles_per_row = get_arg_val<uint32_t>(10);
  uint32_t output_tiles_per_batch = get_arg_val<uint32_t>(11);

  constexpr uint32_t cb_lhs = tt::CBIndex::c_0;
  constexpr uint32_t cb_rhs = tt::CBIndex::c_1;
  constexpr uint32_t cb_scratch = tt::CBIndex::c_4;
  constexpr uint32_t cb_output = tt::CBIndex::c_16;
  const InterleavedAddrGenFast<true> lhs = {
      .bank_base_address = lhs_addr,
      .page_size = get_tile_size(cb_lhs),
      .data_format = get_dataformat(cb_lhs),
  };
  const InterleavedAddrGenFast<true> rhs = {
      .bank_base_address = rhs_addr,
      .page_size = get_tile_size(cb_rhs),
      .data_format = get_dataformat(cb_rhs),
  };
  const InterleavedAddrGenFast<true> output = {
      .bank_base_address = output_addr,
      .page_size = get_tile_size(cb_output),
      .data_format = get_dataformat(cb_output),
  };
  const View lhs_view = load_view(ARG_LHS_VIEW_KIND);
  const View rhs_view = load_view(ARG_RHS_VIEW_KIND_F32);
  const View output_view = load_view(ARG_OUTPUT_VIEW_KIND_F32);

  for (uint32_t tile = 0; tile < output_tile_count; ++tile) {
    uint32_t output_tile_id = output_tile_offset + tile;
    uint32_t batch = output_tile_id / output_tiles_per_batch;
    if (batch >= batch_count) {
      continue;
    }
    uint32_t matrix_tile = output_tile_id - batch * output_tiles_per_batch;
    uint32_t row_tile = matrix_tile / output_tiles_per_row;
    uint32_t col_tile = matrix_tile - row_tile * output_tiles_per_row;
    uint32_t row_base = row_tile * TILE_R;
    uint32_t col_base = col_tile * TILE_C;
    uint32_t row_count = tile_extent_f32(m, row_base, TILE_R);
    uint32_t col_count = tile_extent_f32(n, col_base, TILE_C);

    cb_reserve_back(cb_output, 1);
    fill_f32_tile(cb_output, 0.0f);

    uint32_t loaded_lhs_tile = INVALID_TILE;
    uint32_t loaded_rhs_tile = INVALID_TILE;
    for (uint32_t row = 0; row < row_count; ++row) {
      uint32_t logical_row = row_base + row;
      for (uint32_t col = 0; col < col_count; ++col) {
        uint32_t logical_col = col_base + col;
        float acc = 0.0f;
        for (uint32_t kk = 0; kk < k; ++kk) {
          float lhs_value =
              read_view_f32(lhs, lhs_view, batch, logical_row, kk, cb_lhs, &loaded_lhs_tile);
          float rhs_value =
              read_view_f32(rhs, rhs_view, batch, kk, logical_col, cb_rhs, &loaded_rhs_tile);
          acc += lhs_value * rhs_value;
        }
        write_output_f32(row, col, acc);
      }
    }

    if (loaded_lhs_tile != INVALID_TILE) {
      cb_pop_front(cb_lhs, 1);
    }
    if (loaded_rhs_tile != INVALID_TILE) {
      cb_pop_front(cb_rhs, 1);
    }

    write_output_tile(output, output_view, batch, row_tile, col_tile,
                      output_tiles_per_batch, output_tiles_per_row,
                      get_write_ptr(cb_output), sizeof(float), cb_scratch);
    noc_async_write_barrier();
    cb_push_back(cb_output, 1);
    cb_pop_front(cb_output, 1);
  }

  (void)output_tile_rows;
}
