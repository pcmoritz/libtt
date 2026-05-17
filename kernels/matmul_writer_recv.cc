#include <cstdint>

#define A(n) get_arg_val<uint32_t>(n)
#define SEM(n) reinterpret_cast<volatile tt_l1_ptr uint32_t *>(get_semaphore(A(n)))

namespace {
constexpr uint32_t TILE_R = 32;
constexpr uint32_t TILE_C = 32;
constexpr uint32_t FACE_R = 16;
constexpr uint32_t FACE_C = 16;
constexpr uint32_t MAX_RANK = 8;
constexpr uint32_t VIEW_CONTIGUOUS = 0;
constexpr uint32_t VIEW_ARG_COUNT = 10 + 4 * MAX_RANK;
constexpr uint32_t ARG_RHS_VIEW_KIND = 37;
constexpr uint32_t ARG_OUTPUT_VIEW_KIND = ARG_RHS_VIEW_KIND + VIEW_ARG_COUNT;

struct View {
  uint32_t kind;
  uint32_t rank;
  uint32_t batch_rank;
  uint32_t row_rank;
  uint32_t col_rank;
  uint32_t logical_rows;
  uint32_t logical_cols;
  uint32_t tile_rows;
  uint32_t tiles_per_row;
  uint32_t iteration_order;
  uint32_t shape[MAX_RANK];
  uint32_t batch_dims[MAX_RANK];
  uint32_t row_dims[MAX_RANK];
  uint32_t col_dims[MAX_RANK];
};

uint32_t tile_element_index(uint32_t row, uint32_t col) {
  uint32_t face_row = row / FACE_R;
  uint32_t face_col = col / FACE_C;
  uint32_t row_in_face = row % FACE_R;
  uint32_t col_in_face = col % FACE_C;
  return ((face_row * 2 + face_col) * FACE_R * FACE_C) + row_in_face * FACE_C + col_in_face;
}

void load_array(uint32_t base, uint32_t *target) {
  for (uint32_t i = 0; i < MAX_RANK; ++i) {
    target[i] = A(base + i);
  }
}

View load_view(uint32_t arg_view_kind) {
  const uint32_t arg_view_shape = arg_view_kind + 10;
  const uint32_t arg_view_batch_dims = arg_view_shape + MAX_RANK;
  const uint32_t arg_view_row_dims = arg_view_batch_dims + MAX_RANK;
  const uint32_t arg_view_col_dims = arg_view_row_dims + MAX_RANK;
  View view;
  view.kind = A(arg_view_kind);
  view.rank = A(arg_view_kind + 1);
  view.batch_rank = A(arg_view_kind + 2);
  view.row_rank = A(arg_view_kind + 3);
  view.col_rank = A(arg_view_kind + 4);
  view.logical_rows = A(arg_view_kind + 5);
  view.logical_cols = A(arg_view_kind + 6);
  view.tile_rows = A(arg_view_kind + 7);
  view.tiles_per_row = A(arg_view_kind + 8);
  view.iteration_order = A(arg_view_kind + 9);
  load_array(arg_view_shape, view.shape);
  load_array(arg_view_batch_dims, view.batch_dims);
  load_array(arg_view_row_dims, view.row_dims);
  load_array(arg_view_col_dims, view.col_dims);
  return view;
}

void decompose_into_dims(
    uint32_t flat,
    const uint32_t *dims,
    uint32_t dim_count,
    const uint32_t *shape,
    uint32_t *indices) {
  for (int32_t i = static_cast<int32_t>(dim_count) - 1; i >= 0; --i) {
    uint32_t dim = dims[i];
    uint32_t extent = shape[dim];
    indices[dim] = flat % extent;
    flat /= extent;
  }
}

uint32_t tile_id_for_indices(
    const View &view,
    const uint32_t *indices,
    uint32_t *row_in_tile,
    uint32_t *col_in_tile) {
  uint32_t prefix = 0;
  for (uint32_t dim = 0; dim + 2 < view.rank; ++dim) {
    prefix = prefix * view.shape[dim] + indices[dim];
  }
  uint32_t row = indices[view.rank - 2];
  uint32_t col = indices[view.rank - 1];
  *row_in_tile = row % TILE_R;
  *col_in_tile = col % TILE_C;
  return (prefix * view.tile_rows + row / TILE_R) * view.tiles_per_row + col / TILE_C;
}

uint32_t output_tile_for_element(
    const View &view,
    uint32_t batch,
    uint32_t logical_row,
    uint32_t logical_col,
    uint32_t *row_in_tile,
    uint32_t *col_in_tile) {
  uint32_t indices[MAX_RANK];
  for (uint32_t i = 0; i < MAX_RANK; ++i) {
    indices[i] = 0;
  }
  decompose_into_dims(batch, view.batch_dims, view.batch_rank, view.shape, indices);
  decompose_into_dims(logical_row, view.row_dims, view.row_rank, view.shape, indices);
  decompose_into_dims(logical_col, view.col_dims, view.col_rank, view.shape, indices);
  return tile_id_for_indices(view, indices, row_in_tile, col_in_tile);
}

void write_output_tile(
    const InterleavedAddrGenFast<true> &out_gen,
    const View &output_view,
    uint32_t batch,
    uint32_t canonical_row_tile,
    uint32_t canonical_col_tile,
    uint32_t output_batch_stride,
    uint32_t logical_mt,
    uint32_t logical_nt,
    uint32_t src_l1_addr,
    uint32_t element_bytes) {
  if (output_view.kind == VIEW_CONTIGUOUS) {
    noc_async_write_tile(
        batch * output_batch_stride + canonical_row_tile * logical_nt + canonical_col_tile,
        out_gen,
        src_l1_addr);
    return;
  }

  const uint32_t row_base = canonical_row_tile * TILE_R;
  const uint32_t col_base = canonical_col_tile * TILE_C;
  for (uint32_t row = 0; row < TILE_R; ++row) {
    const uint32_t logical_row = row_base + row;
    if (logical_row >= output_view.logical_rows) {
      continue;
    }
    uint32_t col = 0;
    while (col < TILE_C) {
      const uint32_t logical_col = col_base + col;
      if (logical_col >= output_view.logical_cols) {
        break;
      }
      uint32_t dst_row = 0;
      uint32_t dst_col = 0;
      const uint32_t dst_tile = output_tile_for_element(
          output_view, batch, logical_row, logical_col, &dst_row, &dst_col);
      const uint32_t src_offset = tile_element_index(row, col) * element_bytes;
      const uint32_t dst_offset = tile_element_index(dst_row, dst_col) * element_bytes;
      uint32_t run = 1;
      while (col + run < TILE_C && col_base + col + run < output_view.logical_cols) {
        uint32_t next_dst_row = 0;
        uint32_t next_dst_col = 0;
        const uint32_t next_dst_tile = output_tile_for_element(
            output_view,
            batch,
            logical_row,
            col_base + col + run,
            &next_dst_row,
            &next_dst_col);
        const uint32_t next_src_offset =
            tile_element_index(row, col + run) * element_bytes;
        const uint32_t next_dst_offset =
            tile_element_index(next_dst_row, next_dst_col) * element_bytes;
        if (next_dst_tile != dst_tile ||
            next_src_offset != src_offset + run * element_bytes ||
            next_dst_offset != dst_offset + run * element_bytes) {
          break;
        }
        ++run;
      }
      noc_async_write(
          src_l1_addr + src_offset,
          get_noc_addr(dst_tile, out_gen, dst_offset),
          run * element_bytes);
      col += run;
    }
  }
}
}  // namespace

void kernel_main() {
  constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
  constexpr uint32_t cb_out = tt::CBIndex::c_16;
  const uint32_t out_tile_bytes = get_tile_size(cb_out);
  const uint32_t block_tiles = A(7);
  const uint32_t nblocks = A(8);
  const uint32_t out_start = A(19);
  const uint32_t out_stride_w = A(20);
  const uint32_t out_stride_h = A(21);
  const uint32_t out_next_sb_w = A(22);
  const uint32_t out_next_sb_h = A(23);
  const uint32_t out_sb_w = A(24);
  const uint32_t out_sb_h = A(25);
  const uint32_t out_sb_tiles = A(26);
  const uint32_t out_num_sb_w = A(27);
  const uint32_t out_num_sb_h = A(28);
  const uint32_t logical_mt = A(29);
  const uint32_t logical_nt = A(30);
  const uint32_t out_col_offset = A(31);
  const uint32_t local_batch_count = A(32);
  const uint32_t batch_start = A(33);
  const uint32_t total_batch_count = A(34);
  const uint32_t output_batch_stride = A(36);
  const View output_view = load_view(ARG_OUTPUT_VIEW_KIND);
  volatile tt_l1_ptr uint32_t *recv_sem = SEM(17);

  const InterleavedAddrGenFast<true> out_gen = {
      .bank_base_address = A(18),
      .page_size = out_tile_bytes,
      .data_format = get_dataformat(cb_out),
  };

  const uint32_t padded_nt = out_next_sb_h / out_sb_h;
  for (uint32_t local_batch = 0; local_batch < local_batch_count; local_batch++) {
    const uint32_t batch = batch_start + local_batch;
    const bool valid_batch = batch < total_batch_count;
    for (uint32_t block = 0; block < nblocks; block++) {
      cb_reserve_back(cb_in1, block_tiles);
      noc_semaphore_set(recv_sem, INVALID);
      noc_semaphore_inc(get_noc_addr(A(14), A(15), get_semaphore(A(16))), 1);
      noc_semaphore_wait(recv_sem, VALID);
      cb_push_back(cb_in1, block_tiles);
    }

    uint32_t sbh_start = out_start;
    for (uint32_t sbh = 0; sbh < out_num_sb_h; sbh++) {
      uint32_t sbw_start = sbh_start;
      for (uint32_t sbw = 0; sbw < out_num_sb_w; sbw++) {
        cb_wait_front(cb_out, out_sb_tiles);
        uint32_t l1_addr = get_read_ptr(cb_out);
        uint32_t row_start = sbw_start;
        for (uint32_t h = 0; h < out_sb_h; h++) {
          uint32_t tile_id = row_start;
          for (uint32_t w = 0; w < out_sb_w; w++) {
            const uint32_t out_row = tile_id / padded_nt;
            const uint32_t out_col = out_col_offset + tile_id - out_row * padded_nt;
            if (valid_batch && out_row < logical_mt && out_col < logical_nt) {
              write_output_tile(
                  out_gen,
                  output_view,
                  batch,
                  out_row,
                  out_col,
                  output_batch_stride,
                  logical_mt,
                  logical_nt,
                  l1_addr,
                  out_tile_bytes / (TILE_R * TILE_C));
            }
            l1_addr += out_tile_bytes;
            tile_id += out_stride_w;
          }
          row_start += out_stride_h;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, out_sb_tiles);
        sbw_start += out_next_sb_w;
      }
      sbh_start += out_next_sb_h;
    }
  }
}
