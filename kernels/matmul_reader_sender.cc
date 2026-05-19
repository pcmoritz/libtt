#include <cstdint>

#define A(n) get_arg_val<uint32_t>(n)
#define SEM(n) reinterpret_cast<volatile tt_l1_ptr uint32_t *>(get_semaphore(A(n)))

namespace {
constexpr uint32_t ARG_VIEW_KIND = 28;
constexpr uint32_t ARG_VIEW_SHAPE = ARG_VIEW_KIND + 9;
constexpr uint32_t ARG_VIEW_BATCH_DIMS = ARG_VIEW_SHAPE + MAX_RANK;
constexpr uint32_t ARG_VIEW_ROW_DIMS = ARG_VIEW_BATCH_DIMS + MAX_RANK;
constexpr uint32_t ARG_VIEW_COL_DIMS = ARG_VIEW_ROW_DIMS + MAX_RANK;

void load_array(uint32_t base, uint32_t *target) {
  for (uint32_t i = 0; i < MAX_RANK; ++i) {
    target[i] = A(base + i);
  }
}

View load_view() {
  View view;
  view.kind = A(ARG_VIEW_KIND);
  view.rank = A(ARG_VIEW_KIND + 1);
  view.batch_rank = A(ARG_VIEW_KIND + 2);
  view.row_rank = A(ARG_VIEW_KIND + 3);
  view.col_rank = A(ARG_VIEW_KIND + 4);
  view.logical_rows = A(ARG_VIEW_KIND + 5);
  view.logical_cols = A(ARG_VIEW_KIND + 6);
  view.tile_rows = A(ARG_VIEW_KIND + 7);
  view.tiles_per_row = A(ARG_VIEW_KIND + 8);
  load_array(ARG_VIEW_SHAPE, view.shape);
  load_array(ARG_VIEW_BATCH_DIMS, view.batch_dims);
  load_array(ARG_VIEW_ROW_DIMS, view.row_dims);
  load_array(ARG_VIEW_COL_DIMS, view.col_dims);
  return view;
}

void fill_transposed_tile(
    const InterleavedAddrGenFast<true> &input,
    const View &view,
    uint32_t batch,
    uint32_t row_tile,
    uint32_t col_tile,
    uint32_t dst_addr,
    uint32_t tile_bytes,
    uint32_t cb_source) {
  zero_tile_at(dst_addr, tile_bytes);
  uint32_t row_base = row_tile * TILE_R;
  uint32_t col_base = col_tile * TILE_C;
  if (row_base >= view.logical_rows || col_base >= view.logical_cols) {
    return;
  }
  uint32_t source_tile =
      batch * view.tile_rows * view.tiles_per_row + col_tile * view.tiles_per_row + row_tile;
  read_source_tile(input, source_tile, cb_source);
  for (uint32_t row = 0; row < TILE_R; ++row) {
    if (row_base + row >= view.logical_rows) {
      continue;
    }
    for (uint32_t col = 0; col < TILE_C; ++col) {
      if (col_base + col >= view.logical_cols) {
        continue;
      }
      copy_element_from_source(cb_source, dst_addr, col, row, row, col);
    }
  }
  cb_pop_front(cb_source, 1);
}

}  // namespace

void kernel_main() {
  constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
  constexpr uint32_t cb_source = tt::CBIndex::c_2;
  const uint32_t tile_bytes = get_tile_size(cb_in0);
  const uint32_t block_w = A(5);
  const uint32_t block_h = A(6);
  const uint32_t block_tiles = A(7);
  const uint32_t nblocks = A(8);
  const uint32_t w_nd = A(13);
  const uint32_t e_nd = A(18);
  const uint32_t logical_mt = A(23);
  const uint32_t local_batch_count = A(24);
  const uint32_t batch_start = A(25);
  const uint32_t total_batch_count = A(26);
  const uint32_t batch_stride = A(27);
  const View view = load_view();
  volatile tt_l1_ptr uint32_t *sender_sem = SEM(21);
  volatile tt_l1_ptr uint32_t *recv_sem = SEM(22);
  *recv_sem = VALID;

  const InterleavedAddrGenFast<true> in0_gen = {
      .bank_base_address = A(0),
      .page_size = tile_bytes,
      .data_format = DataFormat::Float16_b,
  };
  for (uint32_t local_batch = 0; local_batch < local_batch_count; local_batch++) {
    const uint32_t batch = batch_start + local_batch;
    const bool valid_batch = batch < total_batch_count;
    uint32_t cur_block = A(1) + batch * batch_stride;
    for (uint32_t block = 0; block < nblocks; block++) {
      cb_reserve_back(cb_in0, block_tiles);
      uint32_t l1_addr = get_write_ptr(cb_in0);
      uint32_t start_addr = l1_addr;
      uint32_t row = cur_block;
      uint32_t row_tile = (row - batch * batch_stride) / A(3);
      uint32_t block_bytes = 0;
      if (!valid_batch) {
        for (uint32_t tile = 0; tile < block_tiles; ++tile) {
          zero_tile_at(l1_addr, tile_bytes);
          l1_addr += tile_bytes;
          block_bytes += tile_bytes;
        }
      } else if (view.kind == VIEW_CONTIGUOUS) {
        for (uint32_t h = 0; h < block_h; h++) {
          uint32_t tile_id = row;
          for (uint32_t w = 0; w < block_w; w++) {
            if (row_tile < logical_mt) {
              noc_async_read_tile(tile_id, in0_gen, l1_addr);
            }
            l1_addr += tile_bytes;
            tile_id += A(2);
            block_bytes += tile_bytes;
          }
          row += A(3);
          row_tile++;
        }
        noc_async_read_barrier();
      } else {
        uint32_t canonical_base = cur_block - batch * batch_stride;
        for (uint32_t h = 0; h < block_h; h++) {
          for (uint32_t w = 0; w < block_w; w++) {
            uint32_t canonical_tile = canonical_base + h * A(3) + w;
            uint32_t canonical_row_tile = canonical_tile / A(3);
            uint32_t canonical_col_tile = canonical_tile - canonical_row_tile * A(3);
            if (view.kind == VIEW_TRANSPOSE_LAST_TWO) {
              fill_transposed_tile(
                  in0_gen,
                  view,
                  batch,
                  canonical_row_tile,
                  canonical_col_tile,
                  l1_addr,
                  tile_bytes,
                  cb_source);
            } else if (view.kind == VIEW_GROUPED_ROWS) {
              fill_grouped_rows_tile(
                  in0_gen,
                  view,
                  batch,
                  canonical_row_tile,
                  canonical_col_tile,
                  l1_addr,
                  tile_bytes,
                  cb_source);
            } else if (view.kind == VIEW_TOKEN_COLUMNS) {
              fill_token_columns_tile(
                  in0_gen,
                  view,
                  batch,
                  canonical_row_tile,
                  canonical_col_tile,
                  l1_addr,
                  tile_bytes,
                  cb_source);
            } else if (view.kind == VIEW_GROUPED_COLUMNS) {
              fill_grouped_columns_tile(
                  in0_gen,
                  view,
                  batch,
                  canonical_row_tile,
                  canonical_col_tile,
                  l1_addr,
                  tile_bytes,
                  cb_source);
            } else {
              fill_generic_tile(
                  in0_gen,
                  view,
                  batch,
                  canonical_row_tile,
                  canonical_col_tile,
                  l1_addr,
                  tile_bytes,
                  cb_source);
            }
            l1_addr += tile_bytes;
            block_bytes += tile_bytes;
          }
        }
      }
      cur_block += A(4);

      noc_semaphore_wait(sender_sem, w_nd + e_nd);
      noc_semaphore_set(sender_sem, 0);
      if (w_nd > 0) {
        uint64_t wa = get_noc_multicast_addr(A(9), A(10), A(11), A(12), start_addr);
        noc_async_write_multicast(start_addr, wa, block_bytes, w_nd);
        noc_async_writes_flushed();
        noc_semaphore_set_multicast(
            get_semaphore(A(22)),
            get_noc_multicast_addr(A(9), A(10), A(11), A(12), get_semaphore(A(22))),
            w_nd);
      }
      if (e_nd > 0) {
        uint64_t ea = get_noc_multicast_addr(A(14), A(15), A(16), A(17), start_addr);
        noc_async_write_multicast(start_addr, ea, block_bytes, e_nd);
        noc_async_writes_flushed();
        noc_semaphore_set_multicast(
            get_semaphore(A(22)),
            get_noc_multicast_addr(A(14), A(15), A(16), A(17), get_semaphore(A(22))),
            e_nd);
      }
      cb_push_back(cb_in0, block_tiles);
    }
  }
}
