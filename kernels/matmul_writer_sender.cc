namespace {
void read_source_tile(
    const InterleavedAddrGenFast<true> &input,
    uint32_t tile_id,
    uint32_t cb_source) {
  cb_reserve_back(cb_source, 1);
  noc_async_read_tile(tile_id, input, get_write_ptr(cb_source));
  noc_async_read_barrier();
  cb_push_back(cb_source, 1);
  cb_wait_front(cb_source, 1);
}

void ensure_source_tile(
    const InterleavedAddrGenFast<true> &input,
    uint32_t tile_id,
    uint32_t cb_source,
    uint32_t *loaded_tile) {
  if (*loaded_tile == tile_id) {
    return;
  }
  if (*loaded_tile != INVALID_TILE) {
    cb_pop_front(cb_source, 1);
  }
  read_source_tile(input, tile_id, cb_source);
  *loaded_tile = tile_id;
}

void copy_element_from_source(
    uint32_t cb_source,
    uint32_t dst_addr,
    uint32_t source_row,
    uint32_t source_col,
    uint32_t dst_row,
    uint32_t dst_col) {
  volatile tt_l1_ptr uint16_t *source =
      reinterpret_cast<volatile tt_l1_ptr uint16_t *>(get_read_ptr(cb_source));
  volatile tt_l1_ptr uint16_t *dst = reinterpret_cast<volatile tt_l1_ptr uint16_t *>(dst_addr);
  dst[tile_element_index(dst_row, dst_col)] =
      source[tile_element_index(source_row, source_col)];
}

void fill_generic_element(
    const InterleavedAddrGenFast<true> &input,
    const View &view,
    uint32_t cb_source,
    uint32_t dst_addr,
    uint32_t *loaded_tile,
    uint32_t *indices,
    uint32_t logical_row,
    uint32_t logical_col,
    uint32_t dst_row,
    uint32_t dst_col) {
  decompose_into_dims(logical_row, view.row_dims, view.row_rank, view.shape, indices);
  decompose_into_dims(logical_col, view.col_dims, view.col_rank, view.shape, indices);
  uint32_t source_row = 0;
  uint32_t source_col = 0;
  uint32_t source_tile = tile_id_for_indices(view, indices, &source_row, &source_col);
  ensure_source_tile(input, source_tile, cb_source, loaded_tile);
  copy_element_from_source(cb_source, dst_addr, source_row, source_col, dst_row, dst_col);
}

void fill_generic_tile(
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

  uint32_t indices[MAX_RANK];
  for (uint32_t i = 0; i < MAX_RANK; ++i) {
    indices[i] = 0;
  }
  decompose_into_dims(batch, view.batch_dims, view.batch_rank, view.shape, indices);

  uint32_t loaded_tile = INVALID_TILE;
  if (view.iteration_order == 0) {
    for (uint32_t row = 0; row < TILE_R; ++row) {
      uint32_t logical_row = row_base + row;
      if (logical_row >= view.logical_rows) {
        continue;
      }
      for (uint32_t col = 0; col < TILE_C; ++col) {
        uint32_t logical_col = col_base + col;
        if (logical_col >= view.logical_cols) {
          continue;
        }
        fill_generic_element(
            input,
            view,
            cb_source,
            dst_addr,
            &loaded_tile,
            indices,
            logical_row,
            logical_col,
            row,
            col);
      }
    }
  } else {
    for (uint32_t col = 0; col < TILE_C; ++col) {
      uint32_t logical_col = col_base + col;
      if (logical_col >= view.logical_cols) {
        continue;
      }
      for (uint32_t row = 0; row < TILE_R; ++row) {
        uint32_t logical_row = row_base + row;
        if (logical_row >= view.logical_rows) {
          continue;
        }
        fill_generic_element(
            input,
            view,
            cb_source,
            dst_addr,
            &loaded_tile,
            indices,
            logical_row,
            logical_col,
            row,
            col);
      }
    }
  }
  if (loaded_tile != INVALID_TILE) {
    cb_pop_front(cb_source, 1);
  }
}

void fill_grouped_rows_tile(
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
  uint32_t heads = view.shape[2];
  uint32_t groups = view.shape[3];
  uint32_t batch_index = batch / heads;
  uint32_t head_index = batch - batch_index * heads;
  uint32_t loaded_tile = INVALID_TILE;
  for (uint32_t row = 0; row < TILE_R; ++row) {
    uint32_t logical_row = row_base + row;
    if (logical_row >= view.logical_rows) {
      continue;
    }
    uint32_t query = logical_row / groups;
    uint32_t group = logical_row - query * groups;
    uint32_t source_prefix = (batch_index * view.shape[1] + query) * heads + head_index;
    uint32_t source_tile =
        (source_prefix * view.tile_rows + group / TILE_R) * view.tiles_per_row + col_tile;
    ensure_source_tile(input, source_tile, cb_source, &loaded_tile);
    for (uint32_t col = 0; col < TILE_C; ++col) {
      if (col_base + col >= view.logical_cols) {
        continue;
      }
      copy_element_from_source(cb_source, dst_addr, group % TILE_R, col, row, col);
    }
  }
  if (loaded_tile != INVALID_TILE) {
    cb_pop_front(cb_source, 1);
  }
}

void fill_token_columns_tile(
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
  uint32_t heads = view.shape[2];
  uint32_t batch_index = batch / heads;
  uint32_t head_index = batch - batch_index * heads;
  for (uint32_t col = 0; col < TILE_C; ++col) {
    uint32_t token = col_base + col;
    if (token >= view.logical_cols) {
      continue;
    }
    uint32_t source_prefix = batch_index * view.shape[1] + token;
    uint32_t source_tile =
        (source_prefix * view.tile_rows + head_index / TILE_R) * view.tiles_per_row + row_tile;
    read_source_tile(input, source_tile, cb_source);
    for (uint32_t row = 0; row < TILE_R; ++row) {
      if (row_base + row >= view.logical_rows) {
        continue;
      }
      copy_element_from_source(cb_source, dst_addr, head_index % TILE_R, row, row, col);
    }
    cb_pop_front(cb_source, 1);
  }
}

void fill_grouped_columns_tile(
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
  uint32_t batch_size = view.shape[1];
  uint32_t heads = view.shape[2];
  uint32_t queries = view.shape[3];
  uint32_t batch_index = batch / heads;
  uint32_t head_index = batch - batch_index * heads;
  uint32_t loaded_tile = INVALID_TILE;
  for (uint32_t col = 0; col < TILE_C; ++col) {
    uint32_t logical_col = col_base + col;
    if (logical_col >= view.logical_cols) {
      continue;
    }
    uint32_t group = logical_col / queries;
    uint32_t query = logical_col - group * queries;
    uint32_t source_prefix = (group * batch_size + batch_index) * heads + head_index;
    uint32_t source_tile =
        (source_prefix * view.tile_rows + query / TILE_R) * view.tiles_per_row + row_tile;
    ensure_source_tile(input, source_tile, cb_source, &loaded_tile);
    for (uint32_t row = 0; row < TILE_R; ++row) {
      if (row_base + row >= view.logical_rows) {
        continue;
      }
      copy_element_from_source(cb_source, dst_addr, query % TILE_R, row, row, col);
    }
  }
  if (loaded_tile != INVALID_TILE) {
    cb_pop_front(cb_source, 1);
  }
}

}  // namespace

void kernel_main() {
  constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
  constexpr uint32_t cb_source = tt::CBIndex::c_3;
  constexpr uint32_t cb_out = tt::CBIndex::c_16;
  const uint32_t in1_tile_bytes = get_tile_size(cb_in1);
  const uint32_t out_tile_bytes = get_tile_size(cb_out);
  const uint32_t block_w = A(5);
  const uint32_t block_h = A(6);
  const uint32_t block_tiles = A(7);
  const uint32_t nblocks = A(8);
  const uint32_t i1_nd = A(13);
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
  const uint32_t rhs_batch_stride = A(35);
  const uint32_t output_batch_stride = A(36);
  const View view = load_view(ARG_RHS_VIEW_KIND);
  const View output_view = load_view(ARG_OUTPUT_VIEW_KIND);
  volatile tt_l1_ptr uint32_t *sender_sem = SEM(16);
  volatile tt_l1_ptr uint32_t *recv_sem = SEM(17);
  *recv_sem = VALID;

  const InterleavedAddrGenFast<true> in1_gen = {
      .bank_base_address = A(0),
      .page_size = in1_tile_bytes,
      .data_format = DataFormat::Float16_b,
  };
  const InterleavedAddrGenFast<true> out_gen = {
      .bank_base_address = A(18),
      .page_size = out_tile_bytes,
      .data_format = get_dataformat(cb_out),
  };

  const uint32_t padded_nt = out_next_sb_h / out_sb_h;
  for (uint32_t local_batch = 0; local_batch < local_batch_count; local_batch++) {
    const uint32_t batch = batch_start + local_batch;
    const bool valid_batch = batch < total_batch_count;
    uint32_t cur_block = A(1) + batch * rhs_batch_stride;
    for (uint32_t block = 0; block < nblocks; block++) {
      cb_reserve_back(cb_in1, block_tiles);
      uint32_t l1_addr = get_write_ptr(cb_in1);
      uint32_t start_addr = l1_addr;
      uint32_t row = cur_block;
      uint32_t block_bytes = 0;
      if (!valid_batch) {
        for (uint32_t tile = 0; tile < block_tiles; ++tile) {
          zero_tile_at(l1_addr, in1_tile_bytes);
          l1_addr += in1_tile_bytes;
          block_bytes += in1_tile_bytes;
        }
      } else if (view.kind == VIEW_CONTIGUOUS) {
        for (uint32_t h = 0; h < block_h; h++) {
          uint32_t tile_id = row;
          for (uint32_t w = 0; w < block_w; w++) {
            if (A(1) + w < logical_nt) {
              noc_async_read_tile(tile_id, in1_gen, l1_addr);
            }
            l1_addr += in1_tile_bytes;
            tile_id += A(2);
            block_bytes += in1_tile_bytes;
          }
          row += A(3);
        }
        noc_async_read_barrier();
      } else {
        uint32_t canonical_base = cur_block - batch * rhs_batch_stride;
        for (uint32_t h = 0; h < block_h; h++) {
          for (uint32_t w = 0; w < block_w; w++) {
            uint32_t canonical_tile = canonical_base + h * A(3) + w;
            uint32_t canonical_row_tile = canonical_tile / A(3);
            uint32_t canonical_col_tile = canonical_tile - canonical_row_tile * A(3);
            if (view.kind == VIEW_TRANSPOSE_LAST_TWO) {
              const uint32_t row_base = canonical_row_tile * TILE_R;
              const uint32_t col_base = canonical_col_tile * TILE_C;
              if (row_base < view.logical_rows && col_base < view.logical_cols) {
                const uint32_t source_tile =
                    batch * view.tile_rows * view.tiles_per_row +
                    canonical_col_tile * view.tiles_per_row +
                    canonical_row_tile;
                noc_async_read_tile(source_tile, in1_gen, l1_addr);
              }
            } else if (view.kind == VIEW_GROUPED_ROWS) {
              fill_grouped_rows_tile(
                  in1_gen,
                  view,
                  batch,
                  canonical_row_tile,
                  canonical_col_tile,
                  l1_addr,
                  in1_tile_bytes,
                  cb_source);
            } else if (view.kind == VIEW_TOKEN_COLUMNS) {
              fill_token_columns_tile(
                  in1_gen,
                  view,
                  batch,
                  canonical_row_tile,
                  canonical_col_tile,
                  l1_addr,
                  in1_tile_bytes,
                  cb_source);
            } else if (view.kind == VIEW_GROUPED_COLUMNS) {
              fill_grouped_columns_tile(
                  in1_gen,
                  view,
                  batch,
                  canonical_row_tile,
                  canonical_col_tile,
                  l1_addr,
                  in1_tile_bytes,
                  cb_source);
            } else {
              fill_generic_tile(
                  in1_gen,
                  view,
                  batch,
                  canonical_row_tile,
                  canonical_col_tile,
                  l1_addr,
                  in1_tile_bytes,
                  cb_source);
            }
            l1_addr += in1_tile_bytes;
            block_bytes += in1_tile_bytes;
          }
        }
        if (view.kind == VIEW_TRANSPOSE_LAST_TWO) {
          noc_async_read_barrier();
        }
      }
      cur_block += A(4);

      noc_semaphore_wait(sender_sem, i1_nd);
      noc_semaphore_set(sender_sem, 0);
      if (i1_nd > 0) {
        uint64_t ma = get_noc_multicast_addr(A(9), A(10), A(11), A(12), start_addr);
        noc_async_write_multicast(start_addr, ma, block_bytes, i1_nd);
        noc_async_writes_flushed();
        noc_semaphore_set_multicast(
            get_semaphore(A(17)),
            get_noc_multicast_addr(A(9), A(10), A(11), A(12), get_semaphore(A(17))),
            i1_nd);
      }
      cb_push_back(cb_in1, block_tiles);
    }

    uint32_t sbh_start = out_start;
    for (uint32_t sbh = 0; sbh < out_num_sb_h; sbh++) {
      uint32_t sbw_start = sbh_start;
      for (uint32_t sbw = 0; sbw < out_num_sb_w; sbw++) {
        cb_wait_front(cb_out, out_sb_tiles);
        uint32_t l1_addr = get_read_ptr(cb_out);
        uint32_t row_start = sbw_start;
        if (valid_batch && output_rows_are_physical_tiles(output_view) &&
            out_col_offset == 0 && out_sb_w == logical_nt) {
          for (uint32_t h = 0; h < out_sb_h; h++) {
            const uint32_t out_row = row_start / padded_nt;
            if (out_row < logical_mt) {
              write_output_row_physical_tiles(
                  out_gen,
                  output_view,
                  tt::CBIndex::c_4,
                  batch,
                  out_row,
                  0,
                  out_sb_w,
                  l1_addr,
                  out_tile_bytes,
                  out_tile_bytes / (TILE_R * TILE_C));
            }
            l1_addr += out_sb_w * out_tile_bytes;
            row_start += out_stride_h;
          }
        } else {
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
                    logical_nt,
                    l1_addr,
                    out_tile_bytes / (TILE_R * TILE_C));
              }
              l1_addr += out_tile_bytes;
              tile_id += out_stride_w;
            }
            row_start += out_stride_h;
          }
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, out_sb_tiles);
        sbw_start += out_next_sb_w;
      }
      sbh_start += out_next_sb_h;
    }
  }
}
