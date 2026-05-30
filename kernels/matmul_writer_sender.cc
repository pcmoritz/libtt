void kernel_main() {
  constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
  constexpr uint32_t cb_source = tt::CBIndex::c_3;
  const uint32_t in1_tile_bytes = get_tile_size(cb_in1);
  const uint32_t block_w = A(5);
  const uint32_t block_h = A(6);
  const uint32_t block_tiles = A(7);
  const uint32_t nblocks = A(8);
  const uint32_t i1_nd = A(13);
  const uint32_t logical_nt = A(30);
  const uint32_t local_batch_count = A(32);
  const uint32_t batch_start = A(33);
  const uint32_t total_batch_count = A(34);
  const uint32_t rhs_batch_stride = A(35);
  const View view = load_view(ARG_RHS_VIEW_KIND);
  const OutputDrain output_drain = load_output_drain();
  volatile tt_l1_ptr uint32_t *sender_sem = SEM(16);
  volatile tt_l1_ptr uint32_t *recv_sem = SEM(17);
  *recv_sem = VALID;

  const InterleavedAddrGenFast<true> in1_gen = {
      .bank_base_address = A(0),
      .page_size = in1_tile_bytes,
      .data_format = get_dataformat(cb_in1),
  };

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
            if (view.kind == VIEW_TILE_TRANSPOSE) {
              fill_tile_transpose_tile(
                  in1_gen,
                  view,
                  batch,
                  canonical_row_tile,
                  canonical_col_tile,
                  l1_addr,
                  in1_tile_bytes,
                  cb_source);
            } else if (view.kind == VIEW_TILED_INDEX_MAP) {
              fill_tiled_index_map_tile(
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

    drain_output_blocks(output_drain, batch, valid_batch);
  }
}
