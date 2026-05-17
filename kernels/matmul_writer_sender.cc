#include <cstdint>

#define A(n) get_arg_val<uint32_t>(n)
#define SEM(n) reinterpret_cast<volatile tt_l1_ptr uint32_t *>(get_semaphore(A(n)))

void kernel_main() {
  constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
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

  uint32_t cur_block = A(1);
  for (uint32_t block = 0; block < nblocks; block++) {
    cb_reserve_back(cb_in1, block_tiles);
    uint32_t l1_addr = get_write_ptr(cb_in1);
    uint32_t start_addr = l1_addr;
    uint32_t row = cur_block;
    uint32_t block_bytes = 0;
    for (uint32_t h = 0; h < block_h; h++) {
      uint32_t tile_id = row;
      for (uint32_t w = 0; w < block_w; w++) {
        // Padded columns only feed padded outputs, so avoid the out-of-bounds DRAM read.
        if (A(1) + w < logical_nt) {
          noc_async_read_tile(tile_id, in1_gen, l1_addr);
        }
        l1_addr += in1_tile_bytes;
        tile_id += A(2);
        block_bytes += in1_tile_bytes;
      }
      row += A(3);
    }
    cur_block += A(4);
    noc_async_read_barrier();

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

  const uint32_t padded_nt = out_next_sb_h / out_sb_h;
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
          if (out_row < logical_mt && out_col < logical_nt) {
            noc_async_write_tile(out_row * logical_nt + out_col, out_gen, l1_addr);
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
