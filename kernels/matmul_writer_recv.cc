#include <cstdint>

#define A(n) get_arg_val<uint32_t>(n)
#define SEM(n) reinterpret_cast<volatile tt_l1_ptr uint32_t *>(get_semaphore(A(n)))

void kernel_main() {
  constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
  constexpr uint32_t cb_out = tt::CBIndex::c_16;
  const uint32_t tile_bytes = get_tile_size(cb_in1);
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
  volatile tt_l1_ptr uint32_t *recv_sem = SEM(17);

  const InterleavedAddrGenFast<true> out_gen = {
      .bank_base_address = A(18),
      .page_size = tile_bytes,
      .data_format = DataFormat::Float16_b,
  };

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
          noc_async_write_tile(tile_id, out_gen, l1_addr);
          l1_addr += tile_bytes;
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
