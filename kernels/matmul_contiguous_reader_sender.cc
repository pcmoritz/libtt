#include <cstdint>

#define A(n) get_arg_val<uint32_t>(n)
#define SEM(n) reinterpret_cast<volatile tt_l1_ptr uint32_t *>(get_semaphore(A(n)))

void kernel_main() {
  constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
  const uint32_t tile_bytes = get_tile_size(cb_in0);
  const uint32_t block_w = A(5);
  const uint32_t block_h = A(6);
  const uint32_t block_tiles = A(7);
  const uint32_t nblocks = A(8);
  const uint32_t w_nd = A(13);
  const uint32_t e_nd = A(18);
  const uint32_t logical_mt = A(23);
  volatile tt_l1_ptr uint32_t *sender_sem = SEM(21);
  volatile tt_l1_ptr uint32_t *recv_sem = SEM(22);
  *recv_sem = VALID;

  const InterleavedAddrGenFast<true> in0_gen = {
      .bank_base_address = A(0),
      .page_size = tile_bytes,
      .data_format = DataFormat::Float16_b,
  };
  uint32_t cur_block = A(1);
  for (uint32_t block = 0; block < nblocks; block++) {
    cb_reserve_back(cb_in0, block_tiles);
    uint32_t l1_addr = get_write_ptr(cb_in0);
    uint32_t start_addr = l1_addr;
    uint32_t row = cur_block;
    uint32_t row_tile = row / A(3);
    uint32_t block_bytes = 0;
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
    cur_block += A(4);
    noc_async_read_barrier();

    const uint32_t receivers = w_nd + e_nd;
    if (receivers > 0) {
      noc_semaphore_wait(sender_sem, receivers);
      noc_semaphore_set(sender_sem, 0);
    }
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
