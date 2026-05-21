#include <cstdint>

#define A(n) get_arg_val<uint32_t>(n)
#define SEM(n) reinterpret_cast<volatile tt_l1_ptr uint32_t *>(get_semaphore(A(n)))

void kernel_main() {
  constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
  const uint32_t block_tiles = A(7);
  const uint32_t nblocks = A(8);
  const uint32_t local_batch_count = A(24);
  volatile tt_l1_ptr uint32_t *recv_sem = SEM(22);

  for (uint32_t batch = 0; batch < local_batch_count; batch++) {
    for (uint32_t block = 0; block < nblocks; block++) {
      cb_reserve_back(cb_in0, block_tiles);
      noc_semaphore_set(recv_sem, INVALID);
      noc_semaphore_inc(get_noc_addr(A(19), A(20), get_semaphore(A(21))), 1);
      noc_semaphore_wait(recv_sem, VALID);
      cb_push_back(cb_in0, block_tiles);
    }
  }
}
