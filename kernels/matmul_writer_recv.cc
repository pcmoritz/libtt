void kernel_main() {
  constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
  const uint32_t block_tiles = A(7);
  const uint32_t nblocks = A(8);
  const uint32_t local_batch_count = A(32);
  const uint32_t batch_start = A(33);
  const uint32_t total_batch_count = A(34);
  const OutputDrain output_drain = load_output_drain();
  volatile tt_l1_ptr uint32_t *recv_sem = SEM(17);

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

    drain_output_blocks(output_drain, batch, valid_batch);
  }
}
