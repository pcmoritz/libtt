void kernel_main() {
  constexpr uint32_t cb_compute_out = tt::CBIndex::c_16;
  constexpr uint32_t cb_output = tt::CBIndex::c_17;
  constexpr uint32_t one_tile = 1;
  const uint32_t out_addr = A(0);
  const uint32_t work_tile_offset = A(1);
  const uint32_t work_tile_count = A(2);
  const uint32_t groups = A(3);
  const uint32_t query_tokens = A(4);
  const uint32_t kv_heads = A(5);
  const uint32_t head_dim = A(6);
  const uint32_t head_tiles = (head_dim + TILE_R - 1) / TILE_R;
  const uint32_t output_tiles_per_row = (query_tokens + TILE_C - 1) / TILE_C;
  const InterleavedAddrGenFast<true> out = {
      .bank_base_address = out_addr,
      .page_size = get_tile_size(cb_output),
      .data_format = get_dataformat(cb_output),
  };
  const uint32_t compute_tile_size = get_tile_size(cb_compute_out);
  for (uint32_t local_work = 0; local_work < work_tile_count; ++local_work) {
    const uint32_t work_tile = work_tile_offset + local_work;
    const uint32_t t_tile = work_tile % output_tiles_per_row;
    uint32_t prefix = work_tile / output_tiles_per_row;
    const uint32_t head_tile = prefix % head_tiles;
    prefix /= head_tiles;
    const uint32_t kv_head = prefix % kv_heads;
    const uint32_t batch = prefix / kv_heads;
    const uint32_t head_base = head_tile * TILE_R;
    cb_wait_front(cb_compute_out, groups);
    const uint32_t compute_base = get_read_ptr(cb_compute_out);
    for (uint32_t head_row = 0; head_row < TILE_R; ++head_row) {
      const uint32_t head = head_base + head_row;
      if (head >= head_dim) {
        break;
      }
      cb_reserve_back(cb_output, one_tile);
      zero_bf16_tile(cb_output);
      volatile tt_l1_ptr uint16_t *packed_output =
          reinterpret_cast<volatile tt_l1_ptr uint16_t *>(get_write_ptr(cb_output));
      for (uint32_t group = 0; group < groups; ++group) {
        volatile tt_l1_ptr uint16_t *compute_output =
            reinterpret_cast<volatile tt_l1_ptr uint16_t *>(compute_base + group * compute_tile_size);
        for (uint32_t col = 0; col < TILE_C; ++col) {
          const uint32_t query_token = t_tile * TILE_C + col;
          if (query_token >= query_tokens) {
            break;
          }
          const uint32_t src_index = tile_element_index(head_row, col);
          const uint32_t dst_index = tile_element_index(group, col);
          packed_output[dst_index] = compute_output[src_index];
        }
      }
      cb_push_back(cb_output, one_tile);
      cb_wait_front(cb_output, one_tile);
      const uint32_t output_tile =
          (((batch * kv_heads + kv_head) * head_dim + head) * output_tiles_per_row) + t_tile;
      noc_async_write_tile(output_tile, out, get_read_ptr(cb_output));
      noc_async_write_barrier();
      cb_pop_front(cb_output, one_tile);
    }
    cb_pop_front(cb_compute_out, groups);
  }
}
