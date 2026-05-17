#include <cstdint>

namespace {

constexpr uint32_t TILE_R = 32;
constexpr uint32_t TILE_C = 32;
constexpr uint32_t FACE_R = 16;
constexpr uint32_t FACE_C = 16;

#define A(n) get_arg_val<uint32_t>(n)

uint32_t tile_element_index(uint32_t row, uint32_t col) {
  uint32_t face_row = row / FACE_R;
  uint32_t face_col = col / FACE_C;
  uint32_t row_in_face = row % FACE_R;
  uint32_t col_in_face = col % FACE_C;
  return ((face_row * 2 + face_col) * FACE_R * FACE_C) + row_in_face * FACE_C + col_in_face;
}

void zero_bf16_tile(uint32_t cb) {
  volatile tt_l1_ptr uint32_t *ptr =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(get_write_ptr(cb));
  constexpr uint32_t words = TILE_R * TILE_C / 2;
  for (uint32_t i = 0; i < words; ++i) {
    ptr[i] = 0;
  }
}

}  // namespace

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
  const uint32_t head_tiles = A(7);
  const uint32_t output_tiles_per_row = A(8);

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
