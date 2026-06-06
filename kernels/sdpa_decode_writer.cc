#include <cstdint>

namespace {

constexpr uint32_t TILE_R = 32;
constexpr uint32_t TILE_C = 32;
constexpr uint32_t FACE_R = 16;
constexpr uint32_t FACE_C = 16;
constexpr uint32_t BF16_BYTES = 2;

uint32_t tile_element_index(uint32_t row, uint32_t col) {
  uint32_t face_row = row / FACE_R;
  uint32_t face_col = col / FACE_C;
  uint32_t row_in_face = row % FACE_R;
  uint32_t col_in_face = col % FACE_C;
  return ((face_row * 2 + face_col) * FACE_R * FACE_C) + row_in_face * FACE_C + col_in_face;
}

void zero_tile(uint32_t cb) {
  volatile tt_l1_ptr uint32_t *ptr =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(get_write_ptr(cb));
  uint32_t words = get_tile_size(cb) / sizeof(uint32_t);
  for (uint32_t i = 0; i < words; ++i) {
    ptr[i] = 0;
  }
}

void generate_bcast_scalar(uint32_t cb, uint32_t bf16_packed) {
  cb_reserve_back(cb, 1);
  volatile tt_l1_ptr uint32_t *ptr =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(get_write_ptr(cb));
  ptr[0] = bf16_packed >> 16;
  cb_push_back(cb, 1);
}

void generate_reduce_scaler(uint32_t cb, uint32_t bf16_packed) {
  cb_reserve_back(cb, 1);
  zero_tile(cb);
  volatile tt_l1_ptr uint32_t *ptr =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(get_write_ptr(cb));
  for (uint32_t face = 0; face < 4; ++face) {
    uint32_t face_offset = face << 7;
    for (uint32_t col = 0; col < 8; ++col) {
      ptr[face_offset + col] = bf16_packed;
    }
  }
  cb_push_back(cb, 1);
}

template <typename AddrGen>
void write_bf16_row_to_tile(
    const AddrGen &gen,
    uint32_t output_tile,
    uint32_t source_addr,
    uint32_t row) {
  uint32_t first = tile_element_index(row, 0) * BF16_BYTES;
  uint32_t second = tile_element_index(row, FACE_C) * BF16_BYTES;
  noc_async_write(source_addr + first, get_noc_addr(output_tile, gen, first), FACE_C * BF16_BYTES);
  noc_async_write(source_addr + second, get_noc_addr(output_tile, gen, second), FACE_C * BF16_BYTES);
}

}  // namespace

void kernel_main() {
  constexpr uint32_t DHT = SDPA_DHT;
  constexpr uint32_t Q_HEADS = SDPA_Q_HEADS;
  constexpr uint32_t KV_HEADS = SDPA_KV_HEADS;
  constexpr uint32_t HEADS_PER_KV = Q_HEADS / KV_HEADS;
  constexpr uint32_t OUT_CHUNK_TILES = DHT;
  constexpr uint32_t SCALE_BF16_PACKED = SDPA_SCALE_BF16_PACKED;
  constexpr uint32_t IDENTITY_BF16_PACKED = 0x3f803f80;

  uint32_t arg_idx = 0;
  uint32_t out_addr = get_arg_val<uint32_t>(arg_idx++);
  uint32_t cur_kv_head = get_arg_val<uint32_t>(arg_idx++);

  constexpr uint32_t cb_scale = tt::CBIndex::c_4;
  constexpr uint32_t cb_identity_scale = tt::CBIndex::c_5;
  constexpr uint32_t cb_out = tt::CBIndex::c_20;

  generate_bcast_scalar(cb_scale, SCALE_BF16_PACKED);
  generate_reduce_scaler(cb_identity_scale, IDENTITY_BF16_PACKED);

  const InterleavedAddrGenFast<true> out_writer = {
      .bank_base_address = out_addr,
      .page_size = get_tile_size(cb_out),
      .data_format = get_dataformat(cb_out),
  };

  cb_wait_front(cb_out, OUT_CHUNK_TILES);
  uint32_t out_l1 = get_read_ptr(cb_out);
  uint32_t row_start = cur_kv_head * HEADS_PER_KV;
  for (uint32_t d = 0; d < DHT; ++d) {
    uint32_t tile_l1 = out_l1 + d * get_tile_size(cb_out);
    for (uint32_t row = 0; row < HEADS_PER_KV; ++row) {
      write_bf16_row_to_tile(out_writer, d, tile_l1, row_start + row);
    }
  }
  noc_async_write_barrier();
  cb_pop_front(cb_out, OUT_CHUNK_TILES);
}
