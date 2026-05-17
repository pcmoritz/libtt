#include <cstdint>

namespace {

constexpr uint32_t TILE_R = 32;
constexpr uint32_t TILE_C = 32;
constexpr uint32_t FACE_R = 16;
constexpr uint32_t FACE_C = 16;
constexpr uint32_t INVALID_KEY = 0xffffffffu;

#define A(n) get_arg_val<uint32_t>(n)

uint32_t tile_element_index(uint32_t row, uint32_t col) {
  uint32_t face_row = row / FACE_R;
  uint32_t face_col = col / FACE_C;
  uint32_t row_in_face = row % FACE_R;
  uint32_t col_in_face = col % FACE_C;
  return ((face_row * 2 + face_col) * FACE_R * FACE_C) + row_in_face * FACE_C + col_in_face;
}

void zero_bf16_tile_at(uint32_t tile_addr) {
  volatile tt_l1_ptr uint32_t *ptr =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(tile_addr);
  constexpr uint32_t words = TILE_R * TILE_C / 2;
  for (uint32_t i = 0; i < words; ++i) {
    ptr[i] = 0;
  }
}

void copy_bf16_tile(uint32_t dst_addr, uint32_t src_addr) {
  volatile tt_l1_ptr uint32_t *dst =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(dst_addr);
  volatile tt_l1_ptr uint32_t *src =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(src_addr);
  constexpr uint32_t words = TILE_R * TILE_C / 2;
  for (uint32_t i = 0; i < words; ++i) {
    dst[i] = src[i];
  }
}

}  // namespace

void kernel_main() {
  constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
  constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
  constexpr uint32_t cb_tmp = tt::CBIndex::c_2;
  constexpr uint32_t cb_rhs_cache = tt::CBIndex::c_3;
  constexpr uint32_t one_tile = 1;

  const uint32_t lhs_addr = A(0);
  const uint32_t rhs_addr = A(1);
  const uint32_t output_tile_offset = A(2);
  const uint32_t output_tile_count = A(3);
  const uint32_t query_tokens = A(4);
  const uint32_t key_tokens = A(5);
  const uint32_t kv_heads = A(6);
  const uint32_t groups = A(7);
  const uint32_t head_dim = A(8);
  const uint32_t lhs_tiles_per_head = A(9);
  const uint32_t rhs_tiles_per_head = A(10);
  const uint32_t output_tiles_per_row = A(11);
  const uint32_t kt = A(12);

  const InterleavedAddrGenFast<true> lhs = {
      .bank_base_address = lhs_addr,
      .page_size = get_tile_size(cb_in0),
      .data_format = DataFormat::Float16_b,
  };
  const InterleavedAddrGenFast<true> rhs = {
      .bank_base_address = rhs_addr,
      .page_size = get_tile_size(cb_tmp),
      .data_format = DataFormat::Float16_b,
  };

  uint32_t cached_batch = INVALID_KEY;
  uint32_t cached_kv_head = INVALID_KEY;
  uint32_t cached_s_tile = INVALID_KEY;
  bool cache_valid = false;
  const uint32_t rhs_cache_tile_size = get_tile_size(cb_rhs_cache);

  for (uint32_t local_tile = 0; local_tile < output_tile_count; ++local_tile) {
    const uint32_t output_tile = output_tile_offset + local_tile;
    const uint32_t s_tile = output_tile % output_tiles_per_row;
    uint32_t prefix = output_tile / output_tiles_per_row;
    const uint32_t query_token = prefix % query_tokens;
    prefix /= query_tokens;
    const uint32_t kv_head = prefix % kv_heads;
    const uint32_t batch = prefix / kv_heads;

    if (!cache_valid || cached_batch != batch || cached_kv_head != kv_head ||
        cached_s_tile != s_tile) {
      if (cache_valid) {
        cb_pop_front(cb_rhs_cache, kt);
      }
      cb_reserve_back(cb_rhs_cache, kt);
      const uint32_t cache_base = get_write_ptr(cb_rhs_cache);
      for (uint32_t k_tile = 0; k_tile < kt; ++k_tile) {
        const uint32_t cache_tile_addr = cache_base + k_tile * rhs_cache_tile_size;
        zero_bf16_tile_at(cache_tile_addr);
        volatile tt_l1_ptr uint16_t *packed_rhs =
            reinterpret_cast<volatile tt_l1_ptr uint16_t *>(cache_tile_addr);

        for (uint32_t col = 0; col < TILE_C; ++col) {
          const uint32_t key_token = s_tile * TILE_C + col;
          if (key_token >= key_tokens) {
            continue;
          }

          cb_reserve_back(cb_tmp, one_tile);
          const uint32_t rhs_prefix = batch * key_tokens + key_token;
          const uint32_t rhs_tile = rhs_prefix * rhs_tiles_per_head + k_tile;
          noc_async_read_tile(rhs_tile, rhs, get_write_ptr(cb_tmp));
          noc_async_read_barrier();
          cb_push_back(cb_tmp, one_tile);
          cb_wait_front(cb_tmp, one_tile);

          volatile tt_l1_ptr uint16_t *source_rhs =
              reinterpret_cast<volatile tt_l1_ptr uint16_t *>(get_read_ptr(cb_tmp));
          for (uint32_t row = 0; row < TILE_R; ++row) {
            const uint32_t head_offset = k_tile * TILE_C + row;
            if (head_offset >= head_dim) {
              break;
            }
            const uint32_t src_index = tile_element_index(kv_head, row);
            const uint32_t dst_index = tile_element_index(row, col);
            packed_rhs[dst_index] = source_rhs[src_index];
          }
          cb_pop_front(cb_tmp, one_tile);
        }
      }
      cb_push_back(cb_rhs_cache, kt);
      cb_wait_front(cb_rhs_cache, kt);
      cached_batch = batch;
      cached_kv_head = kv_head;
      cached_s_tile = s_tile;
      cache_valid = true;
    }

    for (uint32_t k_tile = 0; k_tile < kt; ++k_tile) {
      cb_reserve_back(cb_in0, one_tile);
      const uint32_t lhs_prefix = (batch * query_tokens + query_token) * kv_heads + kv_head;
      const uint32_t lhs_tile = lhs_prefix * lhs_tiles_per_head + k_tile;
      noc_async_read_tile(lhs_tile, lhs, get_write_ptr(cb_in0));
      noc_async_read_barrier();
      cb_push_back(cb_in0, one_tile);

      cb_reserve_back(cb_in1, one_tile);
      const uint32_t cache_tile_addr = get_read_ptr(cb_rhs_cache) + k_tile * rhs_cache_tile_size;
      copy_bf16_tile(get_write_ptr(cb_in1), cache_tile_addr);
      cb_push_back(cb_in1, one_tile);
    }
  }

  if (cache_valid) {
    cb_pop_front(cb_rhs_cache, kt);
  }
}
