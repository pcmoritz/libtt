#include <cstdint>

namespace {

constexpr uint32_t TILE_R = 32;
constexpr uint32_t TILE_C = 32;
constexpr uint32_t FACE_R = 16;
constexpr uint32_t FACE_C = 16;
constexpr uint32_t NEG_INF_BF16_PAIR = 0xff7fff7f;

uint32_t tile_element_index(uint32_t row, uint32_t col) {
  uint32_t face_row = row / FACE_R;
  uint32_t face_col = col / FACE_C;
  uint32_t row_in_face = row % FACE_R;
  uint32_t col_in_face = col % FACE_C;
  return ((face_row * 2 + face_col) * FACE_R * FACE_C) + row_in_face * FACE_C + col_in_face;
}

uint32_t tile_row_word_index(uint32_t row, uint32_t face_col) {
  return ((row / FACE_R * 2 + face_col) * FACE_R * FACE_C +
          (row % FACE_R) * FACE_C) /
         2;
}

int32_t read_s32_element(uint32_t l1_addr, uint32_t row) {
  volatile tt_l1_ptr int32_t *ptr = reinterpret_cast<volatile tt_l1_ptr int32_t *>(l1_addr);
  return ptr[tile_element_index(0, row)];
}

template <typename AddrGen>
void read_s32_tile(const AddrGen &gen, uint32_t tile_id, uint32_t cb) {
  cb_reserve_back(cb, 1);
  noc_async_read_tile(tile_id, gen, get_write_ptr(cb));
  noc_async_read_barrier();
}

void copy_bf16_row_from_l1(
    uint32_t source_l1,
    uint32_t source_row,
    uint32_t dst_addr,
    uint32_t dst_row) {
  volatile tt_l1_ptr uint32_t *src = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(source_l1);
  volatile tt_l1_ptr uint32_t *dst = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(dst_addr);
  for (uint32_t face_col = 0; face_col < 2; ++face_col) {
    uint32_t source_word = tile_row_word_index(source_row, face_col);
    uint32_t dst_word = tile_row_word_index(dst_row, face_col);
    for (uint32_t word = 0; word < FACE_C / 2; ++word) {
      dst[dst_word + word] = src[source_word + word];
    }
  }
}

void zero_bf16_row(uint32_t l1_addr, uint32_t row) {
  volatile tt_l1_ptr uint32_t *ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(l1_addr);
  for (uint32_t face_col = 0; face_col < 2; ++face_col) {
    uint32_t dst_word = tile_row_word_index(row, face_col);
    for (uint32_t word = 0; word < FACE_C / 2; ++word) {
      ptr[dst_word + word] = 0;
    }
  }
}

template <typename KAddrGen, typename VAddrGen>
void copy_bf16_kv_rows_from_tiles(
    const KAddrGen &k_gen,
    const VAddrGen &v_gen,
    uint32_t k_tile,
    uint32_t v_tile,
    uint32_t source_row,
    uint32_t k_dst_addr,
    uint32_t v_dst_addr,
    uint32_t dst_row,
    uint32_t cb_temp) {
  uint32_t tile_bytes = get_tile_size(cb_temp);
  cb_reserve_back(cb_temp, 2);
  uint32_t temp_base = get_write_ptr(cb_temp);
  noc_async_read_tile(k_tile, k_gen, temp_base);
  noc_async_read_tile(v_tile, v_gen, temp_base + tile_bytes);
  noc_async_read_barrier();
  copy_bf16_row_from_l1(temp_base, source_row, k_dst_addr, dst_row);
  copy_bf16_row_from_l1(temp_base + tile_bytes, source_row, v_dst_addr, dst_row);
  cb_push_back(cb_temp, 2);
  cb_pop_front(cb_temp, 2);
}

template <typename AddrGen>
void copy_bf16_fused_kv_rows_from_tile(
    const AddrGen &kv_gen,
    uint32_t kv_tile,
    uint32_t source_row,
    uint32_t k_dst_addr,
    uint32_t v_dst_addr,
    uint32_t dst_row,
    uint32_t cb_temp) {
  cb_reserve_back(cb_temp, 1);
  uint32_t temp_base = get_write_ptr(cb_temp);
  noc_async_read_tile(kv_tile, kv_gen, temp_base);
  noc_async_read_barrier();
  copy_bf16_row_from_l1(temp_base, source_row, k_dst_addr, dst_row);
  copy_bf16_row_from_l1(temp_base, source_row + 1, v_dst_addr, dst_row);
  cb_push_back(cb_temp, 1);
  cb_pop_front(cb_temp, 1);
}

void write_mask_tile(uint32_t l1_addr, const bool valid[TILE_C]) {
  volatile tt_l1_ptr uint32_t *ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(l1_addr);
  uint32_t mask_words[2][FACE_C / 2];
  for (uint32_t face_col = 0; face_col < 2; ++face_col) {
    for (uint32_t word = 0; word < FACE_C / 2; ++word) {
      uint32_t col_pair = face_col * FACE_C + word * 2;
      uint32_t first = valid[col_pair] ? 0 : (NEG_INF_BF16_PAIR & 0xffffu);
      uint32_t second = valid[col_pair + 1] ? 0 : (NEG_INF_BF16_PAIR & 0xffff0000u);
      mask_words[face_col][word] = first | second;
    }
  }
  for (uint32_t row = 0; row < TILE_R; ++row) {
    for (uint32_t face_col = 0; face_col < 2; ++face_col) {
      uint32_t dst_word = tile_row_word_index(row, face_col);
      for (uint32_t word = 0; word < FACE_C / 2; ++word) {
        ptr[dst_word + word] = mask_words[face_col][word];
      }
    }
  }
}

}  // namespace

void kernel_main() {
  constexpr uint32_t ST = SDPA_ST;
  constexpr uint32_t DHT = SDPA_DHT;
  constexpr uint32_t SK_CHUNK_T = SDPA_SK_CHUNK_T;
  constexpr uint32_t KV_HEADS = SDPA_KV_HEADS;
  constexpr uint32_t CACHE_TOKENS = SDPA_CACHE_TOKENS;
  constexpr uint32_t Q_CHUNK_TILES = DHT;
  constexpr uint32_t KV_CHUNK_TILES = SK_CHUNK_T * DHT;
  constexpr uint32_t MASK_CHUNK_TILES = SK_CHUNK_T;

  uint32_t arg_idx = 0;
  uint32_t q_addr = get_arg_val<uint32_t>(arg_idx++);
  uint32_t k_addr = get_arg_val<uint32_t>(arg_idx++);
  uint32_t v_addr = get_arg_val<uint32_t>(arg_idx++);
  uint32_t seq_lens_addr = get_arg_val<uint32_t>(arg_idx++);
  uint32_t loc_addr = get_arg_val<uint32_t>(arg_idx++);
  uint32_t cur_kv_head = get_arg_val<uint32_t>(arg_idx++);

  constexpr uint32_t cb_q = tt::CBIndex::c_0;
  constexpr uint32_t cb_k = tt::CBIndex::c_1;
  constexpr uint32_t cb_v = tt::CBIndex::c_2;
  constexpr uint32_t cb_mask = tt::CBIndex::c_3;
  constexpr uint32_t cb_seq = tt::CBIndex::c_8;
  constexpr uint32_t cb_loc = tt::CBIndex::c_9;
  constexpr uint32_t cb_temp = tt::CBIndex::c_10;

  const InterleavedAddrGenFast<true> q_reader = {
      .bank_base_address = q_addr,
      .page_size = get_tile_size(cb_q),
      .data_format = get_dataformat(cb_q),
  };
  const InterleavedAddrGenFast<true> k_reader = {
      .bank_base_address = k_addr,
      .page_size = get_tile_size(cb_k),
      .data_format = get_dataformat(cb_k),
  };
#ifndef SDPA_FUSED_KV
  const InterleavedAddrGenFast<true> v_reader = {
      .bank_base_address = v_addr,
      .page_size = get_tile_size(cb_v),
      .data_format = get_dataformat(cb_v),
  };
#endif
  const InterleavedAddrGenFast<true> seq_reader = {
      .bank_base_address = seq_lens_addr,
      .page_size = get_tile_size(cb_seq),
      .data_format = get_dataformat(cb_seq),
  };
  const InterleavedAddrGenFast<true> loc_reader = {
      .bank_base_address = loc_addr,
      .page_size = get_tile_size(cb_loc),
      .data_format = get_dataformat(cb_loc),
  };

  read_s32_tile(seq_reader, 0, cb_seq);
  int32_t seq_len = read_s32_element(get_write_ptr(cb_seq), 0);
  uint32_t effective_seq_len = seq_len > 0 ? static_cast<uint32_t>(seq_len) : 0;
  cb_push_back(cb_seq, 1);

  cb_reserve_back(cb_q, Q_CHUNK_TILES);
  uint32_t q_write = get_write_ptr(cb_q);
  for (uint32_t tile = 0; tile < Q_CHUNK_TILES; ++tile) {
    noc_async_read_tile(tile, q_reader, q_write + tile * get_tile_size(cb_q));
  }
  noc_async_read_barrier();
  cb_push_back(cb_q, Q_CHUNK_TILES);

  uint32_t loaded_loc_tile = 0xffffffffu;
  uint32_t loc_l1_addr = 0;
  uint32_t active_st = (effective_seq_len + TILE_R - 1) / TILE_R;
  if (active_st > ST) {
    active_st = ST;
  }
  uint32_t active_sk_chunk_t = active_st < SK_CHUNK_T ? active_st : SK_CHUNK_T;
  for (uint32_t chunk = 0; chunk < active_st; chunk += active_sk_chunk_t) {
    uint32_t active_kv_tiles = active_sk_chunk_t * DHT;
    cb_reserve_back(cb_k, active_kv_tiles);
    cb_reserve_back(cb_v, active_kv_tiles);
    cb_reserve_back(cb_mask, active_sk_chunk_t);
    uint32_t k_base = get_write_ptr(cb_k);
    uint32_t v_base = get_write_ptr(cb_v);
    uint32_t mask_base = get_write_ptr(cb_mask);
    uint32_t tile_bytes = get_tile_size(cb_k);
    bool valid_rows[SK_CHUNK_T][TILE_R];

    for (uint32_t sk = 0; sk < active_sk_chunk_t; ++sk) {
      uint32_t global_tile = chunk + sk;
      uint32_t cache_tiles[TILE_R];
      if (loaded_loc_tile != global_tile) {
        if (loaded_loc_tile != 0xffffffffu) {
          cb_push_back(cb_loc, 1);
          cb_wait_front(cb_loc, 1);
          cb_pop_front(cb_loc, 1);
        }
        read_s32_tile(loc_reader, global_tile, cb_loc);
        loc_l1_addr = get_write_ptr(cb_loc);
        loaded_loc_tile = global_tile;
      }

      for (uint32_t row = 0; row < TILE_R; ++row) {
        uint32_t pos = global_tile * TILE_R + row;
        int32_t cache_index = read_s32_element(loc_l1_addr, row);
        bool valid = pos < effective_seq_len && cache_index > 0 &&
                     cache_index < static_cast<int32_t>(CACHE_TOKENS);
        valid_rows[sk][row] = valid;
#ifdef SDPA_FUSED_KV
        cache_tiles[row] = valid ? static_cast<uint32_t>(cache_index) * DHT : 0;
#else
        cache_tiles[row] = valid ? static_cast<uint32_t>(cache_index) * DHT : 0;
#endif
      }

      for (uint32_t d = 0; d < DHT; ++d) {
        uint32_t k_tile_index = d * active_sk_chunk_t + sk;
        uint32_t v_tile_index = sk * DHT + d;
        uint32_t k_dst = k_base + k_tile_index * tile_bytes;
        uint32_t v_dst = v_base + v_tile_index * tile_bytes;

        for (uint32_t row = 0; row < TILE_R; ++row) {
          if (valid_rows[sk][row]) {
            uint32_t cache_tile = cache_tiles[row] + d;
#ifdef SDPA_FUSED_KV
            copy_bf16_fused_kv_rows_from_tile(
                k_reader,
                cache_tile,
                cur_kv_head * 2,
                k_dst,
                v_dst,
                row,
                cb_temp);
#else
            copy_bf16_kv_rows_from_tiles(
                k_reader,
                v_reader,
                cache_tile,
                cache_tile,
                cur_kv_head,
                k_dst,
                v_dst,
                row,
                cb_temp);
#endif
          } else {
            zero_bf16_row(k_dst, row);
            zero_bf16_row(v_dst, row);
          }
        }
      }
    }

    noc_async_read_barrier();
    for (uint32_t sk = 0; sk < active_sk_chunk_t; ++sk) {
      write_mask_tile(mask_base + sk * get_tile_size(cb_mask), valid_rows[sk]);
    }
    cb_push_back(cb_k, active_kv_tiles);
    cb_push_back(cb_v, active_kv_tiles);
    cb_push_back(cb_mask, active_sk_chunk_t);
  }

  if (loaded_loc_tile != 0xffffffffu) {
    cb_push_back(cb_loc, 1);
    cb_wait_front(cb_loc, 1);
    cb_pop_front(cb_loc, 1);
  }
}
