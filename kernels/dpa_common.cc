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
void fill_u32_tile_at(uint32_t tile_addr, uint32_t words, uint32_t value) {
  volatile tt_l1_ptr uint32_t *ptr =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(tile_addr);
  for (uint32_t i = 0; i < words; ++i) {
    ptr[i] = value;
  }
}
void zero_bf16_tile(uint32_t cb) {
  fill_u32_tile_at(get_write_ptr(cb), TILE_R * TILE_C / 2, 0);
}
void zero_bf16_tile_at(uint32_t tile_addr) {
  fill_u32_tile_at(tile_addr, TILE_R * TILE_C / 2, 0);
}
void copy_bf16_tile(uint32_t dst_addr, uint32_t src_addr) {
  volatile tt_l1_ptr uint32_t *dst =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(dst_addr);
  volatile tt_l1_ptr uint32_t *src =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(src_addr);
  for (uint32_t i = 0; i < TILE_R * TILE_C / 2; ++i) {
    dst[i] = src[i];
  }
}
}  // namespace
