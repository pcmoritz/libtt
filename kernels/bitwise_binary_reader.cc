#include <cstdint>

namespace {

using Element = BITWISE_ELEMENT_TYPE;
using SignedElement = BITWISE_SIGNED_ELEMENT_TYPE;
using UnsignedElement = BITWISE_UNSIGNED_ELEMENT_TYPE;
constexpr uint32_t OP_AND = 0;
constexpr uint32_t OP_OR = 1;
constexpr uint32_t OP_XOR = 2;
constexpr uint32_t OP_SHIFT_LEFT = 3;
constexpr uint32_t OP_SHIFT_RIGHT_LOGICAL = 4;
constexpr uint32_t OP_SHIFT_RIGHT_ARITHMETIC = 5;
constexpr uint32_t OP = BITWISE_OP;
constexpr uint32_t BIT_WIDTH = BITWISE_BIT_WIDTH;

Element all_sign_bits(Element lhs) {
  return static_cast<SignedElement>(lhs) < 0
             ? static_cast<Element>(
                   static_cast<UnsignedElement>(~static_cast<UnsignedElement>(0)))
             : static_cast<Element>(0);
}

Element apply(Element lhs, Element rhs) {
  if constexpr (OP == OP_AND) {
    return lhs & rhs;
  } else if constexpr (OP == OP_OR) {
    return lhs | rhs;
  } else if constexpr (OP == OP_XOR) {
    return lhs ^ rhs;
  } else {
    uint32_t amount = static_cast<uint32_t>(static_cast<UnsignedElement>(rhs));
    if (amount >= BIT_WIDTH) {
      if constexpr (OP == OP_SHIFT_RIGHT_ARITHMETIC) {
        return all_sign_bits(lhs);
      }
      return static_cast<Element>(0);
    }
    if constexpr (OP == OP_SHIFT_LEFT) {
      return static_cast<Element>(
          static_cast<UnsignedElement>(lhs) << amount);
    } else if constexpr (OP == OP_SHIFT_RIGHT_LOGICAL) {
      return static_cast<Element>(
          static_cast<UnsignedElement>(lhs) >> amount);
    } else {
      return static_cast<Element>(
          static_cast<SignedElement>(lhs) >> amount);
    }
  }
}

void read_tile_to_cb(const InterleavedAddrGenFast<true> &input, uint32_t tile_id,
                     uint32_t cb) {
  cb_reserve_back(cb, 1);
  noc_async_read_tile(tile_id, input, get_write_ptr(cb));
  noc_async_read_barrier();
  cb_push_back(cb, 1);
  cb_wait_front(cb, 1);
}

}  // namespace

void kernel_main() {
  uint32_t lhs_addr = get_arg_val<uint32_t>(0);
  uint32_t rhs_addr = get_arg_val<uint32_t>(1);
  uint32_t output_tile_offset = get_arg_val<uint32_t>(2);
  uint32_t output_tile_count = get_arg_val<uint32_t>(3);

  constexpr uint32_t cb_lhs = tt::CBIndex::c_0;
  constexpr uint32_t cb_rhs = tt::CBIndex::c_1;
  constexpr uint32_t cb_output = tt::CBIndex::c_16;
  const InterleavedAddrGenFast<true> lhs = {
      .bank_base_address = lhs_addr,
      .page_size = get_tile_size(cb_lhs),
      .data_format = get_dataformat(cb_lhs),
  };
  const InterleavedAddrGenFast<true> rhs = {
      .bank_base_address = rhs_addr,
      .page_size = get_tile_size(cb_rhs),
      .data_format = get_dataformat(cb_rhs),
  };

  for (uint32_t i = 0; i < output_tile_count; ++i) {
    uint32_t tile_id = output_tile_offset + i;
    read_tile_to_cb(lhs, tile_id, cb_lhs);
    read_tile_to_cb(rhs, tile_id, cb_rhs);

    cb_reserve_back(cb_output, 1);
    volatile tt_l1_ptr Element *lhs_ptr =
        reinterpret_cast<volatile tt_l1_ptr Element *>(get_read_ptr(cb_lhs));
    volatile tt_l1_ptr Element *rhs_ptr =
        reinterpret_cast<volatile tt_l1_ptr Element *>(get_read_ptr(cb_rhs));
    volatile tt_l1_ptr Element *out_ptr =
        reinterpret_cast<volatile tt_l1_ptr Element *>(get_write_ptr(cb_output));
    uint32_t elements = get_tile_size(cb_output) / sizeof(Element);
    for (uint32_t element = 0; element < elements; ++element) {
      out_ptr[element] = apply(lhs_ptr[element], rhs_ptr[element]);
    }
    cb_push_back(cb_output, 1);

    cb_pop_front(cb_lhs, 1);
    cb_pop_front(cb_rhs, 1);
  }
}
