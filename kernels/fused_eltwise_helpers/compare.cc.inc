
enum class CompareDirection : uint32_t {
  Eq,
  Ne,
  Ge,
  Gt,
  Le,
  Lt,
};

template <bool Int32Input>
ALWI void compare_sub_init() {
  if constexpr (Int32Input) {
    sub_int_tile_init();
  } else {
    sub_binary_tile_init();
  }
}

template <bool Int32Input>
ALWI void compare_sub_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
  if constexpr (Int32Input) {
    sub_int32_tile(idst0, idst1, odst);
  } else {
    sub_binary_tile(idst0, idst1, odst);
  }
}

ALWI void compare_zero_init(CompareDirection direction) {
  switch (direction) {
    case CompareDirection::Eq: eqz_tile_init(); break;
    case CompareDirection::Ne: nez_tile_init(); break;
    case CompareDirection::Ge: gez_tile_init(); break;
    case CompareDirection::Gt: gtz_tile_init(); break;
    case CompareDirection::Le: lez_tile_init(); break;
    case CompareDirection::Lt: ltz_tile_init(); break;
    default: break;
  }
}

template <bool Int32Input>
ALWI void compare_zero_tile(CompareDirection direction, uint32_t idst) {
  switch (direction) {
    case CompareDirection::Eq:
      if constexpr (Int32Input) {
        eqz_tile_int32(idst);
      } else {
        eqz_tile(idst);
      }
      break;
    case CompareDirection::Ne:
      if constexpr (Int32Input) {
        nez_tile_int32(idst);
      } else {
        nez_tile(idst);
      }
      break;
    case CompareDirection::Ge:
      if constexpr (Int32Input) {
        gez_tile_int32(idst);
      } else {
        gez_tile(idst);
      }
      break;
    case CompareDirection::Gt:
      if constexpr (Int32Input) {
        gtz_tile_int32(idst);
      } else {
        gtz_tile(idst);
      }
      break;
    case CompareDirection::Le:
      if constexpr (Int32Input) {
        lez_tile_int32(idst);
      } else {
        lez_tile(idst);
      }
      break;
    case CompareDirection::Lt:
      if constexpr (Int32Input) {
        ltz_tile_int32(idst);
      } else {
        ltz_tile(idst);
      }
      break;
    default: break;
  }
}

