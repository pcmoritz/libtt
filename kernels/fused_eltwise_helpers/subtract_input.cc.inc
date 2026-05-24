
template <DataFormat Format>
ALWI void subtract_input_init() {
  if constexpr (Format == DataFormat::Float16 || Format == DataFormat::Float16_b ||
                Format == DataFormat::Float32) {
    sub_binary_tile_init();
  } else if constexpr (Format == DataFormat::Int32) {
    sub_int_tile_init();
  }
}

template <DataFormat Format>
ALWI void subtract_input_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
  if constexpr (Format == DataFormat::Float16 || Format == DataFormat::Float16_b ||
                Format == DataFormat::Float32) {
    sub_binary_tile(idst0, idst1, odst);
  } else if constexpr (Format == DataFormat::Int32) {
    sub_int32_tile(idst0, idst1, odst);
  }
}

