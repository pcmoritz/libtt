
constexpr DataFormat binary_input_data_format(uint32_t cb_lhs, uint32_t cb_out) {
#ifdef UCK_CHLKC_PACK
  return static_cast<DataFormat>((uint)pack_src_format[cb_out]);
#else
  return static_cast<DataFormat>((uint)unpack_src_format[cb_lhs]);
#endif
}

