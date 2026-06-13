#ifndef LIBTT_CPP_TT_METAL_MATMUL_RUNTIME_H_
#define LIBTT_CPP_TT_METAL_MATMUL_RUNTIME_H_

#include "cpp/pjrt_internal.h"

#include <cstddef>
#include <cstdint>
#include <vector>

struct TtMetalMatmulOperand {
  PJRT_Buffer_Type type = PJRT_Buffer_Type_INVALID;
  std::vector<int64_t> dims;
  std::vector<std::byte> data;
};

struct TtMetalMatmulRequest {
  int local_hardware_id = 0;
  TtMetalMatmulOperand lhs;
  TtMetalMatmulOperand rhs;
  PJRT_Buffer_Type output_type = PJRT_Buffer_Type_INVALID;
  std::vector<int64_t> output_dims;
  std::vector<int64_t> lhs_batching_dimensions;
  std::vector<int64_t> rhs_batching_dimensions;
  std::vector<int64_t> lhs_contracting_dimensions;
  std::vector<int64_t> rhs_contracting_dimensions;
};

struct TtMetalEmbeddingRequest {
  int local_hardware_id = 0;
  TtMetalMatmulOperand indices;
  TtMetalMatmulOperand table;
  PJRT_Buffer_Type output_type = PJRT_Buffer_Type_INVALID;
  std::vector<int64_t> output_dims;
};

struct TtMetalTopKRequest {
  int local_hardware_id = 0;
  TtMetalMatmulOperand input;
  uint32_t k = 0;
  PJRT_Buffer_Type values_output_type = PJRT_Buffer_Type_INVALID;
  std::vector<int64_t> values_output_dims;
  PJRT_Buffer_Type indices_output_type = PJRT_Buffer_Type_INVALID;
  std::vector<int64_t> indices_output_dims;
};

PJRT_Error* ExecuteTtMetalMatmul(const TtMetalMatmulRequest& request,
                                 std::vector<std::byte>* output);
PJRT_Error* ExecuteTtMetalEmbedding(const TtMetalEmbeddingRequest& request,
                                    std::vector<std::byte>* output);
PJRT_Error* ExecuteTtMetalTopK(const TtMetalTopKRequest& request,
                               std::vector<std::byte>* values_output,
                               std::vector<std::byte>* indices_output);

#endif  // LIBTT_CPP_TT_METAL_MATMUL_RUNTIME_H_
