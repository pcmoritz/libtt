#ifndef LIBTT_CPP_PJRT_BUFFER_H_
#define LIBTT_CPP_PJRT_BUFFER_H_

#include "cpp/pjrt_internal.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include <ttnn/tensor/tensor.hpp>

namespace tt::tt_metal::distributed {
class MeshDevice;
}  // namespace tt::tt_metal::distributed

struct PJRT_Device;
struct PJRT_Memory;

struct PJRT_Buffer {
  PJRT_Buffer(PJRT_Buffer_Type buffer_type,
              std::vector<int64_t> dims,
              PJRT_Device* device,
              PJRT_Memory* memory,
              ttnn::Tensor tensor);
  PJRT_Buffer_Type buffer_type;
  std::vector<int64_t> dims;
  PJRT_Device* device;
  PJRT_Memory* memory;
  std::optional<ttnn::Tensor> tensor;
  size_t external_reference_count = 0;

  ~PJRT_Buffer();

  ttnn::Tensor* TtnnTensor();
  const ttnn::Tensor* TtnnTensor() const;
  bool IsDeleted() const;
  void Delete();
};

size_t BytesPerElement(PJRT_Buffer_Type type);
bool IsSupportedBufferType(PJRT_Buffer_Type type);
std::optional<tt::tt_metal::DataType> TtnnDataTypeForPjrtBufferType(PJRT_Buffer_Type type);

PJRT_Error* CopyDims(const int64_t* dims, size_t num_dims, std::vector<int64_t>* out);
PJRT_Error* HostByteSize(PJRT_Buffer_Type type, const std::vector<int64_t>& dims, size_t* out);
PJRT_Error* ValidateDenseRowMajorStrides(PJRT_Buffer_Type type,
                                         const std::vector<int64_t>& dims,
                                         const int64_t* byte_strides,
                                         size_t num_byte_strides);

PJRT_Error* CreatePjrtBufferFromHostBytes(PJRT_Buffer_Type type,
                                          const std::vector<int64_t>& dims,
                                          PJRT_Device* target_device,
                                          PJRT_Memory* target_memory,
                                          const void* data,
                                          size_t byte_size,
                                          PJRT_Buffer** out);
PJRT_Error* CreatePjrtBufferFromTtnnTensor(PJRT_Buffer_Type type,
                                           const std::vector<int64_t>& dims,
                                           PJRT_Device* target_device,
                                           PJRT_Memory* target_memory,
                                           ttnn::Tensor tensor,
                                           PJRT_Buffer** out);
PJRT_Error* CopyPjrtBufferToTtnnDeviceTensor(
    const PJRT_Buffer& buffer,
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    ttnn::Tensor* out);
PJRT_Error* ReadBufferLogicalBytes(const PJRT_Buffer& buffer, std::vector<std::byte>* out);
PJRT_Error* TtnnTensorPhysicalByteSize(const PJRT_Buffer& buffer, size_t* out);

#endif  // LIBTT_CPP_PJRT_BUFFER_H_
