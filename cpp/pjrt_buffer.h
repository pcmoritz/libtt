#ifndef LIBTT_CPP_PJRT_BUFFER_H_
#define LIBTT_CPP_PJRT_BUFFER_H_

#include "cpp/pjrt_internal.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

struct PJRT_Device;
struct PJRT_Memory;
struct PjrtTensorStorage;

struct PJRT_Buffer {
  PJRT_Buffer_Type buffer_type;
  std::vector<int64_t> dims;
  PJRT_Device* device;
  PJRT_Memory* memory;
  std::unique_ptr<PjrtTensorStorage> storage;
  bool deleted;
  size_t external_reference_count;

  ~PJRT_Buffer();
};

size_t BytesPerElement(PJRT_Buffer_Type type);
bool IsSupportedBufferType(PJRT_Buffer_Type type);

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
void DeletePjrtBufferStorage(PJRT_Buffer* buffer);
PJRT_Error* ReadBufferLogicalBytes(const PJRT_Buffer& buffer, std::vector<std::byte>* out);
PJRT_Error* TtnnTensorPhysicalByteSize(const PJRT_Buffer& buffer, size_t* out);

#endif  // LIBTT_CPP_PJRT_BUFFER_H_
