#include "cpp/pjrt_buffer.h"

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/alignment.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/page_config.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/tensor_layout.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/types.hpp>

#include <algorithm>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

struct PjrtTensorStorage {
  explicit PjrtTensorStorage(ttnn::Tensor tensor) : tensor(std::move(tensor)) {}

  ttnn::Tensor tensor;
};

PJRT_Buffer::~PJRT_Buffer() = default;

namespace {

constexpr uint32_t kTileRows = 32;
constexpr uint32_t kTileCols = 32;

std::optional<tt::tt_metal::DataType> MetalDataType(PJRT_Buffer_Type type) {
  switch (type) {
    case PJRT_Buffer_Type_PRED:
    case PJRT_Buffer_Type_U8:
      return tt::tt_metal::DataType::UINT8;
    case PJRT_Buffer_Type_U16:
      return tt::tt_metal::DataType::UINT16;
    case PJRT_Buffer_Type_S32:
      return tt::tt_metal::DataType::INT32;
    case PJRT_Buffer_Type_U32:
      return tt::tt_metal::DataType::UINT32;
    case PJRT_Buffer_Type_BF16:
      return tt::tt_metal::DataType::BFLOAT16;
    case PJRT_Buffer_Type_F32:
      return tt::tt_metal::DataType::FLOAT32;
    default:
      return std::nullopt;
  }
}

template <typename F>
PJRT_Error* DispatchByElementType(PJRT_Buffer_Type type, F&& f) {
  switch (type) {
    case PJRT_Buffer_Type_PRED:
    case PJRT_Buffer_Type_U8:
      return f(uint8_t{});
    case PJRT_Buffer_Type_U16:
      return f(uint16_t{});
    case PJRT_Buffer_Type_S32:
      return f(int32_t{});
    case PJRT_Buffer_Type_U32:
      return f(uint32_t{});
    case PJRT_Buffer_Type_BF16:
      return f(bfloat16{});
    case PJRT_Buffer_Type_F32:
      return f(float{});
    default:
      return Unimplemented("unsupported TTNN tensor buffer type");
  }
}

PJRT_Error* ShapeFromDims(const std::vector<int64_t>& dims, tt::tt_metal::Shape* out) {
  tt::tt_metal::Shape::Container values;
  for (int64_t dim : dims) {
    if (dim < 0 || dim > std::numeric_limits<uint32_t>::max()) {
      return InvalidArgument("shape dimensions must fit uint32_t for TTNN tensors");
    }
    values.push_back(static_cast<uint32_t>(dim));
  }
  *out = tt::tt_metal::Shape(std::move(values));
  return nullptr;
}

PJRT_Error* CreateTensorSpec(PJRT_Buffer_Type type,
                             const std::vector<int64_t>& logical_dims,
                             tt::tt_metal::Layout target_layout,
                             tt::tt_metal::MemoryConfig memory_config,
                             std::optional<ttnn::TensorSpec>* out) {
  const std::optional<tt::tt_metal::DataType> dtype = MetalDataType(type);
  if (!dtype.has_value()) {
    return Unimplemented("PJRT buffer type cannot be represented as a TTNN Tensor dtype");
  }

  tt::tt_metal::Shape logical_shape;
  if (PJRT_Error* error = ShapeFromDims(logical_dims, &logical_shape)) {
    return error;
  }

  tt::tt_metal::TensorLayout tensor_layout(
      *dtype,
      tt::tt_metal::PageConfig(target_layout),
      std::move(memory_config),
      tt::tt_metal::Alignment({kTileRows, kTileCols}));
  out->emplace(logical_shape, std::move(tensor_layout));
  return nullptr;
}

template <typename T>
PJRT_Error* CopyBytesToVector(const void* data, size_t byte_size, std::vector<T>* out) {
  static_assert(std::is_trivially_copyable_v<T>, "PJRT tensor element type must be trivially copyable");
  if (byte_size % sizeof(T) != 0) {
    return InvalidArgument("host buffer byte size is not a multiple of the tensor element size");
  }
  out->resize(byte_size / sizeof(T));
  if (byte_size > 0) {
    std::memcpy(out->data(), data, byte_size);
  }
  return nullptr;
}

template <typename T>
PJRT_Error* CreateTensorFromBytes(const void* data,
                                  size_t byte_size,
                                  const ttnn::TensorSpec& spec,
                                  std::optional<ttnn::Tensor>* out) {
  std::vector<T> values;
  if (PJRT_Error* error = CopyBytesToVector(data, byte_size, &values)) {
    return error;
  }
  try {
    out->emplace(ttnn::Tensor::from_vector(std::move(values), spec));
  } catch (const std::exception& e) {
    return Internal(std::string("failed to create TTNN tensor from host buffer: ") + e.what());
  }
  return nullptr;
}

template <typename T>
void CopyVectorToBytes(const std::vector<T>& values, std::vector<std::byte>* out) {
  static_assert(std::is_trivially_copyable_v<T>, "PJRT tensor element type must be trivially copyable");
  out->resize(values.size() * sizeof(T));
  if (!values.empty()) {
    std::memcpy(out->data(), values.data(), out->size());
  }
}

template <typename T>
PJRT_Error* TensorToBytes(const ttnn::Tensor& tensor, std::vector<std::byte>* out) {
  try {
    std::vector<T> values = tensor.to_vector<T>();
    CopyVectorToBytes(values, out);
  } catch (const std::exception& e) {
    return Internal(std::string("failed to read TTNN tensor to host buffer: ") + e.what());
  }
  return nullptr;
}

template <typename T>
PJRT_Error* CopyTensorToDevice(const ttnn::Tensor& tensor,
                               const ttnn::TensorSpec& spec,
                               tt::tt_metal::distributed::MeshDevice* mesh_device,
                               ttnn::Tensor* out) {
  try {
    std::vector<T> values = tensor.to_vector<T>();
    *out = ttnn::Tensor::from_vector(std::move(values), spec, mesh_device);
  } catch (const std::exception& e) {
    return Internal(std::string("failed to copy TTNN tensor to device: ") + e.what());
  }
  return nullptr;
}

}  // namespace

size_t BytesPerElement(PJRT_Buffer_Type type) {
  switch (type) {
    case PJRT_Buffer_Type_PRED:
    case PJRT_Buffer_Type_S8:
    case PJRT_Buffer_Type_U8:
      return 1;
    case PJRT_Buffer_Type_S16:
    case PJRT_Buffer_Type_U16:
    case PJRT_Buffer_Type_F16:
    case PJRT_Buffer_Type_BF16:
      return 2;
    case PJRT_Buffer_Type_S32:
    case PJRT_Buffer_Type_U32:
    case PJRT_Buffer_Type_F32:
      return 4;
    case PJRT_Buffer_Type_S64:
    case PJRT_Buffer_Type_U64:
    case PJRT_Buffer_Type_F64:
    case PJRT_Buffer_Type_C64:
      return 8;
    case PJRT_Buffer_Type_C128:
      return 16;
    default:
      return 0;
  }
}

bool IsSupportedBufferType(PJRT_Buffer_Type type) { return BytesPerElement(type) != 0; }

std::optional<tt::tt_metal::DataType> TtnnDataTypeForPjrtBufferType(PJRT_Buffer_Type type) {
  return MetalDataType(type);
}

PJRT_Error* CopyDims(const int64_t* dims, size_t num_dims, std::vector<int64_t>* out) {
  out->clear();
  if (num_dims == 0) {
    return nullptr;
  }
  if (dims == nullptr) {
    return InvalidArgument("dims must not be null when num_dims > 0");
  }
  out->assign(dims, dims + num_dims);
  for (int64_t dim : *out) {
    if (dim < 0) {
      return InvalidArgument("shape dimensions must be >= 0");
    }
  }
  return nullptr;
}

PJRT_Error* HostByteSize(PJRT_Buffer_Type type, const std::vector<int64_t>& dims, size_t* out) {
  const size_t bytes_per_element = BytesPerElement(type);
  if (bytes_per_element == 0) {
    return Unimplemented("unsupported PJRT buffer type");
  }

  size_t elements = 1;
  for (int64_t dim_i64 : dims) {
    const size_t dim = static_cast<size_t>(dim_i64);
    if (dim != 0 && elements > std::numeric_limits<size_t>::max() / dim) {
      return ResourceExhausted("host buffer element count overflow");
    }
    elements *= dim;
  }
  if (bytes_per_element != 0 &&
      elements > std::numeric_limits<size_t>::max() / bytes_per_element) {
    return ResourceExhausted("host buffer byte size overflow");
  }
  *out = elements * bytes_per_element;
  return nullptr;
}

PJRT_Error* ValidateDenseRowMajorStrides(PJRT_Buffer_Type type,
                                         const std::vector<int64_t>& dims,
                                         const int64_t* byte_strides,
                                         size_t num_byte_strides) {
  if (byte_strides == nullptr && num_byte_strides == 0) {
    return nullptr;
  }
  if (num_byte_strides != dims.size()) {
    return InvalidArgument("num_byte_strides must match num_dims for strided host buffers");
  }
  const size_t bytes_per_element = BytesPerElement(type);
  if (bytes_per_element == 0) {
    return Unimplemented("unsupported PJRT buffer type");
  }

  size_t expected = bytes_per_element;
  for (size_t i = dims.size(); i > 0; --i) {
    const int64_t stride = byte_strides[i - 1];
    if (stride < 0) {
      return InvalidArgument("negative byte strides are not supported");
    }
    if (static_cast<size_t>(stride) != expected) {
      return Unimplemented("only dense row-major host buffers are supported");
    }
    const size_t dim = static_cast<size_t>(std::max<int64_t>(dims[i - 1], 1));
    if (dim != 0 && expected > std::numeric_limits<size_t>::max() / dim) {
      return ResourceExhausted("byte stride overflow");
    }
    expected *= dim;
  }
  return nullptr;
}

PJRT_Error* CreatePjrtBufferFromHostBytes(PJRT_Buffer_Type type,
                                          const std::vector<int64_t>& dims,
                                          PJRT_Device* target_device,
                                          PJRT_Memory* target_memory,
                                          const void* data,
                                          size_t byte_size,
                                          PJRT_Buffer** out) {
  if (out == nullptr) {
    return InvalidArgument("out must not be null");
  }
  *out = nullptr;
  if (target_device == nullptr) {
    return InvalidArgument("no target device available");
  }
  if (target_memory == nullptr) {
    return InvalidArgument("no target memory available");
  }
  if (byte_size > 0 && data == nullptr) {
    return InvalidArgument("data must not be null");
  }

  size_t expected_byte_size = 0;
  if (PJRT_Error* error = HostByteSize(type, dims, &expected_byte_size)) {
    return error;
  }
  if (byte_size != expected_byte_size) {
    return InvalidArgument("host buffer byte size does not match tensor shape");
  }

  std::optional<ttnn::TensorSpec> tensor_spec;
  if (PJRT_Error* error = CreateTensorSpec(type, dims, tt::tt_metal::Layout::ROW_MAJOR,
                                           tt::tt_metal::MemoryConfig{}, &tensor_spec)) {
    return error;
  }
  std::optional<ttnn::Tensor> tensor;
  if (PJRT_Error* error = DispatchByElementType(type, [&](auto tag) {
        using Element = decltype(tag);
        return CreateTensorFromBytes<Element>(data, byte_size, *tensor_spec, &tensor);
      })) {
    return error;
  }

  auto buffer = std::make_unique<PJRT_Buffer>();
  buffer->buffer_type = type;
  buffer->dims = dims;
  buffer->device = target_device;
  buffer->memory = target_memory;
  buffer->storage = std::make_unique<PjrtTensorStorage>(std::move(*tensor));
  buffer->deleted = false;
  buffer->external_reference_count = 0;
  *out = buffer.release();
  return nullptr;
}

PJRT_Error* CreatePjrtBufferFromTtnnTensor(PJRT_Buffer_Type type,
                                           const std::vector<int64_t>& dims,
                                           PJRT_Device* target_device,
                                           PJRT_Memory* target_memory,
                                           ttnn::Tensor tensor,
                                           PJRT_Buffer** out) {
  if (out == nullptr) {
    return InvalidArgument("out must not be null");
  }
  *out = nullptr;
  if (target_device == nullptr) {
    return InvalidArgument("no target device available");
  }
  if (target_memory == nullptr) {
    return InvalidArgument("no target memory available");
  }
  const std::optional<tt::tt_metal::DataType> expected_dtype = MetalDataType(type);
  if (!expected_dtype.has_value()) {
    return Unimplemented("PJRT buffer type cannot be represented as a TTNN Tensor dtype");
  }
  if (tensor.dtype() != *expected_dtype) {
    return InvalidArgument("TTNN tensor dtype does not match PJRT buffer type");
  }
  const tt::tt_metal::Shape& shape = tensor.logical_shape();
  if (shape.rank() != dims.size()) {
    return InvalidArgument("TTNN tensor rank does not match PJRT buffer rank");
  }
  for (size_t i = 0; i < dims.size(); ++i) {
    if (dims[i] < 0 || static_cast<uint64_t>(dims[i]) != shape[i]) {
      return InvalidArgument("TTNN tensor shape does not match PJRT buffer shape");
    }
  }

  auto buffer = std::make_unique<PJRT_Buffer>();
  buffer->buffer_type = type;
  buffer->dims = dims;
  buffer->device = target_device;
  buffer->memory = target_memory;
  buffer->storage = std::make_unique<PjrtTensorStorage>(std::move(tensor));
  buffer->deleted = false;
  buffer->external_reference_count = 0;
  *out = buffer.release();
  return nullptr;
}

ttnn::Tensor* PjrtBufferTtnnTensor(PJRT_Buffer* buffer) {
  return buffer == nullptr || buffer->storage == nullptr ? nullptr : &buffer->storage->tensor;
}

const ttnn::Tensor* PjrtBufferTtnnTensor(const PJRT_Buffer* buffer) {
  return buffer == nullptr || buffer->storage == nullptr ? nullptr : &buffer->storage->tensor;
}

PJRT_Error* CopyPjrtBufferToTtnnDeviceTensor(
    const PJRT_Buffer& buffer,
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    ttnn::Tensor* out) {
  if (mesh_device == nullptr) {
    return InvalidArgument("mesh_device must not be null");
  }
  if (out == nullptr) {
    return InvalidArgument("out must not be null");
  }
  if (buffer.deleted) {
    return FailedPrecondition("buffer has been deleted");
  }
  const ttnn::Tensor* tensor = PjrtBufferTtnnTensor(&buffer);
  if (tensor == nullptr) {
    return FailedPrecondition("buffer has no TTNN tensor storage");
  }
  if (tensor->storage_type() == tt::tt_metal::StorageType::DEVICE &&
      tensor->layout() == ttnn::TILE_LAYOUT &&
      tensor->device() == mesh_device) {
    *out = *tensor;
    return nullptr;
  }

  std::optional<ttnn::TensorSpec> tensor_spec;
  if (PJRT_Error* error = CreateTensorSpec(buffer.buffer_type, buffer.dims,
                                           tt::tt_metal::Layout::TILE,
                                           ttnn::DRAM_MEMORY_CONFIG, &tensor_spec)) {
    return error;
  }
  return DispatchByElementType(buffer.buffer_type, [&](auto tag) {
    using Element = decltype(tag);
    return CopyTensorToDevice<Element>(*tensor, *tensor_spec, mesh_device, out);
  });
}

void DeletePjrtBufferStorage(PJRT_Buffer* buffer) {
  if (buffer != nullptr) {
    buffer->storage.reset();
  }
}

PJRT_Error* ReadBufferLogicalBytes(const PJRT_Buffer& buffer, std::vector<std::byte>* out) {
  if (buffer.deleted) {
    return FailedPrecondition("buffer has been deleted");
  }
  if (buffer.storage == nullptr) {
    return FailedPrecondition("buffer has no TTNN tensor storage");
  }

  if (PJRT_Error* error = DispatchByElementType(buffer.buffer_type, [&](auto tag) {
        using Element = decltype(tag);
        return TensorToBytes<Element>(buffer.storage->tensor, out);
      })) {
    return error;
  }

  size_t expected_byte_size = 0;
  if (PJRT_Error* byte_size_error = HostByteSize(buffer.buffer_type, buffer.dims, &expected_byte_size)) {
    return byte_size_error;
  }
  if (out->size() != expected_byte_size) {
    return Internal("TTNN tensor readback byte size does not match logical byte size");
  }
  return nullptr;
}

PJRT_Error* TtnnTensorPhysicalByteSize(const PJRT_Buffer& buffer, size_t* out) {
  if (buffer.storage == nullptr) {
    return FailedPrecondition("buffer has no TTNN tensor storage");
  }
  const ttnn::Tensor& tensor = buffer.storage->tensor;
  const uint64_t physical_volume = tensor.physical_volume();
  const uint32_t element_size = tensor.element_size();
  if (element_size != 0 &&
      physical_volume > std::numeric_limits<size_t>::max() / element_size) {
    return ResourceExhausted("TTNN tensor physical byte size overflow");
  }
  *out = static_cast<size_t>(physical_volume) * static_cast<size_t>(element_size);
  return nullptr;
}
