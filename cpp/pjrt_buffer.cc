#include "cpp/pjrt_buffer.h"

#include <tt-metalium/experimental/tensor/host_tensor.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/experimental/tensor/topology/tensor_topology.hpp>
#include <tt-metalium/host_buffer.hpp>

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <utility>

struct PjrtHostBufferStorage {
  explicit PjrtHostBufferStorage(tt::tt_metal::HostTensor host_tensor)
      : host_tensor(std::move(host_tensor)) {}

  tt::tt_metal::HostTensor host_tensor;
};

PJRT_Buffer::~PJRT_Buffer() = default;

namespace {

constexpr int64_t kTileRows = 32;
constexpr int64_t kTileCols = 32;

PJRT_Error* CopyBetweenHostShapes(const std::vector<std::byte>& source,
                                  std::vector<std::byte>* target,
                                  PJRT_Buffer_Type type,
                                  const std::vector<int64_t>& source_shape,
                                  const std::vector<int64_t>& target_shape,
                                  const std::vector<int64_t>& copy_shape) {
  if (source_shape.size() != target_shape.size() || source_shape.size() != copy_shape.size()) {
    return InvalidArgument("rank must match when copying between padded host shapes");
  }
  if (copy_shape.size() < 2) {
    return InvalidArgument("padded host shape copy requires rank >= 2");
  }
  for (size_t i = 0; i < copy_shape.size(); ++i) {
    if (copy_shape[i] < 0 || copy_shape[i] > source_shape[i] || copy_shape[i] > target_shape[i]) {
      return InvalidArgument("copy shape exceeds source or target shape");
    }
  }

  size_t source_size = 0;
  size_t target_size = 0;
  if (PJRT_Error* error = HostByteSize(type, source_shape, &source_size)) {
    return error;
  }
  if (PJRT_Error* error = HostByteSize(type, target_shape, &target_size)) {
    return error;
  }
  if (source.size() != source_size || target->size() != target_size) {
    return InvalidArgument("host buffer size does not match row-major shape");
  }

  const size_t rank = copy_shape.size();
  const size_t copy_rows = static_cast<size_t>(copy_shape[rank - 2]);
  const size_t copy_cols = static_cast<size_t>(copy_shape[rank - 1]);
  const size_t source_rows = static_cast<size_t>(source_shape[rank - 2]);
  const size_t source_cols = static_cast<size_t>(source_shape[rank - 1]);
  const size_t target_rows = static_cast<size_t>(target_shape[rank - 2]);
  const size_t target_cols = static_cast<size_t>(target_shape[rank - 1]);
  size_t batch = 1;
  for (size_t i = 0; i + 2 < rank; ++i) {
    const size_t dim = static_cast<size_t>(copy_shape[i]);
    if (dim != 0 && batch > std::numeric_limits<size_t>::max() / dim) {
      return ResourceExhausted("shape dimensions overflow");
    }
    batch *= dim;
  }

  const size_t bytes_per_element = BytesPerElement(type);
  if (bytes_per_element == 0) {
    return Unimplemented("unsupported PJRT buffer type");
  }
  if (copy_cols > std::numeric_limits<size_t>::max() / bytes_per_element) {
    return ResourceExhausted("shape dimensions overflow");
  }
  const size_t row_bytes = copy_cols * bytes_per_element;

  for (size_t batch_index = 0; batch_index < batch; ++batch_index) {
    for (size_t row = 0; row < copy_rows; ++row) {
      const size_t source_element = (batch_index * source_rows + row) * source_cols;
      const size_t target_element = (batch_index * target_rows + row) * target_cols;
      const size_t source_start = source_element * bytes_per_element;
      const size_t target_start = target_element * bytes_per_element;
      std::memcpy(target->data() + target_start, source.data() + source_start, row_bytes);
    }
  }
  return nullptr;
}

PJRT_Error* PaddedHostData(const void* data, size_t byte_size, PJRT_Buffer_Type type,
                           const std::vector<int64_t>& logical_dims,
                           const std::vector<int64_t>& allocation_dims,
                           std::vector<std::byte>* out) {
  out->resize(byte_size);
  if (byte_size > 0) {
    std::memcpy(out->data(), data, byte_size);
  }
  if (logical_dims == allocation_dims) {
    return nullptr;
  }

  size_t allocation_size = 0;
  if (PJRT_Error* error = HostByteSize(type, allocation_dims, &allocation_size)) {
    return error;
  }
  if (byte_size > allocation_size) {
    return InvalidArgument("logical buffer is larger than allocation buffer");
  }
  std::vector<std::byte> padded(allocation_size);
  if (logical_dims.size() < 2) {
    if (byte_size > 0) {
      std::memcpy(padded.data(), out->data(), byte_size);
    }
    *out = std::move(padded);
    return nullptr;
  }
  if (PJRT_Error* error =
          CopyBetweenHostShapes(*out, &padded, type, logical_dims, allocation_dims, logical_dims)) {
    return error;
  }
  *out = std::move(padded);
  return nullptr;
}

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

PJRT_Error* ShapeFromDims(const std::vector<int64_t>& dims, tt::tt_metal::Shape* out) {
  tt::tt_metal::Shape::Container values;
  for (int64_t dim : dims) {
    if (dim < 0 || dim > std::numeric_limits<uint32_t>::max()) {
      return InvalidArgument("shape dimensions must fit uint32_t for tt-metal tensors");
    }
    values.push_back(static_cast<uint32_t>(dim));
  }
  *out = tt::tt_metal::Shape(std::move(values));
  return nullptr;
}

std::vector<int64_t> DimsFromShape(const tt::tt_metal::Shape& shape) {
  std::vector<int64_t> dims;
  dims.reserve(shape.size());
  for (uint32_t dim : shape) {
    dims.push_back(static_cast<int64_t>(dim));
  }
  return dims;
}

PJRT_Error* CreateTensorSpec(PJRT_Buffer_Type type, const std::vector<int64_t>& logical_dims,
                             std::optional<tt::tt_metal::TensorSpec>* out) {
  const std::optional<tt::tt_metal::DataType> dtype = MetalDataType(type);
  if (!dtype.has_value()) {
    return Unimplemented("PJRT buffer type cannot be represented as a tt-metal HostTensor dtype");
  }

  tt::tt_metal::Shape logical_shape;
  if (PJRT_Error* error = ShapeFromDims(logical_dims, &logical_shape)) {
    return error;
  }

  tt::tt_metal::TensorLayout layout(
      *dtype,
      tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
      tt::tt_metal::MemoryConfig{},
      tt::tt_metal::Alignment({static_cast<uint32_t>(kTileRows),
                               static_cast<uint32_t>(kTileCols)}));
  out->emplace(logical_shape, std::move(layout));
  return nullptr;
}

PJRT_Error* CreateHostTensor(const tt::tt_metal::TensorSpec& spec,
                             std::vector<std::byte> storage,
                             std::optional<tt::tt_metal::HostTensor>* out) {
  tt::tt_metal::HostBuffer host_buffer(std::move(storage));
  out->emplace(std::move(host_buffer), spec, tt::tt_metal::TensorTopology{});
  return nullptr;
}

PJRT_Error* HostTensorPhysicalBytes(const PJRT_Buffer& buffer, std::vector<std::byte>* out) {
  if (buffer.storage == nullptr) {
    return FailedPrecondition("buffer has no host tensor storage");
  }
  const auto shard =
      buffer.storage->host_tensor.buffer().get_shard(tt::tt_metal::distributed::MeshCoordinate(0, 0));
  if (!shard.has_value()) {
    return Internal("host tensor has no local shard at coordinate (0, 0)");
  }
  const auto bytes = shard->view_bytes();
  out->assign(bytes.begin(), bytes.end());
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

  std::optional<tt::tt_metal::TensorSpec> tensor_spec;
  if (PJRT_Error* error = CreateTensorSpec(type, dims, &tensor_spec)) {
    return error;
  }
  std::vector<int64_t> allocation_dims = DimsFromShape(tensor_spec->padded_shape());
  std::vector<std::byte> storage;
  if (PJRT_Error* error = PaddedHostData(data, byte_size, type, dims, allocation_dims, &storage)) {
    return error;
  }
  std::optional<tt::tt_metal::HostTensor> host_tensor;
  if (PJRT_Error* error = CreateHostTensor(*tensor_spec, std::move(storage), &host_tensor)) {
    return error;
  }
  auto host_storage = std::make_unique<PjrtHostBufferStorage>(std::move(*host_tensor));

  auto buffer = std::make_unique<PJRT_Buffer>();
  buffer->buffer_type = type;
  buffer->dims = dims;
  buffer->device = target_device;
  buffer->memory = target_memory;
  buffer->storage = std::move(host_storage);
  buffer->allocation_dims = std::move(allocation_dims);
  buffer->source_shape = std::nullopt;
  buffer->deleted = false;
  buffer->external_reference_count = 0;
  *out = buffer.release();
  return nullptr;
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
  size_t logical_size = 0;
  if (PJRT_Error* error = HostByteSize(buffer.buffer_type, buffer.dims, &logical_size)) {
    return error;
  }
  const std::vector<int64_t>& read_shape =
      buffer.source_shape.has_value() ? *buffer.source_shape : buffer.dims;
  size_t read_size = 0;
  if (PJRT_Error* error = HostByteSize(buffer.buffer_type, read_shape, &read_size)) {
    return error;
  }
  if (buffer.source_shape.has_value() && read_size != logical_size) {
    return Internal("reshape view source byte size does not match logical byte size");
  }
  std::vector<std::byte> physical_data;
  if (PJRT_Error* error = HostTensorPhysicalBytes(buffer, &physical_data)) {
    return error;
  }
  if (physical_data.size() == read_size) {
    *out = std::move(physical_data);
    return nullptr;
  }
  if (read_shape.size() < 2 && physical_data.size() >= read_size) {
    out->assign(physical_data.begin(), physical_data.begin() + static_cast<ptrdiff_t>(read_size));
    return nullptr;
  }
  if (read_shape.size() == buffer.allocation_dims.size() && physical_data.size() > read_size) {
    out->assign(logical_size, std::byte{0});
    return CopyBetweenHostShapes(physical_data, out, buffer.buffer_type, buffer.allocation_dims,
                                 read_shape, read_shape);
  }
  return Internal("readback byte size does not match buffer byte size");
}

PJRT_Error* HostTensorPhysicalByteSize(const PJRT_Buffer& buffer, size_t* out) {
  if (buffer.storage == nullptr) {
    return FailedPrecondition("buffer has no host tensor storage");
  }
  const auto shard =
      buffer.storage->host_tensor.buffer().get_shard(tt::tt_metal::distributed::MeshCoordinate(0, 0));
  if (!shard.has_value()) {
    return Internal("host tensor has no local shard at coordinate (0, 0)");
  }
  *out = shard->view_bytes().size();
  return nullptr;
}
