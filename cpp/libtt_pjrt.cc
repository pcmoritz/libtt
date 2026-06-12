#include "cpp/libtt_pjrt.h"

#include "mlir/executable.pb.h"

#include <tt-metalium/experimental/tensor/host_tensor.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/experimental/tensor/topology/tensor_topology.hpp>
#include <tt-metalium/host_buffer.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

extern "C" {
using TT_MlirAllocOutput = char* (*)(size_t size, void* user_data);

bool TT_MlirAnalyzeProgram(const char* format, size_t format_size, const char* code,
                           size_t code_size, TT_MlirAllocOutput alloc_output,
                           void* user_data);
}

struct PJRT_Error {
  PJRT_Error_Code code;
  std::string message;
};

struct PJRT_Event {
  bool ready;
  std::optional<std::pair<PJRT_Error_Code, std::string>> error;
};

struct PJRT_DeviceDescription {
  int id;
  int process_index;
  std::string device_kind;
  std::string debug_string;
  std::string to_string;
};

struct PJRT_Memory;

struct PJRT_Device {
  int id;
  int local_hardware_id;
  PJRT_DeviceDescription* description;
  bool addressable;
  PJRT_Memory* default_memory;
  std::vector<PJRT_Memory*> memory_ptrs;
};

struct PJRT_Memory {
  int id;
  std::string kind;
  std::string debug_string;
  std::string to_string;
  std::vector<PJRT_Device*> device_ptrs;
};

struct PJRT_TopologyDescription {
  std::string platform_name;
  std::string platform_version;
  std::vector<PJRT_DeviceDescription*> device_description_ptrs;
};

struct PJRT_Buffer {
  PJRT_Buffer_Type buffer_type;
  std::vector<int64_t> dims;
  PJRT_Device* device;
  PJRT_Memory* memory;
  std::optional<tt::tt_metal::HostTensor> host_tensor;
  std::vector<int64_t> allocation_dims;
  std::optional<std::vector<int64_t>> source_shape;
  bool deleted;
  size_t external_reference_count;
};

struct ExecutableMetadata {
  std::string name;
  std::string fingerprint;
  size_t num_outputs = 0;
  std::vector<PJRT_Buffer_Type> output_types;
  std::vector<int64_t> output_dims;
  std::vector<size_t> output_dim_sizes;
  std::vector<std::string> output_memory_kinds;
  std::vector<const char*> output_memory_kind_ptrs;
  std::vector<size_t> output_memory_kind_sizes;
  std::string executable_proto;

  ExecutableMetadata() = default;

  ExecutableMetadata(const ExecutableMetadata& other)
      : name(other.name),
        fingerprint(other.fingerprint),
        num_outputs(other.num_outputs),
        output_types(other.output_types),
        output_dims(other.output_dims),
        output_dim_sizes(other.output_dim_sizes),
        output_memory_kinds(other.output_memory_kinds),
        output_memory_kind_sizes(other.output_memory_kind_sizes),
        executable_proto(other.executable_proto) {
    RefreshMemoryKindPointers();
  }

  ExecutableMetadata& operator=(const ExecutableMetadata& other) {
    if (this == &other) {
      return *this;
    }
    name = other.name;
    fingerprint = other.fingerprint;
    num_outputs = other.num_outputs;
    output_types = other.output_types;
    output_dims = other.output_dims;
    output_dim_sizes = other.output_dim_sizes;
    output_memory_kinds = other.output_memory_kinds;
    output_memory_kind_sizes = other.output_memory_kind_sizes;
    executable_proto = other.executable_proto;
    RefreshMemoryKindPointers();
    return *this;
  }

  ExecutableMetadata(ExecutableMetadata&& other) noexcept
      : name(std::move(other.name)),
        fingerprint(std::move(other.fingerprint)),
        num_outputs(other.num_outputs),
        output_types(std::move(other.output_types)),
        output_dims(std::move(other.output_dims)),
        output_dim_sizes(std::move(other.output_dim_sizes)),
        output_memory_kinds(std::move(other.output_memory_kinds)),
        output_memory_kind_sizes(std::move(other.output_memory_kind_sizes)),
        executable_proto(std::move(other.executable_proto)) {
    RefreshMemoryKindPointers();
  }

  ExecutableMetadata& operator=(ExecutableMetadata&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    name = std::move(other.name);
    fingerprint = std::move(other.fingerprint);
    num_outputs = other.num_outputs;
    output_types = std::move(other.output_types);
    output_dims = std::move(other.output_dims);
    output_dim_sizes = std::move(other.output_dim_sizes);
    output_memory_kinds = std::move(other.output_memory_kinds);
    output_memory_kind_sizes = std::move(other.output_memory_kind_sizes);
    executable_proto = std::move(other.executable_proto);
    RefreshMemoryKindPointers();
    return *this;
  }

  void RefreshMemoryKindPointers() {
    output_memory_kind_ptrs.clear();
    output_memory_kind_ptrs.reserve(output_memory_kinds.size());
    for (const std::string& kind : output_memory_kinds) {
      output_memory_kind_ptrs.push_back(kind.data());
    }
  }
};

struct PJRT_Executable {
  ExecutableMetadata metadata;
};

struct PJRT_LoadedExecutable {
  ExecutableMetadata metadata;
  std::vector<PJRT_Device*> addressable_devices;
  bool deleted;
};

struct PJRT_ExecuteContext {};
struct PJRT_Device_Attributes {};

struct PJRT_Client {
  std::string platform_name;
  std::string platform_version;
  PJRT_TopologyDescription topology;
  std::vector<PJRT_DeviceDescription> device_descriptions_storage;
  std::vector<PJRT_Memory> memories_storage;
  std::vector<PJRT_Device> devices_storage;
  std::vector<PJRT_Device*> device_ptrs;
  std::vector<PJRT_Device*> addressable_device_ptrs;
  std::vector<PJRT_Memory*> memory_ptrs;
};

namespace {

constexpr const char* kPlatformName = "tt";
constexpr const char* kPlatformVersion = "libtt cpp pjrt 0.1.0";
constexpr const char* kDefaultDeviceRoot = "/dev/tenstorrent";
constexpr const char* kExecutableName = "tt.executable.v1";
constexpr int64_t kTileRows = 32;
constexpr int64_t kTileCols = 32;

PJRT_Error* MakePjrtError(PJRT_Error_Code code, std::string message) {
  return new PJRT_Error{code, std::move(message)};
}

PJRT_Error* InvalidArgument(std::string message) {
  return MakePjrtError(PJRT_Error_Code_INVALID_ARGUMENT, std::move(message));
}

PJRT_Error* Unimplemented(std::string message) {
  return MakePjrtError(PJRT_Error_Code_UNIMPLEMENTED, std::move(message));
}

PJRT_Error* FailedPrecondition(std::string message) {
  return MakePjrtError(PJRT_Error_Code_FAILED_PRECONDITION, std::move(message));
}

PJRT_Error* ResourceExhausted(std::string message) {
  return MakePjrtError(PJRT_Error_Code_RESOURCE_EXHAUSTED, std::move(message));
}

PJRT_Error* Internal(std::string message) {
  return MakePjrtError(PJRT_Error_Code_INTERNAL, std::move(message));
}

PJRT_Error* CloneEventError(const PJRT_Event* event) {
  if (event == nullptr || !event->error.has_value()) {
    return nullptr;
  }
  return MakePjrtError(event->error->first, event->error->second);
}

PJRT_Event* ReadyEvent() { return new PJRT_Event{true, std::nullopt}; }

PJRT_Event* EventWithError(PJRT_Error_Code code, std::string message) {
  return new PJRT_Event{true, std::make_pair(code, std::move(message))};
}

std::string CopyString(std::string_view value) { return std::string(value.data(), value.size()); }

void SetStringOut(const std::string& value, const char** out, size_t* out_size) {
  *out = value.data();
  *out_size = value.size();
}

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

PJRT_Error* ValidateDenseRowMajorStrides(PJRT_Buffer_Type type, const std::vector<int64_t>& dims,
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

PJRT_Error* RoundUpToTileDim(int64_t value, int64_t tile_dim, int64_t* out) {
  if (value < 0) {
    return InvalidArgument("shape dimensions must be >= 0");
  }
  const int64_t normalized = std::max<int64_t>(value, 1);
  if (normalized > std::numeric_limits<int64_t>::max() - (tile_dim - 1)) {
    return ResourceExhausted("shape dimension overflow");
  }
  *out = ((normalized + tile_dim - 1) / tile_dim) * tile_dim;
  return nullptr;
}

PJRT_Error* TiledAllocationDims(const std::vector<int64_t>& logical_dims,
                                std::vector<int64_t>* allocation_dims) {
  allocation_dims->clear();
  if (logical_dims.empty()) {
    *allocation_dims = {kTileRows, kTileCols};
    return nullptr;
  }
  if (logical_dims.size() == 1) {
    int64_t cols = 0;
    if (PJRT_Error* error = RoundUpToTileDim(logical_dims[0], kTileCols, &cols)) {
      return error;
    }
    *allocation_dims = {kTileRows, cols};
    return nullptr;
  }
  *allocation_dims = logical_dims;
  const size_t rank = allocation_dims->size();
  if (PJRT_Error* error =
          RoundUpToTileDim((*allocation_dims)[rank - 2], kTileRows, &(*allocation_dims)[rank - 2])) {
    return error;
  }
  if (PJRT_Error* error =
          RoundUpToTileDim((*allocation_dims)[rank - 1], kTileCols, &(*allocation_dims)[rank - 1])) {
    return error;
  }
  return nullptr;
}

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

PJRT_Error* CreateHostTensor(PJRT_Buffer_Type type, const std::vector<int64_t>& logical_dims,
                             const std::vector<int64_t>& allocation_dims,
                             std::vector<std::byte> storage,
                             std::optional<tt::tt_metal::HostTensor>* out) {
  const std::optional<tt::tt_metal::DataType> dtype = MetalDataType(type);
  if (!dtype.has_value()) {
    return Unimplemented("PJRT buffer type cannot be represented as a tt-metal HostTensor dtype");
  }

  tt::tt_metal::Shape logical_shape;
  if (PJRT_Error* error = ShapeFromDims(logical_dims, &logical_shape)) {
    return error;
  }
  tt::tt_metal::Shape padded_shape;
  if (PJRT_Error* error = ShapeFromDims(allocation_dims, &padded_shape)) {
    return error;
  }

  tt::tt_metal::TensorLayout layout = tt::tt_metal::TensorLayout::fromPaddedShape(
      *dtype,
      tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
      tt::tt_metal::MemoryConfig{},
      logical_shape,
      padded_shape);
  tt::tt_metal::TensorSpec spec(logical_shape, std::move(layout));
  tt::tt_metal::HostBuffer host_buffer(std::move(storage));
  out->emplace(std::move(host_buffer), std::move(spec), tt::tt_metal::TensorTopology{});
  return nullptr;
}

PJRT_Error* HostTensorPhysicalBytes(const PJRT_Buffer& buffer, std::vector<std::byte>* out) {
  if (!buffer.host_tensor.has_value()) {
    return FailedPrecondition("buffer has no host tensor storage");
  }
  const auto shard = buffer.host_tensor->buffer().get_shard(tt::tt_metal::distributed::MeshCoordinate(0, 0));
  if (!shard.has_value()) {
    return Internal("host tensor has no local shard at coordinate (0, 0)");
  }
  const auto bytes = shard->view_bytes();
  out->assign(bytes.begin(), bytes.end());
  return nullptr;
}

PJRT_Error* HostTensorPhysicalByteSize(const PJRT_Buffer& buffer, size_t* out) {
  if (!buffer.host_tensor.has_value()) {
    return FailedPrecondition("buffer has no host tensor storage");
  }
  const auto shard = buffer.host_tensor->buffer().get_shard(tt::tt_metal::distributed::MeshCoordinate(0, 0));
  if (!shard.has_value()) {
    return Internal("host tensor has no local shard at coordinate (0, 0)");
  }
  *out = shard->view_bytes().size();
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
  if (byte_size > 0 && data == nullptr) {
    return InvalidArgument("data must not be null");
  }

  std::vector<int64_t> allocation_dims;
  if (PJRT_Error* error = TiledAllocationDims(dims, &allocation_dims)) {
    return error;
  }
  std::vector<std::byte> storage;
  if (PJRT_Error* error = PaddedHostData(data, byte_size, type, dims, allocation_dims, &storage)) {
    return error;
  }
  std::optional<tt::tt_metal::HostTensor> host_tensor;
  if (PJRT_Error* error = CreateHostTensor(type, dims, allocation_dims, std::move(storage),
                                           &host_tensor)) {
    return error;
  }

  *out = new PJRT_Buffer{
      type,
      dims,
      target_device,
      target_memory != nullptr ? target_memory : target_device->default_memory,
      std::move(host_tensor),
      std::move(allocation_dims),
      std::nullopt,
      false,
      0,
  };
  return nullptr;
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

PJRT_Buffer_Type PjrtBufferTypeFromProto(tt::TensorDesc::ElementType type) {
  switch (type) {
    case tt::TensorDesc::ELEMENT_TYPE_BF16:
      return PJRT_Buffer_Type_BF16;
    case tt::TensorDesc::ELEMENT_TYPE_F16:
      return PJRT_Buffer_Type_F16;
    case tt::TensorDesc::ELEMENT_TYPE_F32:
      return PJRT_Buffer_Type_F32;
    case tt::TensorDesc::ELEMENT_TYPE_U32:
      return PJRT_Buffer_Type_U32;
    case tt::TensorDesc::ELEMENT_TYPE_U16:
      return PJRT_Buffer_Type_U16;
    case tt::TensorDesc::ELEMENT_TYPE_U8:
      return PJRT_Buffer_Type_U8;
    case tt::TensorDesc::ELEMENT_TYPE_S32:
      return PJRT_Buffer_Type_S32;
    case tt::TensorDesc::ELEMENT_TYPE_S8:
      return PJRT_Buffer_Type_S8;
    case tt::TensorDesc::ELEMENT_TYPE_PRED:
      return PJRT_Buffer_Type_PRED;
    default:
      return PJRT_Buffer_Type_INVALID;
  }
}

PJRT_Error* TensorDescDims(const tt::TensorDesc& tensor, std::vector<int64_t>* out) {
  out->clear();
  out->reserve(static_cast<size_t>(tensor.dims_size()));
  for (int64_t dim : tensor.dims()) {
    if (dim < 0) {
      return InvalidArgument("executable tensor shape dimensions must be >= 0");
    }
    out->push_back(dim);
  }
  return nullptr;
}

PJRT_Error* TensorDescBufferType(const tt::TensorDesc& tensor, PJRT_Buffer_Type* out) {
  *out = PjrtBufferTypeFromProto(tensor.element_type());
  if (*out == PJRT_Buffer_Type_INVALID) {
    return Unimplemented("executable tensor has unsupported element type");
  }
  return nullptr;
}

bool TensorDescsMatch(const tt::TensorDesc& lhs, const tt::TensorDesc& rhs) {
  if (lhs.element_type() != rhs.element_type() || lhs.dims_size() != rhs.dims_size()) {
    return false;
  }
  for (int i = 0; i < lhs.dims_size(); ++i) {
    if (lhs.dims(i) != rhs.dims(i)) {
      return false;
    }
  }
  return true;
}

std::string FingerprintString(const std::vector<PJRT_Buffer_Type>& output_types,
                              const std::vector<int64_t>& output_dims,
                              const std::vector<size_t>& output_dim_sizes) {
  std::ostringstream fingerprint;
  fingerprint << "tt:executable_v1:name=" << kExecutableName << ":outputs=";
  size_t dim_offset = 0;
  for (size_t i = 0; i < output_types.size(); ++i) {
    if (i != 0) {
      fingerprint << ",";
    }
    fingerprint << static_cast<uint32_t>(output_types[i]) << ":";
    for (size_t d = 0; d < output_dim_sizes[i]; ++d) {
      if (d != 0) {
        fingerprint << "x";
      }
      fingerprint << output_dims[dim_offset + d];
    }
    dim_offset += output_dim_sizes[i];
  }
  fingerprint << ":v1";
  return fingerprint.str();
}

ExecutableMetadata MakeExecutableMetadata(const tt::AnalysisResult& analysis) {
  ExecutableMetadata metadata;
  metadata.name = kExecutableName;
  metadata.num_outputs = static_cast<size_t>(analysis.outputs_size());
  metadata.output_memory_kinds.assign(metadata.num_outputs, "device");
  metadata.output_memory_kind_sizes.assign(metadata.num_outputs, std::string_view("device").size());
  metadata.output_types.reserve(metadata.num_outputs);
  metadata.output_dim_sizes.reserve(metadata.num_outputs);
  for (const tt::TensorDesc& tensor : analysis.outputs()) {
    metadata.output_types.push_back(PjrtBufferTypeFromProto(tensor.element_type()));
    metadata.output_dim_sizes.push_back(static_cast<size_t>(tensor.dims_size()));
    for (int64_t dim : tensor.dims()) {
      metadata.output_dims.push_back(dim);
    }
  }
  metadata.fingerprint =
      FingerprintString(metadata.output_types, metadata.output_dims, metadata.output_dim_sizes);
  if (analysis.has_executable()) {
    metadata.executable_proto = analysis.executable().SerializeAsString();
  }
  metadata.RefreshMemoryKindPointers();
  return metadata;
}

char* AllocateMlirAnalysisOutput(size_t size, void* user_data) {
  auto* output = static_cast<std::vector<char>*>(user_data);
  output->resize(size);
  return output->data();
}

PJRT_Error* AnalyzeProgramToMetadata(const PJRT_Program& program, ExecutableMetadata* metadata) {
  if (program.format == nullptr && program.format_size != 0) {
    return InvalidArgument("program.format must not be null when size > 0");
  }
  if (program.code == nullptr && program.code_size != 0) {
    return InvalidArgument("program.code must not be null when size > 0");
  }
  const std::string_view format(program.format == nullptr ? "" : program.format, program.format_size);
  if (format != "mlir" && format != "stablehlo") {
    return Unimplemented("unsupported program format; supported formats are \"mlir\" and \"stablehlo\"");
  }

  std::vector<char> serialized_analysis;
  if (!TT_MlirAnalyzeProgram(program.format, program.format_size, program.code, program.code_size,
                             AllocateMlirAnalysisOutput, &serialized_analysis)) {
    return Internal("MLIR analysis failed without serialized diagnostics");
  }

  tt::AnalysisResult analysis;
  if (!analysis.ParseFromArray(serialized_analysis.data(),
                               static_cast<int>(serialized_analysis.size()))) {
    return Internal("failed to parse MLIR analysis result");
  }
  if (analysis.status() != tt::AnalysisResult::STATUS_OK) {
    std::string message = analysis.error_message();
    if (message.empty()) {
      message = "MLIR analysis failed";
    }
    switch (analysis.status()) {
      case tt::AnalysisResult::STATUS_PARSE_ERROR:
        return InvalidArgument(std::move(message));
      case tt::AnalysisResult::STATUS_UNSUPPORTED:
        return Unimplemented(std::move(message));
      default:
        return Internal(std::move(message));
    }
  }

  *metadata = MakeExecutableMetadata(analysis);
  return nullptr;
}

std::vector<int> DiscoverDeviceIds() {
  std::vector<int> ids;
  const std::filesystem::path root(kDefaultDeviceRoot);
  std::error_code ec;
  if (!std::filesystem::exists(root, ec) || !std::filesystem::is_directory(root, ec)) {
    return ids;
  }

  for (const auto& entry : std::filesystem::directory_iterator(root, ec)) {
    if (ec) {
      break;
    }
    const std::string name = entry.path().filename().string();
    if (name.empty() ||
        !std::all_of(name.begin(), name.end(), [](unsigned char c) { return std::isdigit(c); })) {
      continue;
    }
    try {
      ids.push_back(std::stoi(name));
    } catch (...) {
    }
  }
  std::sort(ids.begin(), ids.end());
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  return ids;
}

bool HostFallbackEnabled() {
  const char* value = std::getenv("LIBTT_PJRT_HOST_FALLBACK");
  return value != nullptr && std::string_view(value) == "1";
}

PJRT_Client* CreateClient() {
  auto* client = new PJRT_Client;
  client->platform_name = kPlatformName;
  client->platform_version = kPlatformVersion;

  std::vector<int> discovered_ids = DiscoverDeviceIds();
  const bool host_fallback = discovered_ids.empty() && HostFallbackEnabled();
  if (host_fallback) {
    discovered_ids.push_back(0);
  }
  client->device_descriptions_storage.reserve(discovered_ids.size());
  client->memories_storage.reserve(discovered_ids.size());
  client->devices_storage.reserve(discovered_ids.size());

  for (int device_id : discovered_ids) {
    const std::string suffix = std::to_string(device_id);
    if (host_fallback) {
      client->device_descriptions_storage.push_back(PJRT_DeviceDescription{
          device_id,
          0,
          "Tenstorrent host fallback",
          "Tenstorrent host fallback device " + suffix,
          "TTHostFallbackDevice(id=" + suffix + ")",
      });
      client->memories_storage.push_back(PJRT_Memory{
          device_id,
          "device",
          "Tenstorrent host fallback memory " + suffix,
          "TTHostFallbackMemory(id=" + suffix + ")",
          {},
      });
    } else {
      client->device_descriptions_storage.push_back(PJRT_DeviceDescription{
          device_id,
          0,
          "Tenstorrent",
          "Tenstorrent device /dev/tenstorrent/" + suffix,
          "TTDevice(id=" + suffix + ")",
      });
      client->memories_storage.push_back(PJRT_Memory{
          device_id,
          "device",
          "Tenstorrent device memory " + suffix,
          "TTMemory(id=" + suffix + ")",
          {},
      });
    }
  }

  for (size_t i = 0; i < discovered_ids.size(); ++i) {
    client->devices_storage.push_back(PJRT_Device{
        discovered_ids[i],
        discovered_ids[i],
        &client->device_descriptions_storage[i],
        true,
        &client->memories_storage[i],
        {&client->memories_storage[i]},
    });
  }

  for (auto& memory : client->memories_storage) {
    client->memory_ptrs.push_back(&memory);
  }

  for (size_t i = 0; i < client->devices_storage.size(); ++i) {
    PJRT_Device* device = &client->devices_storage[i];
    client->device_ptrs.push_back(device);
    client->addressable_device_ptrs.push_back(device);
    client->memories_storage[i].device_ptrs.push_back(device);
  }

  client->topology.platform_name = client->platform_name;
  client->topology.platform_version = client->platform_version;
  for (auto& description : client->device_descriptions_storage) {
    client->topology.device_description_ptrs.push_back(&description);
  }

  return client;
}

PJRT_Device* SelectTargetDevice(PJRT_Client* client, PJRT_Device* device, PJRT_Memory* memory) {
  if (device != nullptr) {
    return device;
  }
  if (memory != nullptr && !memory->device_ptrs.empty()) {
    return memory->device_ptrs.front();
  }
  if (client != nullptr && !client->addressable_device_ptrs.empty()) {
    return client->addressable_device_ptrs.front();
  }
  return nullptr;
}

PJRT_Device* ExecuteTargetDevice(const PJRT_LoadedExecutable* executable,
                                 PJRT_LoadedExecutable_Execute_Args* args) {
  if (args->execute_device != nullptr) {
    return args->execute_device;
  }
  if (!executable->addressable_devices.empty()) {
    return executable->addressable_devices.front();
  }
  if (args->argument_lists != nullptr && args->num_devices > 0 && args->num_args > 0 &&
      args->argument_lists[0] != nullptr && args->argument_lists[0][0] != nullptr) {
    return args->argument_lists[0][0]->device;
  }
  return nullptr;
}

std::string OpKindName(tt::Op::KindCase kind) {
  switch (kind) {
    case tt::Op::kParameter:
      return "parameter";
    case tt::Op::kMatmul:
      return "matmul";
    case tt::Op::kConstant:
      return "constant";
    case tt::Op::kSelect:
      return "select";
    case tt::Op::kBroadcastInDim:
      return "broadcast_in_dim";
    case tt::Op::kGather:
      return "gather";
    case tt::Op::kIota:
      return "iota";
    case tt::Op::kConcatenate:
      return "concatenate";
    case tt::Op::kReduce:
      return "reduce";
    case tt::Op::kReshape:
      return "reshape";
    case tt::Op::kSlice:
      return "slice";
    case tt::Op::kTranspose:
      return "transpose";
    case tt::Op::kCustomCall:
      return "custom_call";
    case tt::Op::kTopK:
      return "top_k";
    case tt::Op::kFusedElementwise:
      return "fused_elementwise";
    case tt::Op::kScatter:
      return "scatter";
    case tt::Op::kBitwiseBinary:
      return "bitwise_binary";
    case tt::Op::kReduceWindow:
      return "reduce_window";
    case tt::Op::kSdpaDecode:
      return "sdpa_decode";
    case tt::Op::kRmsNorm:
      return "rms_norm";
    case tt::Op::kRope:
      return "rope";
    case tt::Op::KIND_NOT_SET:
      return "not_set";
  }
  return "unknown(" + std::to_string(static_cast<int>(kind)) + ")";
}

PJRT_Error* ExecuteParameterOnlyProgram(const PJRT_LoadedExecutable* executable,
                                        PJRT_Buffer* const* arguments,
                                        size_t num_args,
                                        PJRT_Device* target_device,
                                        PJRT_Buffer** outputs) {
  tt::Executable program;
  if (executable->metadata.executable_proto.empty() ||
      !program.ParseFromString(executable->metadata.executable_proto)) {
    return Internal("failed to parse executable payload");
  }
  if (program.output_ids_size() != static_cast<int>(executable->metadata.num_outputs)) {
    return Internal("executable output metadata does not match executable payload");
  }

  std::vector<PJRT_Buffer*> values(static_cast<size_t>(program.values_size()), nullptr);
  for (const tt::Op& op : program.ops()) {
    if (op.output_id() >= static_cast<uint32_t>(values.size())) {
      return Internal("executable op output id is out of bounds");
    }
    switch (op.kind_case()) {
      case tt::Op::kParameter: {
        const size_t parameter_index = static_cast<size_t>(op.parameter().parameter_index());
        if (parameter_index >= num_args) {
          return InvalidArgument("executable parameter index exceeds supplied arguments");
        }
        if (arguments == nullptr || arguments[parameter_index] == nullptr) {
          return InvalidArgument("argument buffer must not be null");
        }
        PJRT_Buffer* argument = arguments[parameter_index];
        if (argument->deleted) {
          return FailedPrecondition("argument buffer has been deleted");
        }
        if (target_device != nullptr && argument->device != nullptr && argument->device != target_device) {
          return InvalidArgument("all input buffers and execute_device must be on the same device");
        }
        values[op.output_id()] = argument;
        break;
      }
      case tt::Op::KIND_NOT_SET:
        return Internal("executable op is missing kind");
      case tt::Op::kCustomCall: {
        const tt::CustomCallOp& custom_call = op.custom_call();
        const bool is_jax_result_sharding =
            custom_call.call_target_name() == "xla.sdy.FuncResultSharding";
        if ((custom_call.has_side_effect() && !is_jax_result_sharding) ||
            custom_call.input_ids_size() != 1) {
          return Unimplemented("C++ host execution does not support custom_call: " +
                               custom_call.call_target_name() + " inputs=" +
                               std::to_string(custom_call.input_ids_size()) + " side_effect=" +
                               (custom_call.has_side_effect() ? "true" : "false"));
        }
        const uint32_t input_id = custom_call.input_ids(0);
        if (input_id >= static_cast<uint32_t>(values.size()) || values[input_id] == nullptr) {
          return Internal("custom_call input value was not produced");
        }
        if (input_id >= static_cast<uint32_t>(program.values_size())) {
          return Internal("custom_call input metadata id is out of bounds");
        }
        const tt::ValueDesc& input_desc = program.values(input_id);
        const tt::ValueDesc& output_desc = program.values(op.output_id());
        if (!input_desc.has_tensor() || !output_desc.has_tensor() ||
            !TensorDescsMatch(input_desc.tensor(), output_desc.tensor())) {
          return Unimplemented("C++ host execution only supports identity custom_call ops");
        }
        values[op.output_id()] = values[input_id];
        break;
      }
      default:
        return Unimplemented("C++ host execution does not support op kind: " +
                             OpKindName(op.kind_case()));
    }
  }

  for (int i = 0; i < program.output_ids_size(); ++i) {
    const uint32_t output_id = program.output_ids(i);
    if (output_id >= static_cast<uint32_t>(values.size()) || values[output_id] == nullptr) {
      return Internal("executable output value was not produced");
    }
    const tt::ValueDesc& value_desc = program.values(output_id);
    if (!value_desc.has_tensor()) {
      return Internal("executable output value is missing tensor metadata");
    }
    PJRT_Buffer_Type output_type = PJRT_Buffer_Type_INVALID;
    if (PJRT_Error* error = TensorDescBufferType(value_desc.tensor(), &output_type)) {
      return error;
    }
    std::vector<int64_t> output_dims;
    if (PJRT_Error* error = TensorDescDims(value_desc.tensor(), &output_dims)) {
      return error;
    }

    size_t output_size = 0;
    if (PJRT_Error* error = HostByteSize(output_type, output_dims, &output_size)) {
      return error;
    }
    std::vector<std::byte> logical_data;
    if (PJRT_Error* error = ReadBufferLogicalBytes(*values[output_id], &logical_data)) {
      return error;
    }
    if (logical_data.size() != output_size) {
      return Internal("identity output byte size does not match executable metadata");
    }
    const void* data = logical_data.empty() ? nullptr : logical_data.data();
    PJRT_Device* output_device = target_device != nullptr ? target_device : values[output_id]->device;
    PJRT_Memory* output_memory =
        values[output_id]->memory != nullptr ? values[output_id]->memory
                                             : (output_device == nullptr ? nullptr : output_device->default_memory);
    if (PJRT_Error* error = CreatePjrtBufferFromHostBytes(output_type, output_dims, output_device,
                                                          output_memory, data, logical_data.size(),
                                                          &outputs[i])) {
      return error;
    }
  }
  return nullptr;
}

}  // namespace

extern "C" void TT_Error_Destroy(PJRT_Error_Destroy_Args* args) {
  if (args == nullptr || args->error == nullptr) {
    return;
  }
  delete args->error;
  args->error = nullptr;
}

extern "C" void TT_Error_Message(PJRT_Error_Message_Args* args) {
  if (args == nullptr) {
    return;
  }
  if (args->error == nullptr) {
    args->message = nullptr;
    args->message_size = 0;
    return;
  }
  args->message = args->error->message.data();
  args->message_size = args->error->message.size();
}

extern "C" PJRT_Error* TT_Error_GetCode(PJRT_Error_GetCode_Args* args) {
  if (args == nullptr) {
    return InvalidArgument("args must not be null");
  }
  args->code = args->error == nullptr ? PJRT_Error_Code_OK : args->error->code;
  return nullptr;
}

extern "C" PJRT_Error* TT_Error_ForEachPayload(PJRT_Error_ForEachPayload_Args* args) {
  if (args == nullptr) {
    return InvalidArgument("args must not be null");
  }
  return nullptr;
}

extern "C" PJRT_Error* TT_Plugin_Initialize(PJRT_Plugin_Initialize_Args* args) {
  return args == nullptr ? InvalidArgument("args must not be null") : nullptr;
}

extern "C" PJRT_Error* TT_Plugin_Attributes(PJRT_Plugin_Attributes_Args* args) {
  if (args == nullptr) {
    return InvalidArgument("args must not be null");
  }
  args->attributes = nullptr;
  args->num_attributes = 0;
  return nullptr;
}

extern "C" PJRT_Error* TT_Event_Destroy(PJRT_Event_Destroy_Args* args) {
  if (args == nullptr) {
    return InvalidArgument("args must not be null");
  }
  delete args->event;
  args->event = nullptr;
  return nullptr;
}

extern "C" PJRT_Error* TT_Event_IsReady(PJRT_Event_IsReady_Args* args) {
  if (args == nullptr || args->event == nullptr) {
    return InvalidArgument("event must not be null");
  }
  args->is_ready = args->event->ready;
  return nullptr;
}

extern "C" PJRT_Error* TT_Event_Error(PJRT_Event_Error_Args* args) {
  if (args == nullptr || args->event == nullptr) {
    return InvalidArgument("event must not be null");
  }
  if (!args->event->ready) {
    return FailedPrecondition("event is not ready");
  }
  return CloneEventError(args->event);
}

extern "C" PJRT_Error* TT_Event_Await(PJRT_Event_Await_Args* args) {
  if (args == nullptr || args->event == nullptr) {
    return InvalidArgument("event must not be null");
  }
  args->event->ready = true;
  return CloneEventError(args->event);
}

extern "C" PJRT_Error* TT_Event_OnReady(PJRT_Event_OnReady_Args* args) {
  if (args == nullptr || args->event == nullptr) {
    return InvalidArgument("event must not be null");
  }
  if (args->callback == nullptr) {
    return InvalidArgument("callback must not be null");
  }
  args->event->ready = true;
  args->callback(CloneEventError(args->event), args->user_arg);
  return nullptr;
}

extern "C" PJRT_Error* TT_Event_Create(PJRT_Event_Create_Args* args) {
  if (args == nullptr) {
    return InvalidArgument("args must not be null");
  }
  args->event = new PJRT_Event{false, std::nullopt};
  return nullptr;
}

extern "C" PJRT_Error* TT_Event_Set(PJRT_Event_Set_Args* args) {
  if (args == nullptr || args->event == nullptr) {
    return InvalidArgument("event must not be null");
  }
  args->event->ready = true;
  if (args->error_code == PJRT_Error_Code_OK) {
    args->event->error.reset();
  } else {
    args->event->error = std::make_pair(
        args->error_code,
        args->error_message == nullptr ? std::string()
                                       : CopyString({args->error_message, args->error_message_size}));
  }
  return nullptr;
}

extern "C" PJRT_Error* TT_Client_Create(PJRT_Client_Create_Args* args) {
  if (args == nullptr) {
    return InvalidArgument("args must not be null");
  }
  args->client = CreateClient();
  return nullptr;
}

extern "C" PJRT_Error* TT_Client_Destroy(PJRT_Client_Destroy_Args* args) {
  if (args == nullptr) {
    return InvalidArgument("args must not be null");
  }
  delete args->client;
  args->client = nullptr;
  return nullptr;
}

extern "C" PJRT_Error* TT_Client_PlatformName(PJRT_Client_PlatformName_Args* args) {
  if (args == nullptr || args->client == nullptr) {
    return InvalidArgument("client must not be null");
  }
  SetStringOut(args->client->platform_name, &args->platform_name, &args->platform_name_size);
  return nullptr;
}

extern "C" PJRT_Error* TT_Client_ProcessIndex(PJRT_Client_ProcessIndex_Args* args) {
  if (args == nullptr || args->client == nullptr) {
    return InvalidArgument("client must not be null");
  }
  args->process_index = 0;
  return nullptr;
}

extern "C" PJRT_Error* TT_Client_PlatformVersion(PJRT_Client_PlatformVersion_Args* args) {
  if (args == nullptr || args->client == nullptr) {
    return InvalidArgument("client must not be null");
  }
  SetStringOut(args->client->platform_version, &args->platform_version,
               &args->platform_version_size);
  return nullptr;
}

extern "C" PJRT_Error* TT_Client_TopologyDescription(
    PJRT_Client_TopologyDescription_Args* args) {
  if (args == nullptr || args->client == nullptr) {
    return InvalidArgument("client must not be null");
  }
  args->topology = &args->client->topology;
  return nullptr;
}

extern "C" PJRT_Error* TT_Client_Devices(PJRT_Client_Devices_Args* args) {
  if (args == nullptr || args->client == nullptr) {
    return InvalidArgument("client must not be null");
  }
  args->devices = args->client->device_ptrs.empty() ? nullptr : args->client->device_ptrs.data();
  args->num_devices = args->client->device_ptrs.size();
  return nullptr;
}

extern "C" PJRT_Error* TT_Client_AddressableDevices(
    PJRT_Client_AddressableDevices_Args* args) {
  if (args == nullptr || args->client == nullptr) {
    return InvalidArgument("client must not be null");
  }
  args->addressable_devices = args->client->addressable_device_ptrs.empty()
                                  ? nullptr
                                  : args->client->addressable_device_ptrs.data();
  args->num_addressable_devices = args->client->addressable_device_ptrs.size();
  return nullptr;
}

extern "C" PJRT_Error* TT_Client_LookupDevice(PJRT_Client_LookupDevice_Args* args) {
  if (args == nullptr || args->client == nullptr) {
    return InvalidArgument("client must not be null");
  }
  args->device = nullptr;
  for (PJRT_Device* device : args->client->device_ptrs) {
    if (device->id == args->id) {
      args->device = device;
      break;
    }
  }
  return nullptr;
}

extern "C" PJRT_Error* TT_Client_LookupAddressableDevice(
    PJRT_Client_LookupAddressableDevice_Args* args) {
  if (args == nullptr || args->client == nullptr) {
    return InvalidArgument("client must not be null");
  }
  args->addressable_device = nullptr;
  for (PJRT_Device* device : args->client->addressable_device_ptrs) {
    if (device->local_hardware_id == args->local_hardware_id) {
      args->addressable_device = device;
      break;
    }
  }
  return nullptr;
}

extern "C" PJRT_Error* TT_Client_AddressableMemories(
    PJRT_Client_AddressableMemories_Args* args) {
  if (args == nullptr || args->client == nullptr) {
    return InvalidArgument("client must not be null");
  }
  args->addressable_memories =
      args->client->memory_ptrs.empty() ? nullptr : args->client->memory_ptrs.data();
  args->num_addressable_memories = args->client->memory_ptrs.size();
  return nullptr;
}

extern "C" PJRT_Error* TT_Client_Compile(PJRT_Client_Compile_Args* args) {
  if (args == nullptr || args->client == nullptr) {
    return InvalidArgument("client must not be null");
  }
  if (args->program == nullptr) {
    return InvalidArgument("program must not be null");
  }
  ExecutableMetadata metadata;
  if (PJRT_Error* error = AnalyzeProgramToMetadata(*args->program, &metadata)) {
    return error;
  }
  args->executable = new PJRT_LoadedExecutable{
      std::move(metadata),
      args->client->addressable_device_ptrs,
      false,
  };
  return nullptr;
}

extern "C" PJRT_Error* TT_Compile(PJRT_Compile_Args* args) {
  if (args == nullptr || args->program == nullptr) {
    return InvalidArgument("program must not be null");
  }
  ExecutableMetadata metadata;
  if (PJRT_Error* error = AnalyzeProgramToMetadata(*args->program, &metadata)) {
    return error;
  }
  args->executable = new PJRT_Executable{std::move(metadata)};
  return nullptr;
}

extern "C" PJRT_Error* TT_Client_DefaultDeviceAssignment(
    PJRT_Client_DefaultDeviceAssignment_Args* args) {
  if (args == nullptr || args->client == nullptr) {
    return InvalidArgument("client must not be null");
  }
  if (args->num_replicas < 0 || args->num_partitions < 0) {
    return InvalidArgument("num_replicas and num_partitions must be >= 0");
  }
  const size_t replicas = static_cast<size_t>(args->num_replicas);
  const size_t partitions = static_cast<size_t>(args->num_partitions);
  if (replicas != 0 && partitions > std::numeric_limits<size_t>::max() / replicas) {
    return ResourceExhausted("device assignment size overflow");
  }
  const size_t required = replicas * partitions;
  if (args->default_assignment_size < required) {
    return InvalidArgument("default_assignment buffer is too small");
  }
  if (required > 0 && args->default_assignment == nullptr) {
    return InvalidArgument("default_assignment must not be null");
  }
  if (required > args->client->device_ptrs.size()) {
    return InvalidArgument("not enough devices for requested assignment");
  }
  for (size_t i = 0; i < required; ++i) {
    args->default_assignment[i] = static_cast<int>(i);
  }
  return nullptr;
}

extern "C" PJRT_Error* TT_Executable_Destroy(PJRT_Executable_Destroy_Args* args) {
  if (args == nullptr) {
    return InvalidArgument("args must not be null");
  }
  delete args->executable;
  args->executable = nullptr;
  return nullptr;
}

extern "C" PJRT_Error* TT_Executable_Name(PJRT_Executable_Name_Args* args) {
  if (args == nullptr || args->executable == nullptr) {
    return InvalidArgument("executable must not be null");
  }
  SetStringOut(args->executable->metadata.name, &args->executable_name,
               &args->executable_name_size);
  return nullptr;
}

extern "C" PJRT_Error* TT_Executable_NumReplicas(
    PJRT_Executable_NumReplicas_Args* args) {
  if (args == nullptr || args->executable == nullptr) {
    return InvalidArgument("executable must not be null");
  }
  args->num_replicas = 1;
  return nullptr;
}

extern "C" PJRT_Error* TT_Executable_NumPartitions(
    PJRT_Executable_NumPartitions_Args* args) {
  if (args == nullptr || args->executable == nullptr) {
    return InvalidArgument("executable must not be null");
  }
  args->num_partitions = 1;
  return nullptr;
}

extern "C" PJRT_Error* TT_Executable_OptimizedProgram(
    PJRT_Executable_OptimizedProgram_Args* args) {
  if (args == nullptr || args->executable == nullptr) {
    return InvalidArgument("executable must not be null");
  }
  if (args->program == nullptr) {
    return InvalidArgument("program must not be null");
  }
  return Unimplemented("optimized program serialization is not exposed");
}

extern "C" PJRT_Error* TT_Executable_Fingerprint(
    PJRT_Executable_Fingerprint_Args* args) {
  if (args == nullptr || args->executable == nullptr) {
    return InvalidArgument("executable must not be null");
  }
  SetStringOut(args->executable->metadata.fingerprint, &args->executable_fingerprint,
               &args->executable_fingerprint_size);
  return nullptr;
}

extern "C" PJRT_Error* TT_Executable_GetCompiledMemoryStats(
    PJRT_Executable_GetCompiledMemoryStats_Args* args) {
  if (args == nullptr || args->executable == nullptr) {
    return InvalidArgument("executable must not be null");
  }
  args->generated_code_size_in_bytes = 0;
  args->argument_size_in_bytes = 0;
  args->output_size_in_bytes = 0;
  args->alias_size_in_bytes = 0;
  args->temp_size_in_bytes = 0;
  args->host_generated_code_size_in_bytes = 0;
  args->host_argument_size_in_bytes = 0;
  args->host_output_size_in_bytes = 0;
  args->host_alias_size_in_bytes = 0;
  args->host_temp_size_in_bytes = 0;
  args->peak_memory_in_bytes = 0;
  args->total_size_in_bytes = 0;
  args->total_allocation_bytes = 0;
  args->indefinite_allocations = 0;
  args->peak_unpadded_heap_bytes = 0;
  return nullptr;
}

extern "C" PJRT_Error* TT_Executable_NumOutputs(PJRT_Executable_NumOutputs_Args* args) {
  if (args == nullptr || args->executable == nullptr) {
    return InvalidArgument("executable must not be null");
  }
  args->num_outputs = args->executable->metadata.num_outputs;
  return nullptr;
}

extern "C" PJRT_Error* TT_Executable_OutputElementTypes(
    PJRT_Executable_OutputElementTypes_Args* args) {
  if (args == nullptr || args->executable == nullptr) {
    return InvalidArgument("executable must not be null");
  }
  args->output_types = args->executable->metadata.output_types.empty()
                           ? nullptr
                           : args->executable->metadata.output_types.data();
  args->num_output_types = args->executable->metadata.output_types.size();
  return nullptr;
}

extern "C" PJRT_Error* TT_Executable_OutputDimensions(
    PJRT_Executable_OutputDimensions_Args* args) {
  if (args == nullptr || args->executable == nullptr) {
    return InvalidArgument("executable must not be null");
  }
  args->dims = args->executable->metadata.output_dims.empty()
                   ? nullptr
                   : args->executable->metadata.output_dims.data();
  args->dim_sizes = args->executable->metadata.output_dim_sizes.empty()
                        ? nullptr
                        : args->executable->metadata.output_dim_sizes.data();
  args->num_outputs = args->executable->metadata.output_dim_sizes.size();
  return nullptr;
}

extern "C" PJRT_Error* TT_Executable_OutputMemoryKinds(
    PJRT_Executable_OutputMemoryKinds_Args* args) {
  if (args == nullptr || args->executable == nullptr) {
    return InvalidArgument("executable must not be null");
  }
  args->memory_kinds = args->executable->metadata.output_memory_kind_ptrs.empty()
                           ? nullptr
                           : args->executable->metadata.output_memory_kind_ptrs.data();
  args->memory_kind_sizes = args->executable->metadata.output_memory_kind_sizes.empty()
                                ? nullptr
                                : args->executable->metadata.output_memory_kind_sizes.data();
  args->num_outputs = args->executable->metadata.output_memory_kind_ptrs.size();
  return nullptr;
}

extern "C" PJRT_Error* TT_LoadedExecutable_Destroy(
    PJRT_LoadedExecutable_Destroy_Args* args) {
  if (args == nullptr) {
    return InvalidArgument("args must not be null");
  }
  delete args->executable;
  args->executable = nullptr;
  return nullptr;
}

extern "C" PJRT_Error* TT_LoadedExecutable_GetExecutable(
    PJRT_LoadedExecutable_GetExecutable_Args* args) {
  if (args == nullptr || args->loaded_executable == nullptr) {
    return InvalidArgument("loaded_executable must not be null");
  }
  args->executable = new PJRT_Executable{args->loaded_executable->metadata};
  return nullptr;
}

extern "C" void TT_NoopSerializedDeviceAssignmentDeleter(
    PJRT_DeviceAssignmentSerialized* assignment) {}

extern "C" PJRT_Error* TT_LoadedExecutable_GetDeviceAssignment(
    PJRT_LoadedExecutable_GetDeviceAssignment_Args* args) {
  if (args == nullptr || args->executable == nullptr) {
    return InvalidArgument("executable must not be null");
  }
  args->serialized_bytes = nullptr;
  args->serialized_bytes_size = 0;
  args->serialized_device_assignment = nullptr;
  args->serialized_device_assignment_deleter = TT_NoopSerializedDeviceAssignmentDeleter;
  return nullptr;
}

extern "C" PJRT_Error* TT_LoadedExecutable_AddressableDevices(
    PJRT_LoadedExecutable_AddressableDevices_Args* args) {
  if (args == nullptr || args->executable == nullptr) {
    return InvalidArgument("executable must not be null");
  }
  args->addressable_devices = args->executable->addressable_devices.empty()
                                  ? nullptr
                                  : args->executable->addressable_devices.data();
  args->num_addressable_devices = args->executable->addressable_devices.size();
  return nullptr;
}

extern "C" PJRT_Error* TT_LoadedExecutable_Delete(
    PJRT_LoadedExecutable_Delete_Args* args) {
  if (args == nullptr || args->executable == nullptr) {
    return InvalidArgument("executable must not be null");
  }
  args->executable->deleted = true;
  return nullptr;
}

extern "C" PJRT_Error* TT_LoadedExecutable_IsDeleted(
    PJRT_LoadedExecutable_IsDeleted_Args* args) {
  if (args == nullptr || args->executable == nullptr) {
    return InvalidArgument("executable must not be null");
  }
  args->is_deleted = args->executable->deleted;
  return nullptr;
}

extern "C" PJRT_Error* TT_LoadedExecutable_Fingerprint(
    PJRT_LoadedExecutable_Fingerprint_Args* args) {
  if (args == nullptr || args->executable == nullptr) {
    return InvalidArgument("executable must not be null");
  }
  SetStringOut(args->executable->metadata.fingerprint, &args->executable_fingerprint,
               &args->executable_fingerprint_size);
  return nullptr;
}

extern "C" PJRT_Error* TT_LoadedExecutable_Execute(
    PJRT_LoadedExecutable_Execute_Args* args) {
  if (args == nullptr || args->executable == nullptr) {
    return InvalidArgument("executable must not be null");
  }
  if (args->executable->deleted) {
    return FailedPrecondition("executable has been deleted");
  }
  if (args->num_devices != 1) {
    return Unimplemented("only single-device execution is supported");
  }
  if (args->argument_lists == nullptr && args->num_args > 0) {
    return InvalidArgument("argument_lists must not be null when num_args > 0");
  }
  if (args->argument_lists != nullptr && args->argument_lists[0] == nullptr && args->num_args > 0) {
    return InvalidArgument("argument_lists[0] must not be null when num_args > 0");
  }
  if (args->output_lists == nullptr) {
    return InvalidArgument("output_lists must not be null");
  }
  PJRT_Buffer** device_outputs = args->output_lists[0];
  if (device_outputs == nullptr) {
    return InvalidArgument("output_lists[0] must not be null");
  }
  PJRT_Device* target_device = ExecuteTargetDevice(args->executable, args);
  if (target_device == nullptr) {
    return InvalidArgument("no execute device available");
  }
  PJRT_Buffer* const* arguments = args->num_args == 0 ? nullptr : args->argument_lists[0];
  if (PJRT_Error* error = ExecuteParameterOnlyProgram(args->executable, arguments, args->num_args,
                                                      target_device, device_outputs)) {
    for (size_t i = 0; i < args->executable->metadata.num_outputs; ++i) {
      delete device_outputs[i];
      device_outputs[i] = nullptr;
    }
    return error;
  }
  if (args->device_complete_events != nullptr) {
    args->device_complete_events[0] = ReadyEvent();
  }
  return nullptr;
}

extern "C" PJRT_Error* TT_Client_BufferFromHostBuffer(
    PJRT_Client_BufferFromHostBuffer_Args* args) {
  if (args == nullptr || args->client == nullptr) {
    return InvalidArgument("client must not be null");
  }
  if (args->device_layout != nullptr) {
    return Unimplemented("custom device layouts are not supported");
  }
  if (!IsSupportedBufferType(args->type)) {
    return Unimplemented("unsupported PJRT buffer type");
  }

  std::vector<int64_t> dims;
  if (PJRT_Error* error = CopyDims(args->dims, args->num_dims, &dims)) {
    return error;
  }
  if (PJRT_Error* error =
          ValidateDenseRowMajorStrides(args->type, dims, args->byte_strides, args->num_byte_strides)) {
    return error;
  }

  size_t byte_size = 0;
  if (PJRT_Error* error = HostByteSize(args->type, dims, &byte_size)) {
    return error;
  }
  if (byte_size > 0 && args->data == nullptr) {
    return InvalidArgument("data must not be null");
  }

  PJRT_Device* target_device = SelectTargetDevice(args->client, args->device, args->memory);
  if (target_device == nullptr) {
    return InvalidArgument("no target device available");
  }
  PJRT_Memory* target_memory = args->memory != nullptr ? args->memory : target_device->default_memory;
  PJRT_Buffer* buffer = nullptr;
  if (PJRT_Error* error = CreatePjrtBufferFromHostBytes(args->type, dims, target_device,
                                                        target_memory, args->data, byte_size,
                                                        &buffer)) {
    return error;
  }
  args->done_with_host_buffer = ReadyEvent();
  args->buffer = buffer;
  return nullptr;
}

extern "C" PJRT_Error* TT_DeviceDescription_Id(PJRT_DeviceDescription_Id_Args* args) {
  if (args == nullptr || args->device_description == nullptr) {
    return InvalidArgument("device_description must not be null");
  }
  args->id = args->device_description->id;
  return nullptr;
}

extern "C" PJRT_Error* TT_DeviceDescription_ProcessIndex(
    PJRT_DeviceDescription_ProcessIndex_Args* args) {
  if (args == nullptr || args->device_description == nullptr) {
    return InvalidArgument("device_description must not be null");
  }
  args->process_index = args->device_description->process_index;
  return nullptr;
}

extern "C" PJRT_Error* TT_DeviceDescription_Attributes(
    PJRT_DeviceDescription_Attributes_Args* args) {
  if (args == nullptr || args->device_description == nullptr) {
    return InvalidArgument("device_description must not be null");
  }
  args->attributes = nullptr;
  args->num_attributes = 0;
  return nullptr;
}

extern "C" PJRT_Error* TT_DeviceDescription_Kind(PJRT_DeviceDescription_Kind_Args* args) {
  if (args == nullptr || args->device_description == nullptr) {
    return InvalidArgument("device_description must not be null");
  }
  SetStringOut(args->device_description->device_kind, &args->device_kind, &args->device_kind_size);
  return nullptr;
}

extern "C" PJRT_Error* TT_DeviceDescription_DebugString(
    PJRT_DeviceDescription_DebugString_Args* args) {
  if (args == nullptr || args->device_description == nullptr) {
    return InvalidArgument("device_description must not be null");
  }
  SetStringOut(args->device_description->debug_string, &args->debug_string, &args->debug_string_size);
  return nullptr;
}

extern "C" PJRT_Error* TT_DeviceDescription_ToString(
    PJRT_DeviceDescription_ToString_Args* args) {
  if (args == nullptr || args->device_description == nullptr) {
    return InvalidArgument("device_description must not be null");
  }
  SetStringOut(args->device_description->to_string, &args->to_string, &args->to_string_size);
  return nullptr;
}

extern "C" PJRT_Error* TT_Device_GetDescription(PJRT_Device_GetDescription_Args* args) {
  if (args == nullptr || args->device == nullptr) {
    return InvalidArgument("device must not be null");
  }
  args->device_description = args->device->description;
  return nullptr;
}

extern "C" PJRT_Error* TT_Device_IsAddressable(PJRT_Device_IsAddressable_Args* args) {
  if (args == nullptr || args->device == nullptr) {
    return InvalidArgument("device must not be null");
  }
  args->is_addressable = args->device->addressable;
  return nullptr;
}

extern "C" PJRT_Error* TT_Device_LocalHardwareId(PJRT_Device_LocalHardwareId_Args* args) {
  if (args == nullptr || args->device == nullptr) {
    return InvalidArgument("device must not be null");
  }
  args->local_hardware_id = args->device->local_hardware_id;
  return nullptr;
}

extern "C" PJRT_Error* TT_Device_AddressableMemories(
    PJRT_Device_AddressableMemories_Args* args) {
  if (args == nullptr || args->device == nullptr) {
    return InvalidArgument("device must not be null");
  }
  args->memories = args->device->memory_ptrs.empty() ? nullptr : args->device->memory_ptrs.data();
  args->num_memories = args->device->memory_ptrs.size();
  return nullptr;
}

extern "C" PJRT_Error* TT_Device_DefaultMemory(PJRT_Device_DefaultMemory_Args* args) {
  if (args == nullptr || args->device == nullptr) {
    return InvalidArgument("device must not be null");
  }
  args->memory = args->device->default_memory;
  return nullptr;
}

extern "C" PJRT_Error* TT_Device_MemoryStats(PJRT_Device_MemoryStats_Args* args) {
  if (args == nullptr || args->device == nullptr) {
    return InvalidArgument("device must not be null");
  }
  args->bytes_in_use = 0;
  args->peak_bytes_in_use = 0;
  args->peak_bytes_in_use_is_set = true;
  args->num_allocs = 0;
  args->num_allocs_is_set = false;
  args->largest_alloc_size = 0;
  args->largest_alloc_size_is_set = false;
  args->bytes_limit = 0;
  args->bytes_limit_is_set = false;
  args->bytes_reserved = 0;
  args->bytes_reserved_is_set = false;
  args->peak_bytes_reserved = 0;
  args->peak_bytes_reserved_is_set = false;
  args->bytes_reservable_limit = 0;
  args->bytes_reservable_limit_is_set = false;
  args->largest_free_block_bytes = 0;
  args->largest_free_block_bytes_is_set = false;
  args->pool_bytes = 0;
  args->pool_bytes_is_set = false;
  args->peak_pool_bytes = 0;
  args->peak_pool_bytes_is_set = false;
  return nullptr;
}

extern "C" void TT_Device_Attributes_Delete(PJRT_Device_Attributes* attributes) {
  delete attributes;
}

extern "C" PJRT_Error* TT_Device_GetAttributes(PJRT_Device_GetAttributes_Args* args) {
  if (args == nullptr || args->device == nullptr) {
    return InvalidArgument("device must not be null");
  }
  args->attributes = nullptr;
  args->num_attributes = 0;
  args->device_attributes = new PJRT_Device_Attributes;
  args->attributes_deleter = TT_Device_Attributes_Delete;
  return nullptr;
}

extern "C" PJRT_Error* TT_Memory_Id(PJRT_Memory_Id_Args* args) {
  if (args == nullptr || args->memory == nullptr) {
    return InvalidArgument("memory must not be null");
  }
  args->id = args->memory->id;
  return nullptr;
}

extern "C" PJRT_Error* TT_Memory_Kind(PJRT_Memory_Kind_Args* args) {
  if (args == nullptr || args->memory == nullptr) {
    return InvalidArgument("memory must not be null");
  }
  SetStringOut(args->memory->kind, &args->kind, &args->kind_size);
  return nullptr;
}

extern "C" PJRT_Error* TT_Memory_Kind_Id(PJRT_Memory_Kind_Id_Args* args) {
  if (args == nullptr || args->memory == nullptr) {
    return InvalidArgument("memory must not be null");
  }
  args->kind_id = args->memory->id;
  return nullptr;
}

extern "C" PJRT_Error* TT_Memory_DebugString(PJRT_Memory_DebugString_Args* args) {
  if (args == nullptr || args->memory == nullptr) {
    return InvalidArgument("memory must not be null");
  }
  SetStringOut(args->memory->debug_string, &args->debug_string, &args->debug_string_size);
  return nullptr;
}

extern "C" PJRT_Error* TT_Memory_ToString(PJRT_Memory_ToString_Args* args) {
  if (args == nullptr || args->memory == nullptr) {
    return InvalidArgument("memory must not be null");
  }
  SetStringOut(args->memory->to_string, &args->to_string, &args->to_string_size);
  return nullptr;
}

extern "C" PJRT_Error* TT_Memory_AddressableByDevices(
    PJRT_Memory_AddressableByDevices_Args* args) {
  if (args == nullptr || args->memory == nullptr) {
    return InvalidArgument("memory must not be null");
  }
  args->devices = args->memory->device_ptrs.empty() ? nullptr : args->memory->device_ptrs.data();
  args->num_devices = args->memory->device_ptrs.size();
  return nullptr;
}

extern "C" PJRT_Error* TT_Buffer_Destroy(PJRT_Buffer_Destroy_Args* args) {
  if (args == nullptr) {
    return InvalidArgument("args must not be null");
  }
  delete args->buffer;
  args->buffer = nullptr;
  return nullptr;
}

extern "C" PJRT_Error* TT_Buffer_ElementType(PJRT_Buffer_ElementType_Args* args) {
  if (args == nullptr || args->buffer == nullptr) {
    return InvalidArgument("buffer must not be null");
  }
  args->type = args->buffer->buffer_type;
  return nullptr;
}

extern "C" PJRT_Error* TT_Buffer_Dimensions(PJRT_Buffer_Dimensions_Args* args) {
  if (args == nullptr || args->buffer == nullptr) {
    return InvalidArgument("buffer must not be null");
  }
  args->dims = args->buffer->dims.empty() ? nullptr : args->buffer->dims.data();
  args->num_dims = args->buffer->dims.size();
  return nullptr;
}

extern "C" PJRT_Error* TT_Buffer_UnpaddedDimensions(
    PJRT_Buffer_UnpaddedDimensions_Args* args) {
  if (args == nullptr || args->buffer == nullptr) {
    return InvalidArgument("buffer must not be null");
  }
  args->unpadded_dims = args->buffer->dims.empty() ? nullptr : args->buffer->dims.data();
  args->num_dims = args->buffer->dims.size();
  return nullptr;
}

extern "C" PJRT_Error* TT_Buffer_DynamicDimensionIndices(
    PJRT_Buffer_DynamicDimensionIndices_Args* args) {
  if (args == nullptr || args->buffer == nullptr) {
    return InvalidArgument("buffer must not be null");
  }
  args->dynamic_dim_indices = nullptr;
  args->num_dynamic_dims = 0;
  return nullptr;
}

extern "C" PJRT_Error* TT_Buffer_OnDeviceSizeInBytes(
    PJRT_Buffer_OnDeviceSizeInBytes_Args* args) {
  if (args == nullptr || args->buffer == nullptr) {
    return InvalidArgument("buffer must not be null");
  }
  if (args->buffer->deleted) {
    return FailedPrecondition("buffer has been deleted");
  }
  size_t size = 0;
  if (PJRT_Error* error = HostTensorPhysicalByteSize(*args->buffer, &size)) {
    return error;
  }
  args->on_device_size_in_bytes = size;
  return nullptr;
}

extern "C" PJRT_Error* TT_Buffer_Device(PJRT_Buffer_Device_Args* args) {
  if (args == nullptr || args->buffer == nullptr) {
    return InvalidArgument("buffer must not be null");
  }
  args->device = args->buffer->device;
  return nullptr;
}

extern "C" PJRT_Error* TT_Buffer_Memory(PJRT_Buffer_Memory_Args* args) {
  if (args == nullptr || args->buffer == nullptr) {
    return InvalidArgument("buffer must not be null");
  }
  args->memory = args->buffer->memory;
  return nullptr;
}

extern "C" PJRT_Error* TT_Buffer_Delete(PJRT_Buffer_Delete_Args* args) {
  if (args == nullptr || args->buffer == nullptr) {
    return InvalidArgument("buffer must not be null");
  }
  args->buffer->deleted = true;
  args->buffer->host_tensor.reset();
  return nullptr;
}

extern "C" PJRT_Error* TT_Buffer_IsDeleted(PJRT_Buffer_IsDeleted_Args* args) {
  if (args == nullptr || args->buffer == nullptr) {
    return InvalidArgument("buffer must not be null");
  }
  args->is_deleted = args->buffer->deleted;
  return nullptr;
}

extern "C" PJRT_Error* TT_Buffer_ToHostBuffer(PJRT_Buffer_ToHostBuffer_Args* args) {
  if (args == nullptr || args->src == nullptr) {
    return InvalidArgument("src must not be null");
  }
  if (args->src->deleted) {
    return FailedPrecondition("buffer has been deleted");
  }
  std::vector<std::byte> logical_data;
  if (PJRT_Error* error = ReadBufferLogicalBytes(*args->src, &logical_data)) {
    return error;
  }
  if (args->dst == nullptr) {
    args->dst_size = logical_data.size();
    args->event = ReadyEvent();
    return nullptr;
  }
  if (args->dst_size < logical_data.size()) {
    return InvalidArgument("dst buffer is too small");
  }
  if (!logical_data.empty()) {
    std::memcpy(args->dst, logical_data.data(), logical_data.size());
  }
  args->dst_size = logical_data.size();
  args->event = ReadyEvent();
  return nullptr;
}

extern "C" PJRT_Error* TT_Buffer_CopyRawToHost(PJRT_Buffer_CopyRawToHost_Args* args) {
  if (args == nullptr || args->buffer == nullptr) {
    return InvalidArgument("buffer must not be null");
  }
  if (args->buffer->deleted) {
    return FailedPrecondition("buffer has been deleted");
  }
  if (args->offset < 0 || args->transfer_size < 0) {
    return InvalidArgument("offset and transfer_size must be >= 0");
  }
  const size_t offset = static_cast<size_t>(args->offset);
  const size_t transfer_size = static_cast<size_t>(args->transfer_size);
  if (transfer_size > 0 && args->dst == nullptr) {
    return InvalidArgument("dst must not be null for non-empty copies");
  }
  std::vector<std::byte> logical_data;
  if (PJRT_Error* error = ReadBufferLogicalBytes(*args->buffer, &logical_data)) {
    return error;
  }
  if (offset > logical_data.size() || transfer_size > logical_data.size() - offset) {
    return InvalidArgument("raw host copy range is out of bounds");
  }
  if (transfer_size > 0) {
    std::memcpy(args->dst, logical_data.data() + offset, transfer_size);
  }
  args->event = ReadyEvent();
  return nullptr;
}

extern "C" PJRT_Error* TT_Buffer_IsOnCpu(PJRT_Buffer_IsOnCpu_Args* args) {
  if (args == nullptr || args->buffer == nullptr) {
    return InvalidArgument("buffer must not be null");
  }
  args->is_on_cpu = false;
  return nullptr;
}

extern "C" PJRT_Error* TT_Buffer_ReadyEvent(PJRT_Buffer_ReadyEvent_Args* args) {
  if (args == nullptr || args->buffer == nullptr) {
    return InvalidArgument("buffer must not be null");
  }
  args->event = args->buffer->deleted
                    ? EventWithError(PJRT_Error_Code_FAILED_PRECONDITION, "buffer has been deleted")
                    : ReadyEvent();
  return nullptr;
}

extern "C" PJRT_Error* TT_Buffer_IncreaseExternalReferenceCount(
    PJRT_Buffer_IncreaseExternalReferenceCount_Args* args) {
  if (args == nullptr || args->buffer == nullptr) {
    return InvalidArgument("buffer must not be null");
  }
  ++args->buffer->external_reference_count;
  return nullptr;
}

extern "C" PJRT_Error* TT_Buffer_DecreaseExternalReferenceCount(
    PJRT_Buffer_DecreaseExternalReferenceCount_Args* args) {
  if (args == nullptr || args->buffer == nullptr) {
    return InvalidArgument("buffer must not be null");
  }
  if (args->buffer->external_reference_count == 0) {
    return FailedPrecondition("external reference count is already zero");
  }
  --args->buffer->external_reference_count;
  return nullptr;
}

extern "C" PJRT_Error* TT_ExecuteContext_Create(PJRT_ExecuteContext_Create_Args* args) {
  if (args == nullptr) {
    return InvalidArgument("args must not be null");
  }
  args->context = new PJRT_ExecuteContext;
  return nullptr;
}

extern "C" PJRT_Error* TT_ExecuteContext_Destroy(PJRT_ExecuteContext_Destroy_Args* args) {
  if (args == nullptr) {
    return InvalidArgument("args must not be null");
  }
  delete args->context;
  args->context = nullptr;
  return nullptr;
}

extern "C" PJRT_Error* TT_TopologyDescription_PlatformName(
    PJRT_TopologyDescription_PlatformName_Args* args) {
  if (args == nullptr || args->topology == nullptr) {
    return InvalidArgument("topology must not be null");
  }
  SetStringOut(args->topology->platform_name, &args->platform_name, &args->platform_name_size);
  return nullptr;
}

extern "C" PJRT_Error* TT_TopologyDescription_PlatformVersion(
    PJRT_TopologyDescription_PlatformVersion_Args* args) {
  if (args == nullptr || args->topology == nullptr) {
    return InvalidArgument("topology must not be null");
  }
  SetStringOut(args->topology->platform_version, &args->platform_version,
               &args->platform_version_size);
  return nullptr;
}

extern "C" PJRT_Error* TT_TopologyDescription_GetDeviceDescriptions(
    PJRT_TopologyDescription_GetDeviceDescriptions_Args* args) {
  if (args == nullptr || args->topology == nullptr) {
    return InvalidArgument("topology must not be null");
  }
  args->descriptions = args->topology->device_description_ptrs.empty()
                           ? nullptr
                           : args->topology->device_description_ptrs.data();
  args->num_descriptions = args->topology->device_description_ptrs.size();
  return nullptr;
}

extern "C" PJRT_Error* TT_TopologyDescription_Attributes(
    PJRT_TopologyDescription_Attributes_Args* args) {
  if (args == nullptr || args->topology == nullptr) {
    return InvalidArgument("topology must not be null");
  }
  args->attributes = nullptr;
  args->num_attributes = 0;
  return nullptr;
}

namespace {

PJRT_Api MakePjrtApi() {
  PJRT_Api api{};
  api.struct_size = PJRT_Api_STRUCT_SIZE;
  api.extension_start = nullptr;
  api.pjrt_api_version = PJRT_Api_Version{
      PJRT_Api_Version_STRUCT_SIZE,
      nullptr,
      PJRT_API_MAJOR,
      PJRT_API_MINOR,
  };

  api.PJRT_Error_Destroy = TT_Error_Destroy;
  api.PJRT_Error_Message = TT_Error_Message;
  api.PJRT_Error_GetCode = TT_Error_GetCode;
  api.PJRT_Error_ForEachPayload = TT_Error_ForEachPayload;

  api.PJRT_Plugin_Initialize = TT_Plugin_Initialize;
  api.PJRT_Plugin_Attributes = TT_Plugin_Attributes;

  api.PJRT_Event_Destroy = TT_Event_Destroy;
  api.PJRT_Event_IsReady = TT_Event_IsReady;
  api.PJRT_Event_Error = TT_Event_Error;
  api.PJRT_Event_Await = TT_Event_Await;
  api.PJRT_Event_OnReady = TT_Event_OnReady;
  api.PJRT_Event_Create = TT_Event_Create;
  api.PJRT_Event_Set = TT_Event_Set;

  api.PJRT_Client_Create = TT_Client_Create;
  api.PJRT_Client_Destroy = TT_Client_Destroy;
  api.PJRT_Client_PlatformName = TT_Client_PlatformName;
  api.PJRT_Client_ProcessIndex = TT_Client_ProcessIndex;
  api.PJRT_Client_PlatformVersion = TT_Client_PlatformVersion;
  api.PJRT_Client_TopologyDescription = TT_Client_TopologyDescription;
  api.PJRT_Client_Devices = TT_Client_Devices;
  api.PJRT_Client_AddressableDevices = TT_Client_AddressableDevices;
  api.PJRT_Client_LookupDevice = TT_Client_LookupDevice;
  api.PJRT_Client_LookupAddressableDevice = TT_Client_LookupAddressableDevice;
  api.PJRT_Client_AddressableMemories = TT_Client_AddressableMemories;
  api.PJRT_Client_Compile = TT_Client_Compile;
  api.PJRT_Client_DefaultDeviceAssignment = TT_Client_DefaultDeviceAssignment;
  api.PJRT_Client_BufferFromHostBuffer = TT_Client_BufferFromHostBuffer;

  api.PJRT_DeviceDescription_Id = TT_DeviceDescription_Id;
  api.PJRT_DeviceDescription_ProcessIndex = TT_DeviceDescription_ProcessIndex;
  api.PJRT_DeviceDescription_Attributes = TT_DeviceDescription_Attributes;
  api.PJRT_DeviceDescription_Kind = TT_DeviceDescription_Kind;
  api.PJRT_DeviceDescription_DebugString = TT_DeviceDescription_DebugString;
  api.PJRT_DeviceDescription_ToString = TT_DeviceDescription_ToString;

  api.PJRT_Device_GetDescription = TT_Device_GetDescription;
  api.PJRT_Device_IsAddressable = TT_Device_IsAddressable;
  api.PJRT_Device_LocalHardwareId = TT_Device_LocalHardwareId;
  api.PJRT_Device_AddressableMemories = TT_Device_AddressableMemories;
  api.PJRT_Device_DefaultMemory = TT_Device_DefaultMemory;
  api.PJRT_Device_MemoryStats = TT_Device_MemoryStats;
  api.PJRT_Device_GetAttributes = TT_Device_GetAttributes;

  api.PJRT_Memory_Id = TT_Memory_Id;
  api.PJRT_Memory_Kind = TT_Memory_Kind;
  api.PJRT_Memory_Kind_Id = TT_Memory_Kind_Id;
  api.PJRT_Memory_DebugString = TT_Memory_DebugString;
  api.PJRT_Memory_ToString = TT_Memory_ToString;
  api.PJRT_Memory_AddressableByDevices = TT_Memory_AddressableByDevices;

  api.PJRT_Executable_Destroy = TT_Executable_Destroy;
  api.PJRT_Executable_Name = TT_Executable_Name;
  api.PJRT_Executable_NumReplicas = TT_Executable_NumReplicas;
  api.PJRT_Executable_NumPartitions = TT_Executable_NumPartitions;
  api.PJRT_Executable_NumOutputs = TT_Executable_NumOutputs;
  api.PJRT_Executable_OutputMemoryKinds = TT_Executable_OutputMemoryKinds;
  api.PJRT_Executable_OptimizedProgram = TT_Executable_OptimizedProgram;
  api.PJRT_Executable_Fingerprint = TT_Executable_Fingerprint;
  api.PJRT_Executable_GetCompiledMemoryStats = TT_Executable_GetCompiledMemoryStats;
  api.PJRT_Executable_OutputElementTypes = TT_Executable_OutputElementTypes;
  api.PJRT_Executable_OutputDimensions = TT_Executable_OutputDimensions;

  api.PJRT_LoadedExecutable_Destroy = TT_LoadedExecutable_Destroy;
  api.PJRT_LoadedExecutable_GetExecutable = TT_LoadedExecutable_GetExecutable;
  api.PJRT_LoadedExecutable_GetDeviceAssignment = TT_LoadedExecutable_GetDeviceAssignment;
  api.PJRT_LoadedExecutable_AddressableDevices = TT_LoadedExecutable_AddressableDevices;
  api.PJRT_LoadedExecutable_Delete = TT_LoadedExecutable_Delete;
  api.PJRT_LoadedExecutable_IsDeleted = TT_LoadedExecutable_IsDeleted;
  api.PJRT_LoadedExecutable_Execute = TT_LoadedExecutable_Execute;
  api.PJRT_LoadedExecutable_Fingerprint = TT_LoadedExecutable_Fingerprint;

  api.PJRT_Buffer_Destroy = TT_Buffer_Destroy;
  api.PJRT_Buffer_ElementType = TT_Buffer_ElementType;
  api.PJRT_Buffer_Dimensions = TT_Buffer_Dimensions;
  api.PJRT_Buffer_UnpaddedDimensions = TT_Buffer_UnpaddedDimensions;
  api.PJRT_Buffer_DynamicDimensionIndices = TT_Buffer_DynamicDimensionIndices;
  api.PJRT_Buffer_OnDeviceSizeInBytes = TT_Buffer_OnDeviceSizeInBytes;
  api.PJRT_Buffer_Device = TT_Buffer_Device;
  api.PJRT_Buffer_Memory = TT_Buffer_Memory;
  api.PJRT_Buffer_Delete = TT_Buffer_Delete;
  api.PJRT_Buffer_IsDeleted = TT_Buffer_IsDeleted;
  api.PJRT_Buffer_ToHostBuffer = TT_Buffer_ToHostBuffer;
  api.PJRT_Buffer_CopyRawToHost = TT_Buffer_CopyRawToHost;
  api.PJRT_Buffer_IsOnCpu = TT_Buffer_IsOnCpu;
  api.PJRT_Buffer_ReadyEvent = TT_Buffer_ReadyEvent;
  api.PJRT_Buffer_IncreaseExternalReferenceCount = TT_Buffer_IncreaseExternalReferenceCount;
  api.PJRT_Buffer_DecreaseExternalReferenceCount = TT_Buffer_DecreaseExternalReferenceCount;

  api.PJRT_ExecuteContext_Create = TT_ExecuteContext_Create;
  api.PJRT_ExecuteContext_Destroy = TT_ExecuteContext_Destroy;

  api.PJRT_TopologyDescription_PlatformName = TT_TopologyDescription_PlatformName;
  api.PJRT_TopologyDescription_PlatformVersion = TT_TopologyDescription_PlatformVersion;
  api.PJRT_TopologyDescription_GetDeviceDescriptions =
      TT_TopologyDescription_GetDeviceDescriptions;
  api.PJRT_TopologyDescription_Attributes = TT_TopologyDescription_Attributes;

  api.PJRT_Compile = TT_Compile;
  return api;
}

const PJRT_Api kPjrtApi = MakePjrtApi();

}  // namespace

extern "C" __attribute__((visibility("default"))) const PJRT_Api* GetPjrtApi() {
  return &kPjrtApi;
}
