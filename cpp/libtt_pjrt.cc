#include "cpp/libtt_pjrt.h"

#include "cpp/pjrt_buffer.h"
#include "cpp/tt_metal_runtime.h"
#include "mlir/executable.pb.h"

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/page_config.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/tensor_layout.hpp>
#include <ttnn/operations/core/core.hpp>
#include <ttnn/operations/copy/typecast/typecast.hpp>
#include <ttnn/operations/creation/creation.hpp>
#include <ttnn/operations/data_movement/concat/concat.hpp>
#include <ttnn/operations/data_movement/gather/gather.hpp>
#include <ttnn/operations/data_movement/permute/permute.hpp>
#include <ttnn/operations/data_movement/repeat/repeat.hpp>
#include <ttnn/operations/data_movement/reshape_view/reshape.hpp>
#include <ttnn/operations/data_movement/scatter/scatter.hpp>
#include <ttnn/operations/data_movement/slice/slice.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/eltwise/binary/binary_composite.hpp>
#include <ttnn/operations/eltwise/ternary/ternary.hpp>
#include <ttnn/operations/eltwise/unary/unary.hpp>
#include <ttnn/operations/embedding/embedding.hpp>
#include <ttnn/operations/experimental/transformer/rotary_embedding/rotary_embedding.hpp>
#include <ttnn/operations/matmul/matmul.hpp>
#include <ttnn/operations/normalization/rmsnorm/rmsnorm.hpp>
#include <ttnn/operations/reduction/accumulation/cumsum/cumsum.hpp>
#include <ttnn/operations/reduction/generic/generic_reductions.hpp>
#include <ttnn/operations/reduction/prod/prod.hpp>
#include <ttnn/operations/reduction/topk/topk.hpp>
#include <ttnn/operation.hpp>
#include <ttnn/operations/transformer/sdpa/device/sdpa_device_operation.hpp>
#include <ttnn/operations/transformer/sdpa_decode/sdpa_decode.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/types.hpp>

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

extern "C" {
using TT_MlirAllocOutput = char* (*)(size_t size, void* user_data);

bool TT_MlirAnalyzeProgram(const char* format, size_t format_size, const char* code,
                           size_t code_size, TT_MlirAllocOutput alloc_output,
                           void* user_data);
}

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

struct ExecutableMetadata {
  std::string name;
  std::string fingerprint;
  size_t num_outputs = 0;
  std::vector<PJRT_Buffer_Type> output_types;
  std::vector<int64_t> output_dims;
  std::vector<size_t> output_dim_sizes;
  std::vector<const char*> output_memory_kind_ptrs;
  std::vector<size_t> output_memory_kind_sizes;
  std::string executable_proto;
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
constexpr std::string_view kDeviceMemoryKind = "device";

PJRT_Error* CloneEventError(const PJRT_Event* event) {
  if (event == nullptr || !event->error.has_value()) {
    return nullptr;
  }
  return MakePjrtError(event->error->first, event->error->second);
}

PJRT_Event* ReadyEvent() {
  auto event = std::make_unique<PJRT_Event>();
  event->ready = true;
  event->error = std::nullopt;
  return event.release();
}

PJRT_Event* EventWithError(PJRT_Error_Code code, std::string message) {
  auto event = std::make_unique<PJRT_Event>();
  event->ready = true;
  event->error = std::make_pair(code, std::move(message));
  return event.release();
}

std::string CopyString(std::string_view value) { return std::string(value.data(), value.size()); }

void SetStringOut(const std::string& value, const char** out, size_t* out_size) {
  *out = value.data();
  *out_size = value.size();
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

uint64_t StableHash64(std::string_view value) {
  constexpr uint64_t kFnvOffsetBasis = 14695981039346656037ull;
  constexpr uint64_t kFnvPrime = 1099511628211ull;
  uint64_t hash = kFnvOffsetBasis;
  for (unsigned char byte : value) {
    hash ^= byte;
    hash *= kFnvPrime;
  }
  return hash;
}

std::string FingerprintString(const std::vector<PJRT_Buffer_Type>& output_types,
                              const std::vector<int64_t>& output_dims,
                              const std::vector<size_t>& output_dim_sizes,
                              std::string_view executable_proto,
                              std::string_view program_format,
                              std::string_view program_code) {
  std::ostringstream fingerprint;
  fingerprint << "tt:executable_v2:outputs=";
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
  fingerprint << ":format_hash=" << StableHash64(program_format)
              << ":program_hash=" << StableHash64(program_code)
              << ":executable_hash=" << StableHash64(executable_proto);
  return fingerprint.str();
}

ExecutableMetadata MakeExecutableMetadata(const tt::AnalysisResult& analysis,
                                          std::string_view program_format,
                                          std::string_view program_code) {
  ExecutableMetadata metadata;
  metadata.name = kExecutableName;
  metadata.num_outputs = static_cast<size_t>(analysis.outputs_size());
  metadata.output_memory_kind_ptrs.assign(metadata.num_outputs, kDeviceMemoryKind.data());
  metadata.output_memory_kind_sizes.assign(metadata.num_outputs, kDeviceMemoryKind.size());
  metadata.output_types.reserve(metadata.num_outputs);
  metadata.output_dim_sizes.reserve(metadata.num_outputs);
  for (const tt::TensorDesc& tensor : analysis.outputs()) {
    metadata.output_types.push_back(PjrtBufferTypeFromProto(tensor.element_type()));
    metadata.output_dim_sizes.push_back(static_cast<size_t>(tensor.dims_size()));
    for (int64_t dim : tensor.dims()) {
      metadata.output_dims.push_back(dim);
    }
  }
  if (analysis.has_executable()) {
    metadata.executable_proto = analysis.executable().SerializeAsString();
  }
  metadata.fingerprint = FingerprintString(metadata.output_types, metadata.output_dims,
                                           metadata.output_dim_sizes, metadata.executable_proto,
                                           program_format, program_code);
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
  const std::string_view code(program.code == nullptr ? "" : program.code, program.code_size);
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

  *metadata = MakeExecutableMetadata(analysis, format, code);
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
    if (name.empty()) {
      continue;
    }
    int id = 0;
    const char* begin = name.data();
    const char* end = begin + name.size();
    const auto [ptr, parse_ec] = std::from_chars(begin, end, id);
    if (parse_ec == std::errc() && ptr == end && id >= 0) {
      ids.push_back(id);
    }
  }
  std::sort(ids.begin(), ids.end());
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  return ids;
}

PJRT_Client* CreateClient() {
  auto client = std::make_unique<PJRT_Client>();
  client->platform_name = kPlatformName;
  client->platform_version = kPlatformVersion;

  std::vector<int> discovered_ids = DiscoverDeviceIds();
  const bool has_discovered_devices = !discovered_ids.empty();
  if (discovered_ids.empty()) {
    discovered_ids.push_back(0);
  }
  client->device_descriptions_storage.reserve(discovered_ids.size());
  client->memories_storage.reserve(discovered_ids.size());
  client->devices_storage.reserve(discovered_ids.size());

  for (int device_id : discovered_ids) {
    const std::string suffix = std::to_string(device_id);
    const std::string debug_string = has_discovered_devices
                                         ? "Tenstorrent device /dev/tenstorrent/" + suffix
                                         : "Tenstorrent logical device " + suffix;
    client->device_descriptions_storage.push_back(PJRT_DeviceDescription{
        device_id,
        0,
        "Tenstorrent",
        debug_string,
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

  return client.release();
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
    case tt::Op::kRmsNorm:
      return "rms_norm";
    case tt::Op::kRope:
      return "rope";
    case tt::Op::kPagedSdpaDecode:
      return "paged_sdpa_decode";
    case tt::Op::KIND_NOT_SET:
      return "not_set";
  }
  return "unknown(" + std::to_string(static_cast<int>(kind)) + ")";
}

bool IsTtnnMatmulType(PJRT_Buffer_Type type) {
  return type == PJRT_Buffer_Type_BF16 || type == PJRT_Buffer_Type_F32;
}

bool IsFloatingElementType(tt::TensorDesc::ElementType type) {
  return type == tt::TensorDesc::ELEMENT_TYPE_BF16 ||
         type == tt::TensorDesc::ELEMENT_TYPE_F32;
}

PJRT_Error* TensorDescElementCount(const tt::TensorDesc& desc, uint64_t* out) {
  std::vector<int64_t> dims;
  if (PJRT_Error* error = TensorDescDims(desc, &dims)) {
    return error;
  }
  uint64_t count = 1;
  for (int64_t dim : dims) {
    if (dim != 0 && count > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(dim)) {
      return ResourceExhausted("executable tensor element count overflow");
    }
    count *= static_cast<uint64_t>(dim);
  }
  *out = count;
  return nullptr;
}

PJRT_Error* ShapeFromTensorDesc(const tt::TensorDesc& desc, ttnn::Shape* out) {
  ttnn::Shape::Container values;
  values.reserve(static_cast<size_t>(desc.dims_size()));
  for (int64_t dim : desc.dims()) {
    if (dim < 0 || dim > std::numeric_limits<uint32_t>::max()) {
      return InvalidArgument("shape dimensions must fit uint32_t for TTNN tensors");
    }
    values.push_back(static_cast<uint32_t>(dim));
  }
  *out = ttnn::Shape(std::move(values));
  return nullptr;
}

PJRT_Error* TensorSpecFromDesc(const tt::TensorDesc& desc,
                               tt::tt_metal::Layout layout,
                               tt::tt_metal::MemoryConfig memory_config,
                               std::optional<ttnn::TensorSpec>* out) {
  PJRT_Buffer_Type type = PJRT_Buffer_Type_INVALID;
  if (PJRT_Error* error = TensorDescBufferType(desc, &type)) {
    return error;
  }
  std::optional<tt::tt_metal::DataType> dtype = TtnnDataTypeForPjrtBufferType(type);
  if (!dtype.has_value()) {
    return Unimplemented("tensor dtype cannot be represented as a TTNN tensor dtype");
  }
  ttnn::Shape shape;
  if (PJRT_Error* error = ShapeFromTensorDesc(desc, &shape)) {
    return error;
  }
  tt::tt_metal::TensorLayout tensor_layout(
      *dtype, tt::tt_metal::PageConfig(layout), std::move(memory_config));
  out->emplace(std::move(shape), std::move(tensor_layout));
  return nullptr;
}

tt::tt_metal::Layout PreferredDeviceLayout(const tt::TensorDesc& desc) {
  return IsFloatingElementType(desc.element_type()) ? ttnn::TILE_LAYOUT
                                                    : ttnn::ROW_MAJOR_LAYOUT;
}

float F32FromBits(uint32_t bits) {
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

uint32_t ScaleBf16PackedToF32Bits(uint32_t packed) {
  const uint32_t bf16_bits = packed & 0xffffu;
  return bf16_bits << 16;
}

std::vector<int64_t> TensorShapeVector(const ttnn::Tensor& tensor) {
  const ttnn::Shape& shape = tensor.logical_shape();
  std::vector<int64_t> dims;
  dims.reserve(shape.rank());
  for (size_t i = 0; i < shape.rank(); ++i) {
    dims.push_back(static_cast<int64_t>(shape[i]));
  }
  return dims;
}

std::string DimsToString(const std::vector<int64_t>& dims) {
  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < dims.size(); ++i) {
    if (i != 0) {
      os << ", ";
    }
    os << dims[i];
  }
  os << "]";
  return os.str();
}

std::string TensorDescShapeString(const tt::TensorDesc& desc) {
  std::vector<int64_t> dims;
  dims.reserve(static_cast<size_t>(desc.dims_size()));
  for (int64_t dim : desc.dims()) {
    dims.push_back(dim);
  }
  return DimsToString(dims);
}

std::string RepeatedDimsToString(const google::protobuf::RepeatedField<int64_t>& values) {
  std::vector<int64_t> dims;
  dims.reserve(static_cast<size_t>(values.size()));
  for (int64_t value : values) {
    dims.push_back(value);
  }
  return DimsToString(dims);
}

std::string ReduceDimsToString(const ttnn::SmallVector<int>& dims) {
  std::vector<int64_t> values;
  values.reserve(dims.size());
  for (int dim : dims) {
    values.push_back(dim);
  }
  return DimsToString(values);
}

std::string FusedInputShapes(
    const tt::FusedElementwiseOp& fused,
    const std::vector<std::optional<ttnn::Tensor>>& values) {
  std::ostringstream os;
  os << " inputs";
  for (uint32_t input_id : fused.input_ids()) {
    os << " " << input_id << "=";
    if (input_id >= static_cast<uint32_t>(values.size()) || !values[input_id].has_value()) {
      os << "<missing>";
    } else {
      os << DimsToString(TensorShapeVector(*values[input_id]));
    }
  }
  return os.str();
}

PJRT_Error* ValidateTensorMatchesDesc(const ttnn::Tensor& tensor,
                                      const tt::TensorDesc& desc,
                                      std::string_view context) {
  PJRT_Buffer_Type type = PJRT_Buffer_Type_INVALID;
  if (PJRT_Error* error = TensorDescBufferType(desc, &type)) {
    return error;
  }
  const std::optional<tt::tt_metal::DataType> dtype = TtnnDataTypeForPjrtBufferType(type);
  if (!dtype.has_value()) {
    return Unimplemented("tensor dtype cannot be represented as a TTNN tensor dtype");
  }
  if (tensor.dtype() != *dtype) {
    return InvalidArgument(std::string(context) + " dtype does not match executable metadata");
  }
  if (tensor.logical_shape().rank() != static_cast<size_t>(desc.dims_size())) {
    return InvalidArgument(std::string(context) + " rank does not match executable metadata: got " +
                           DimsToString(TensorShapeVector(tensor)) + ", expected " +
                           TensorDescShapeString(desc));
  }
  for (int i = 0; i < desc.dims_size(); ++i) {
    if (desc.dims(i) < 0 || tensor.logical_shape()[static_cast<size_t>(i)] !=
                                static_cast<uint64_t>(desc.dims(i))) {
      return InvalidArgument(std::string(context) + " shape does not match executable metadata: got " +
                             DimsToString(TensorShapeVector(tensor)) + ", expected " +
                             TensorDescShapeString(desc));
    }
  }
  return nullptr;
}

PJRT_Error* ReshapeTensorToDescIfSameVolume(ttnn::Tensor* tensor,
                                            const tt::TensorDesc& desc,
                                            std::string_view context) {
  std::vector<int64_t> desc_dims;
  if (PJRT_Error* error = TensorDescDims(desc, &desc_dims)) {
    return error;
  }
  if (TensorShapeVector(*tensor) == desc_dims) {
    return nullptr;
  }
  uint64_t desc_volume = 1;
  for (int64_t dim : desc_dims) {
    if (dim < 0) {
      return InvalidArgument(std::string(context) + " has a negative metadata dimension");
    }
    desc_volume *= static_cast<uint64_t>(dim);
  }
  if (tensor->logical_volume() != desc_volume) {
    return nullptr;
  }
  ttnn::Shape shape;
  if (PJRT_Error* error = ShapeFromTensorDesc(desc, &shape)) {
    return error;
  }
  try {
    *tensor = ttnn::reshape(*tensor, shape, ttnn::DRAM_MEMORY_CONFIG);
  } catch (const std::exception& e) {
    return Internal(std::string("TTNN ") + std::string(context) +
                    " metadata reshape failed from " +
                    DimsToString(TensorShapeVector(*tensor)) + " to " +
                    TensorDescShapeString(desc) + ": " + e.what());
  }
  return nullptr;
}

PJRT_Error* EmplaceTensorMatchingDesc(ttnn::Tensor result,
                                      const tt::TensorDesc& desc,
                                      std::string_view context,
                                      std::optional<ttnn::Tensor>* out) {
  if (PJRT_Error* error = ReshapeTensorToDescIfSameVolume(&result, desc, context)) {
    return error;
  }
  out->emplace(std::move(result));
  return ValidateTensorMatchesDesc(out->value(), desc, context);
}

template <typename T>
PJRT_Error* TensorFromBytePayload(const void* data,
                                  size_t byte_size,
                                  const ttnn::TensorSpec& spec,
                                  tt::tt_metal::distributed::MeshDevice* mesh_device,
                                  std::optional<ttnn::Tensor>* out) {
  static_assert(std::is_trivially_copyable_v<T>);
  if (byte_size % sizeof(T) != 0) {
    return InvalidArgument("constant byte payload is not a multiple of its element size");
  }
  std::vector<T> values(byte_size / sizeof(T));
  if (byte_size != 0) {
    std::memcpy(values.data(), data, byte_size);
  }
  try {
    out->emplace(ttnn::Tensor::from_vector(std::move(values), spec, mesh_device));
  } catch (const std::exception& e) {
    return Internal(std::string("failed to create TTNN constant tensor: ") + e.what());
  }
  return nullptr;
}

template <typename T>
PJRT_Error* TensorFromSplat(T value,
                            uint64_t element_count,
                            const ttnn::TensorSpec& spec,
                            tt::tt_metal::distributed::MeshDevice* mesh_device,
                            std::optional<ttnn::Tensor>* out) {
  if (element_count > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    return ResourceExhausted("constant tensor element count overflow");
  }
  std::vector<T> values(static_cast<size_t>(element_count), value);
  try {
    out->emplace(ttnn::Tensor::from_vector(std::move(values), spec, mesh_device));
  } catch (const std::exception& e) {
    return Internal(std::string("failed to create TTNN splat constant tensor: ") + e.what());
  }
  return nullptr;
}

PJRT_Error* CreateConstantTensor(const tt::ConstantOp& constant,
                                 const tt::TensorDesc& output_desc,
                                 tt::tt_metal::distributed::MeshDevice* mesh_device,
                                 std::optional<ttnn::Tensor>* out) {
  std::optional<ttnn::TensorSpec> spec;
  if (PJRT_Error* error = TensorSpecFromDesc(output_desc,
                                             PreferredDeviceLayout(output_desc),
                                             ttnn::DRAM_MEMORY_CONFIG,
                                             &spec)) {
    return error;
  }
  uint64_t element_count = 0;
  if (PJRT_Error* error = TensorDescElementCount(output_desc, &element_count)) {
    return error;
  }

  const std::string& data = constant.data();
  const void* payload = data.empty() ? nullptr : data.data();
  const size_t payload_size = data.size();
  switch (output_desc.element_type()) {
    case tt::TensorDesc::ELEMENT_TYPE_BF16:
      if (!data.empty()) {
        return TensorFromBytePayload<bfloat16>(payload, payload_size, *spec, mesh_device, out);
      }
      return TensorFromSplat<bfloat16>(
          bfloat16(F32FromBits(ScaleBf16PackedToF32Bits(constant.packed_value()))),
          element_count, *spec, mesh_device, out);
    case tt::TensorDesc::ELEMENT_TYPE_F32:
      if (!data.empty()) {
        return TensorFromBytePayload<float>(payload, payload_size, *spec, mesh_device, out);
      }
      return TensorFromSplat<float>(F32FromBits(constant.packed_value()),
                                    element_count, *spec, mesh_device, out);
    case tt::TensorDesc::ELEMENT_TYPE_U32:
      if (!data.empty()) {
        return TensorFromBytePayload<uint32_t>(payload, payload_size, *spec, mesh_device, out);
      }
      return TensorFromSplat<uint32_t>(constant.packed_value(),
                                       element_count, *spec, mesh_device, out);
    case tt::TensorDesc::ELEMENT_TYPE_S32:
      if (!data.empty()) {
        return TensorFromBytePayload<int32_t>(payload, payload_size, *spec, mesh_device, out);
      }
      return TensorFromSplat<int32_t>(static_cast<int32_t>(constant.packed_value()),
                                      element_count, *spec, mesh_device, out);
    case tt::TensorDesc::ELEMENT_TYPE_U16:
      if (!data.empty()) {
        return TensorFromBytePayload<uint16_t>(payload, payload_size, *spec, mesh_device, out);
      }
      return TensorFromSplat<uint16_t>(static_cast<uint16_t>(constant.packed_value()),
                                       element_count, *spec, mesh_device, out);
    case tt::TensorDesc::ELEMENT_TYPE_U8:
    case tt::TensorDesc::ELEMENT_TYPE_PRED:
      if (!data.empty()) {
        return TensorFromBytePayload<uint8_t>(payload, payload_size, *spec, mesh_device, out);
      }
      return TensorFromSplat<uint8_t>(static_cast<uint8_t>(constant.packed_value()),
                                      element_count, *spec, mesh_device, out);
    default:
      return Unimplemented("constant element type is not implemented");
  }
}

ttnn::Tensor CastTensorIfNeeded(const ttnn::Tensor& tensor,
                                tt::tt_metal::DataType dtype,
                                tt::tt_metal::distributed::MeshDevice* mesh_device) {
  if (tensor.dtype() == dtype) {
    return tensor;
  }
  if (tensor.storage_type() == tt::tt_metal::StorageType::DEVICE) {
    return ttnn::typecast(tensor, dtype, ttnn::DRAM_MEMORY_CONFIG);
  }
  return ttnn::to_dtype(tensor, dtype).to_device(mesh_device, ttnn::DRAM_MEMORY_CONFIG);
}

ttnn::Tensor ToDeviceTensor(const ttnn::Tensor& tensor,
                            tt::tt_metal::distributed::MeshDevice* mesh_device,
                            tt::tt_metal::Layout layout) {
  ttnn::Tensor result = tensor;
  if (layout == ttnn::TILE_LAYOUT &&
      result.dtype() == tt::tt_metal::DataType::UINT8) {
    result = CastTensorIfNeeded(result, tt::tt_metal::DataType::UINT32,
                                mesh_device);
  }
  if (result.storage_type() != tt::tt_metal::StorageType::DEVICE) {
    result = result.to_device(mesh_device, ttnn::DRAM_MEMORY_CONFIG);
  } else if (result.memory_config() != ttnn::DRAM_MEMORY_CONFIG) {
    result = ttnn::to_memory_config(result, ttnn::DRAM_MEMORY_CONFIG);
  }
  if (result.layout() != layout) {
    result = ttnn::to_layout(result, layout, std::nullopt, ttnn::DRAM_MEMORY_CONFIG);
  }
  return result;
}

PJRT_Error* ArgumentTensorForParameter(const PJRT_Buffer& buffer,
                                       const tt::TensorDesc& desc,
                                       tt::tt_metal::distributed::MeshDevice* mesh_device,
                                       std::optional<ttnn::Tensor>* out) {
  const ttnn::Tensor* tensor = buffer.TtnnTensor();
  if (tensor == nullptr) {
    return FailedPrecondition("argument buffer has been deleted");
  }
  PJRT_Buffer_Type expected_type = PJRT_Buffer_Type_INVALID;
  if (PJRT_Error* error = TensorDescBufferType(desc, &expected_type)) {
    return error;
  }
  if (buffer.buffer_type != expected_type) {
    return InvalidArgument("argument buffer dtype does not match executable parameter dtype");
  }
  if (buffer.dims.size() != static_cast<size_t>(desc.dims_size())) {
    return InvalidArgument("argument buffer rank does not match executable parameter rank");
  }
  for (int i = 0; i < desc.dims_size(); ++i) {
    if (buffer.dims[static_cast<size_t>(i)] != desc.dims(i)) {
      return InvalidArgument("argument buffer shape does not match executable parameter shape");
    }
  }
  try {
    out->emplace(ToDeviceTensor(*tensor, mesh_device, PreferredDeviceLayout(desc)));
  } catch (const std::exception& e) {
    return Internal(std::string("failed to copy argument tensor to device: ") + e.what());
  }
  return nullptr;
}

PJRT_Error* GetValueTensor(const std::vector<std::optional<ttnn::Tensor>>& values,
                           uint32_t id,
                           std::string_view context,
                           const ttnn::Tensor** out) {
  if (id >= static_cast<uint32_t>(values.size()) || !values[id].has_value()) {
    return Internal(std::string(context) + " input value was not produced");
  }
  *out = &*values[id];
  return nullptr;
}

ttnn::SmallVector<int64_t> Int64SmallVector(
    const google::protobuf::RepeatedField<int64_t>& values) {
  ttnn::SmallVector<int64_t> out;
  out.reserve(static_cast<size_t>(values.size()));
  for (int64_t value : values) {
    out.push_back(value);
  }
  return out;
}

ttnn::SmallVector<int32_t> I32SmallVector(
    const google::protobuf::RepeatedField<int64_t>& values) {
  ttnn::SmallVector<int32_t> out;
  out.reserve(static_cast<size_t>(values.size()));
  for (int64_t value : values) {
    out.push_back(static_cast<int32_t>(value));
  }
  return out;
}

ttnn::SmallVector<int32_t> I32SmallVector(const std::vector<int64_t>& values) {
  ttnn::SmallVector<int32_t> out;
  out.reserve(values.size());
  for (int64_t value : values) {
    out.push_back(static_cast<int32_t>(value));
  }
  return out;
}

ttnn::SmallVector<uint32_t> U32SmallVector(const std::vector<int64_t>& values) {
  ttnn::SmallVector<uint32_t> out;
  out.reserve(values.size());
  for (int64_t value : values) {
    out.push_back(static_cast<uint32_t>(value));
  }
  return out;
}

ttnn::SmallVector<int64_t> I64SmallVector(const std::vector<int64_t>& values) {
  ttnn::SmallVector<int64_t> out;
  out.reserve(values.size());
  for (int64_t value : values) {
    out.push_back(value);
  }
  return out;
}

ttnn::Tensor ReshapeTensorForOutputDType(const ttnn::Tensor& input,
                                         const ttnn::Shape& shape,
                                         tt::tt_metal::DataType output_dtype,
                                         tt::tt_metal::distributed::MeshDevice* mesh_device,
                                         bool keep_supported_dtype = false) {
  if (input.dtype() == tt::tt_metal::DataType::UINT8 &&
      output_dtype == tt::tt_metal::DataType::UINT8) {
    ttnn::Tensor cast_input = ToDeviceTensor(input, mesh_device, ttnn::TILE_LAYOUT);
    cast_input = CastTensorIfNeeded(cast_input, tt::tt_metal::DataType::UINT32,
                                    mesh_device);
    ttnn::Tensor reshaped = ttnn::reshape(cast_input, shape,
                                          ttnn::DRAM_MEMORY_CONFIG);
    if (keep_supported_dtype) {
      return reshaped;
    }
    return CastTensorIfNeeded(reshaped, output_dtype, mesh_device);
  }
  return ttnn::reshape(input, shape, ttnn::DRAM_MEMORY_CONFIG);
}

PJRT_Error* BroadcastFromReshapedDims(const ttnn::Tensor& input,
                                      const std::vector<int64_t>& reshaped_dims,
                                      const std::vector<int64_t>& output_dims,
                                      const tt::TensorDesc& output_desc,
                                      std::string_view context,
                                      tt::tt_metal::distributed::MeshDevice* mesh_device,
                                      std::optional<ttnn::Tensor>* out) {
  PJRT_Buffer_Type output_type = PJRT_Buffer_Type_INVALID;
  if (PJRT_Error* error = TensorDescBufferType(output_desc, &output_type)) {
    return error;
  }
  const std::optional<tt::tt_metal::DataType> output_dtype =
      TtnnDataTypeForPjrtBufferType(output_type);
  if (!output_dtype.has_value()) {
    return Unimplemented(std::string(context) + " output dtype is not supported");
  }

  const bool byte_broadcast = input.dtype() == tt::tt_metal::DataType::UINT8 &&
                              *output_dtype == tt::tt_metal::DataType::UINT8;
  ttnn::Tensor result = ReshapeTensorForOutputDType(
      input,
      ttnn::Shape(U32SmallVector(reshaped_dims)),
      *output_dtype,
      mesh_device,
      byte_broadcast);

  ttnn::SmallVector<uint32_t> repetitions;
  repetitions.reserve(output_dims.size());
  bool needs_repeat = false;
  for (size_t i = 0; i < output_dims.size(); ++i) {
    if (reshaped_dims[i] <= 0 || output_dims[i] % reshaped_dims[i] != 0) {
      return InvalidArgument(std::string(context) +
                             " output shape is not divisible by input shape");
    }
    const uint32_t repeat =
        static_cast<uint32_t>(output_dims[i] / reshaped_dims[i]);
    repetitions.push_back(repeat);
    needs_repeat = needs_repeat || repeat != 1;
  }
  if (needs_repeat) {
    result = ttnn::repeat(result, repetitions, ttnn::DRAM_MEMORY_CONFIG);
  }
  *out = byte_broadcast ? CastTensorIfNeeded(result, *output_dtype, mesh_device)
                        : result;
  return nullptr;
}

PJRT_Error* BroadcastTensorInDim(const ttnn::Tensor& input,
                                 const tt::BroadcastInDimOp& broadcast,
                                 const tt::TensorDesc& output_desc,
                                 tt::tt_metal::distributed::MeshDevice* mesh_device,
                                 std::optional<ttnn::Tensor>* out) {
  std::vector<int64_t> output_dims;
  if (PJRT_Error* error = TensorDescDims(output_desc, &output_dims)) {
    return error;
  }
  const std::vector<int64_t> input_dims = TensorShapeVector(input);
  if (broadcast.broadcast_dimensions_size() != static_cast<int>(input_dims.size())) {
    return Unimplemented("broadcast_in_dim rank-changing input metadata is inconsistent");
  }
  std::vector<int64_t> reshaped_dims(output_dims.size(), 1);
  for (int i = 0; i < broadcast.broadcast_dimensions_size(); ++i) {
    const int64_t output_dim = broadcast.broadcast_dimensions(i);
    if (output_dim < 0 || output_dim >= static_cast<int64_t>(output_dims.size())) {
      return InvalidArgument("broadcast_in_dim dimension is out of bounds");
    }
    const int64_t input_dim = input_dims[static_cast<size_t>(i)];
    if (input_dim != 1 && input_dim != output_dims[static_cast<size_t>(output_dim)]) {
      return InvalidArgument("broadcast_in_dim input dimension is not broadcast-compatible");
    }
    reshaped_dims[static_cast<size_t>(output_dim)] = input_dim;
  }
  return BroadcastFromReshapedDims(input,
                                   reshaped_dims,
                                   output_dims,
                                   output_desc,
                                   "broadcast_in_dim",
                                   mesh_device,
                                   out);
}

PJRT_Error* BroadcastTrailingDims(const ttnn::Tensor& input,
                                  const tt::TensorDesc& output_desc,
                                  tt::tt_metal::distributed::MeshDevice* mesh_device,
                                  std::optional<ttnn::Tensor>* out) {
  std::vector<int64_t> output_dims;
  if (PJRT_Error* error = TensorDescDims(output_desc, &output_dims)) {
    return error;
  }
  const std::vector<int64_t> input_dims = TensorShapeVector(input);
  if (input_dims == output_dims) {
    *out = input;
    return nullptr;
  }
  if (input_dims.size() > output_dims.size()) {
    return InvalidArgument("input rank is greater than broadcast output rank");
  }
  std::vector<int64_t> reshaped_dims(output_dims.size(), 1);
  const size_t offset = output_dims.size() - input_dims.size();
  for (size_t i = 0; i < input_dims.size(); ++i) {
    const int64_t input_dim = input_dims[i];
    const int64_t output_dim = output_dims[offset + i];
    if (input_dim != 1 && input_dim != output_dim) {
      return InvalidArgument("input shape is not trailing-broadcast-compatible");
    }
    reshaped_dims[offset + i] = input_dim;
  }
  return BroadcastFromReshapedDims(input,
                                   reshaped_dims,
                                   output_dims,
                                   output_desc,
                                   "broadcast",
                                   mesh_device,
                                   out);
}

struct DotGeneralPlan {
  std::vector<int64_t> lhs_permutation;
  std::vector<int64_t> rhs_permutation;
  std::vector<int64_t> lhs_canonical_shape;
  std::vector<int64_t> rhs_canonical_shape;
  std::vector<int64_t> output_canonical_shape;
};

uint64_t ProductOfDims(const std::vector<int64_t>& shape,
                       const std::vector<int64_t>& axes) {
  uint64_t product = 1;
  for (int64_t axis : axes) {
    product *= static_cast<uint64_t>(shape[static_cast<size_t>(axis)]);
  }
  return product;
}

PJRT_Error* MarkDotGeneralDims(const google::protobuf::RepeatedField<int64_t>& dims,
                               size_t rank,
                               std::vector<bool>* used,
                               std::string_view context) {
  for (int64_t dim : dims) {
    if (dim < 0 || static_cast<size_t>(dim) >= rank) {
      return InvalidArgument(std::string(context) + " dimension is out of bounds");
    }
    if ((*used)[static_cast<size_t>(dim)]) {
      return InvalidArgument(std::string(context) + " dimensions must be unique");
    }
    (*used)[static_cast<size_t>(dim)] = true;
  }
  return nullptr;
}

PJRT_Error* BuildDotGeneralPlan(const tt::MatmulOp& matmul,
                                const std::vector<int64_t>& lhs_dims,
                                const std::vector<int64_t>& rhs_dims,
                                const std::vector<int64_t>& output_dims,
                                DotGeneralPlan* plan) {
  if (matmul.lhs_contracting_dimensions_size() !=
      matmul.rhs_contracting_dimensions_size()) {
    return InvalidArgument("dot_general lhs/rhs contracting dimension counts must match");
  }
  if (matmul.lhs_batching_dimensions_size() != matmul.rhs_batching_dimensions_size()) {
    return InvalidArgument("dot_general lhs/rhs batching dimension counts must match");
  }

  const size_t lhs_rank = lhs_dims.size();
  const size_t rhs_rank = rhs_dims.size();
  std::vector<bool> lhs_used(lhs_rank, false);
  std::vector<bool> rhs_used(rhs_rank, false);
  if (PJRT_Error* error = MarkDotGeneralDims(matmul.lhs_batching_dimensions(),
                                             lhs_rank, &lhs_used,
                                             "dot_general lhs batching")) {
    return error;
  }
  if (PJRT_Error* error = MarkDotGeneralDims(matmul.rhs_batching_dimensions(),
                                             rhs_rank, &rhs_used,
                                             "dot_general rhs batching")) {
    return error;
  }
  if (PJRT_Error* error = MarkDotGeneralDims(matmul.lhs_contracting_dimensions(),
                                             lhs_rank, &lhs_used,
                                             "dot_general lhs contracting")) {
    return error;
  }
  if (PJRT_Error* error = MarkDotGeneralDims(matmul.rhs_contracting_dimensions(),
                                             rhs_rank, &rhs_used,
                                             "dot_general rhs contracting")) {
    return error;
  }

  for (int i = 0; i < matmul.lhs_contracting_dimensions_size(); ++i) {
    const int64_t lhs_contract = matmul.lhs_contracting_dimensions(i);
    const int64_t rhs_contract = matmul.rhs_contracting_dimensions(i);
    if (lhs_dims[static_cast<size_t>(lhs_contract)] !=
        rhs_dims[static_cast<size_t>(rhs_contract)]) {
      return InvalidArgument("dot_general contracting dimensions must have matching size");
    }
  }

  std::vector<int64_t> batch_shape;
  batch_shape.reserve(static_cast<size_t>(matmul.lhs_batching_dimensions_size()));
  for (int i = 0; i < matmul.lhs_batching_dimensions_size(); ++i) {
    const int64_t lhs_batch = matmul.lhs_batching_dimensions(i);
    const int64_t rhs_batch = matmul.rhs_batching_dimensions(i);
    if (lhs_dims[static_cast<size_t>(lhs_batch)] !=
        rhs_dims[static_cast<size_t>(rhs_batch)]) {
      return InvalidArgument("dot_general batching dimensions must have matching size");
    }
    batch_shape.push_back(lhs_dims[static_cast<size_t>(lhs_batch)]);
  }

  std::vector<int64_t> lhs_free_axes;
  for (size_t axis = 0; axis < lhs_rank; ++axis) {
    if (!lhs_used[axis]) {
      lhs_free_axes.push_back(static_cast<int64_t>(axis));
    }
  }
  std::vector<int64_t> rhs_free_axes;
  for (size_t axis = 0; axis < rhs_rank; ++axis) {
    if (!rhs_used[axis]) {
      rhs_free_axes.push_back(static_cast<int64_t>(axis));
    }
  }
  std::vector<int64_t> lhs_contract_axes;
  std::vector<int64_t> rhs_contract_axes;
  lhs_contract_axes.reserve(
      static_cast<size_t>(matmul.lhs_contracting_dimensions_size()));
  rhs_contract_axes.reserve(
      static_cast<size_t>(matmul.rhs_contracting_dimensions_size()));
  for (int64_t dim : matmul.lhs_contracting_dimensions()) {
    lhs_contract_axes.push_back(dim);
  }
  for (int64_t dim : matmul.rhs_contracting_dimensions()) {
    rhs_contract_axes.push_back(dim);
  }

  std::vector<int64_t> expected_output = batch_shape;
  for (int64_t axis : lhs_free_axes) {
    expected_output.push_back(lhs_dims[static_cast<size_t>(axis)]);
  }
  for (int64_t axis : rhs_free_axes) {
    expected_output.push_back(rhs_dims[static_cast<size_t>(axis)]);
  }
  if (expected_output != output_dims) {
    return InvalidArgument("dot_general output shape does not match dimension numbers: got " +
                           DimsToString(output_dims) + ", expected " +
                           DimsToString(expected_output));
  }

  plan->lhs_permutation.clear();
  plan->rhs_permutation.clear();
  plan->lhs_canonical_shape = batch_shape;
  plan->rhs_canonical_shape = batch_shape;
  for (int64_t dim : matmul.lhs_batching_dimensions()) {
    plan->lhs_permutation.push_back(dim);
  }
  for (int64_t dim : lhs_free_axes) {
    plan->lhs_permutation.push_back(dim);
  }
  for (int64_t dim : lhs_contract_axes) {
    plan->lhs_permutation.push_back(dim);
  }
  for (int64_t dim : matmul.rhs_batching_dimensions()) {
    plan->rhs_permutation.push_back(dim);
  }
  for (int64_t dim : rhs_contract_axes) {
    plan->rhs_permutation.push_back(dim);
  }
  for (int64_t dim : rhs_free_axes) {
    plan->rhs_permutation.push_back(dim);
  }

  plan->lhs_canonical_shape.push_back(static_cast<int64_t>(ProductOfDims(lhs_dims, lhs_free_axes)));
  plan->lhs_canonical_shape.push_back(static_cast<int64_t>(
      ProductOfDims(lhs_dims, lhs_contract_axes)));
  plan->rhs_canonical_shape.push_back(static_cast<int64_t>(
      ProductOfDims(rhs_dims, rhs_contract_axes)));
  plan->rhs_canonical_shape.push_back(static_cast<int64_t>(ProductOfDims(rhs_dims, rhs_free_axes)));
  plan->output_canonical_shape = batch_shape;
  plan->output_canonical_shape.push_back(static_cast<int64_t>(
      ProductOfDims(lhs_dims, lhs_free_axes)));
  plan->output_canonical_shape.push_back(static_cast<int64_t>(
      ProductOfDims(rhs_dims, rhs_free_axes)));
  return nullptr;
}

bool IsIdentityPermutation(const std::vector<int64_t>& permutation) {
  for (size_t i = 0; i < permutation.size(); ++i) {
    if (permutation[i] != static_cast<int64_t>(i)) {
      return false;
    }
  }
  return true;
}

ttnn::Tensor CanonicalizeDotGeneralOperand(const ttnn::Tensor& input,
                                           const std::vector<int64_t>& permutation,
                                           const std::vector<int64_t>& canonical_shape) {
  ttnn::Tensor result = input;
  if (!IsIdentityPermutation(permutation)) {
    result = ttnn::permute(result, I64SmallVector(permutation), ttnn::DRAM_MEMORY_CONFIG);
  }
  if (TensorShapeVector(result) != canonical_shape) {
    result = ttnn::reshape(result,
                           ttnn::Shape(U32SmallVector(canonical_shape)),
                           ttnn::DRAM_MEMORY_CONFIG);
  }
  return result;
}

PJRT_Error* PrepareRopeCacheForTtnn(const ttnn::Tensor& cache,
                                    int64_t head_dim,
                                    ttnn::Tensor* out) {
  const std::vector<int64_t> dims = TensorShapeVector(cache);
  if (dims.size() == 4 && dims[0] == 1 && dims[1] == 1 && dims[3] == head_dim) {
    *out = cache;
    return nullptr;
  }
  if (dims.size() != 2) {
    return InvalidArgument("rope cos/sin cache must be rank-2 or TTNN rank-4");
  }
  if (dims[1] != head_dim && dims[1] * 2 != head_dim) {
    return InvalidArgument("rope cos/sin cache width does not match head dimension");
  }

  ttnn::Tensor reshaped =
      ttnn::reshape(cache,
                    ttnn::Shape(U32SmallVector({1, 1, dims[0], dims[1]})),
                    ttnn::DRAM_MEMORY_CONFIG);
  if (dims[1] == head_dim) {
    *out = reshaped;
    return nullptr;
  }
  *out = ttnn::concat({reshaped, reshaped}, 3, ttnn::DRAM_MEMORY_CONFIG);
  return nullptr;
}

PJRT_Error* NormalizeReduceDims(size_t rank, ttnn::SmallVector<int>* dims) {
  if (rank > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return InvalidArgument("reduce input rank is out of int range");
  }
  const int int_rank = static_cast<int>(rank);
  for (int& dim : *dims) {
    if (dim < 0) {
      dim += int_rank;
    }
    if (dim < 0 || dim >= int_rank) {
      return InvalidArgument("reduce dimension is out of bounds for input rank");
    }
  }
  std::sort(dims->begin(), dims->end());
  for (size_t i = 1; i < dims->size(); ++i) {
    if ((*dims)[i] == (*dims)[i - 1]) {
      return InvalidArgument("reduce dimensions must be unique");
    }
  }
  return nullptr;
}

std::vector<int64_t> ReducedShape(const ttnn::Tensor& input,
                                  const ttnn::SmallVector<int>& dims);

PJRT_Error* CompleteReduceDimsFromOutputShape(const ttnn::Tensor& input,
                                              const tt::TensorDesc& output_desc,
                                              ttnn::SmallVector<int>* dims) {
  std::vector<int64_t> output_dims;
  if (PJRT_Error* error = TensorDescDims(output_desc, &output_dims)) {
    return error;
  }
  if (ReducedShape(input, *dims) == output_dims) {
    return nullptr;
  }

  const size_t rank = input.logical_shape().rank();
  std::vector<bool> reduced(rank, false);
  for (int dim : *dims) {
    if (dim < 0 || static_cast<size_t>(dim) >= rank) {
      return InvalidArgument("reduce dimension is out of bounds for input rank");
    }
    reduced[static_cast<size_t>(dim)] = true;
  }

  bool changed = false;
  size_t output_dim = 0;
  for (size_t axis = 0; axis < rank; ++axis) {
    if (reduced[axis]) {
      continue;
    }
    if (output_dim < output_dims.size() &&
        input.logical_shape()[axis] == static_cast<uint64_t>(output_dims[output_dim])) {
      ++output_dim;
    } else {
      dims->push_back(static_cast<int>(axis));
      reduced[axis] = true;
      changed = true;
    }
  }
  if (output_dim != output_dims.size()) {
    return InvalidArgument("reduce dimensions cannot be completed from input/output shapes");
  }
  if (changed) {
    std::sort(dims->begin(), dims->end());
  }
  return nullptr;
}

std::vector<int64_t> ReducedShape(const ttnn::Tensor& input,
                                  const ttnn::SmallVector<int>& dims) {
  std::vector<int64_t> shape;
  const ttnn::Shape& input_shape = input.logical_shape();
  shape.reserve(input_shape.rank() - dims.size());
  auto dim = dims.begin();
  for (size_t axis = 0; axis < input_shape.rank(); ++axis) {
    if (dim != dims.end() && *dim == static_cast<int>(axis)) {
      ++dim;
      continue;
    }
    shape.push_back(static_cast<int64_t>(input_shape[axis]));
  }
  return shape;
}

PJRT_Error* ValidateReduceOutputShape(const ttnn::Tensor& input,
                                      const ttnn::SmallVector<int>& dims,
                                      const tt::TensorDesc& output_desc) {
  std::vector<int64_t> output_dims;
  if (PJRT_Error* error = TensorDescDims(output_desc, &output_dims)) {
    return error;
  }
  const std::vector<int64_t> expected_dims = ReducedShape(input, dims);
  if (output_dims != expected_dims) {
    return InvalidArgument("reduce output shape does not match reduce dimensions: got " +
                           DimsToString(output_dims) + ", expected " +
                           DimsToString(expected_dims) + ", input " +
                           DimsToString(TensorShapeVector(input)) + ", dims " +
                           ReduceDimsToString(dims));
  }
  return nullptr;
}

PJRT_Error* ReshapeSingletonReduceOutput(const ttnn::Tensor& input,
                                         const ttnn::SmallVector<int>& dims,
                                         const tt::TensorDesc& output_desc,
                                         std::optional<ttnn::Tensor>* out) {
  if (!out->has_value()) {
    return Internal("reduce output tensor was not produced");
  }
  ttnn::Tensor& output = **out;
  if (output.logical_shape().rank() == static_cast<size_t>(output_desc.dims_size())) {
    return nullptr;
  }
  if (output.logical_shape().rank() != input.logical_shape().rank() ||
      input.logical_shape().rank() != static_cast<size_t>(output_desc.dims_size() + dims.size())) {
    return nullptr;
  }

  std::vector<int64_t> output_dims;
  if (PJRT_Error* error = TensorDescDims(output_desc, &output_dims)) {
    return error;
  }
  const size_t rank = input.logical_shape().rank();
  size_t output_dim = 0;
  auto dim = dims.begin();
  for (size_t axis = 0; axis < rank; ++axis) {
    const int64_t actual_dim = static_cast<int64_t>(output.logical_shape()[axis]);
    if (dim != dims.end() && *dim == static_cast<int>(axis)) {
      if (actual_dim != 1) {
        return InvalidArgument("TTNN reduce output kept a non-singleton reduced dimension");
      }
      ++dim;
    } else {
      if (output_dim >= output_dims.size() ||
          actual_dim != output_dims[output_dim]) {
        return InvalidArgument("TTNN reduce output shape does not match expected singleton-reduced shape");
      }
      ++output_dim;
    }
  }
  if (output_dim != output_dims.size() || dim != dims.end()) {
    return InvalidArgument("reduce output shape does not match reduced dimensions");
  }

  ttnn::Shape output_shape;
  if (PJRT_Error* error = ShapeFromTensorDesc(output_desc, &output_shape)) {
    return error;
  }
  output = ttnn::reshape(output, output_shape, ttnn::DRAM_MEMORY_CONFIG);
  return nullptr;
}

PJRT_Error* ExecuteTtnnMatmul(const tt::MatmulOp& matmul,
                              const ttnn::Tensor& lhs_input,
                              const ttnn::Tensor& rhs_input,
                              const tt::TensorDesc& lhs_desc,
                              const tt::TensorDesc& rhs_desc,
                              const tt::TensorDesc& output_desc,
                              tt::tt_metal::distributed::MeshDevice* mesh_device,
                              std::optional<ttnn::Tensor>* out) {
  PJRT_Buffer_Type lhs_type = PJRT_Buffer_Type_INVALID;
  PJRT_Buffer_Type rhs_type = PJRT_Buffer_Type_INVALID;
  PJRT_Buffer_Type output_type = PJRT_Buffer_Type_INVALID;
  if (PJRT_Error* error = TensorDescBufferType(lhs_desc, &lhs_type)) {
    return error;
  }
  if (PJRT_Error* error = TensorDescBufferType(rhs_desc, &rhs_type)) {
    return error;
  }
  if (PJRT_Error* error = TensorDescBufferType(output_desc, &output_type)) {
    return error;
  }
  if (lhs_type != rhs_type) {
    return InvalidArgument("TTNN matmul requires matching lhs/rhs input dtypes");
  }
  if (!IsTtnnMatmulType(lhs_type) || !IsTtnnMatmulType(output_type)) {
    return Unimplemented("TTNN matmul supports bf16 and f32 tensors");
  }

  std::vector<int64_t> lhs_dims;
  std::vector<int64_t> rhs_dims;
  std::vector<int64_t> output_dims;
  if (PJRT_Error* error = TensorDescDims(lhs_desc, &lhs_dims)) {
    return error;
  }
  if (PJRT_Error* error = TensorDescDims(rhs_desc, &rhs_dims)) {
    return error;
  }
  if (PJRT_Error* error = TensorDescDims(output_desc, &output_dims)) {
    return error;
  }
  DotGeneralPlan plan;
  if (PJRT_Error* error = BuildDotGeneralPlan(matmul, lhs_dims, rhs_dims,
                                              output_dims, &plan)) {
    return error;
  }
  const tt::tt_metal::DataType output_dtype =
      *TtnnDataTypeForPjrtBufferType(output_type);

  try {
    ttnn::Tensor lhs = ToDeviceTensor(lhs_input, mesh_device, ttnn::TILE_LAYOUT);
    ttnn::Tensor rhs = ToDeviceTensor(rhs_input, mesh_device, ttnn::TILE_LAYOUT);
    lhs = CanonicalizeDotGeneralOperand(lhs, plan.lhs_permutation,
                                        plan.lhs_canonical_shape);
    rhs = CanonicalizeDotGeneralOperand(rhs, plan.rhs_permutation,
                                        plan.rhs_canonical_shape);
    ttnn::Shape canonical_output_shape(U32SmallVector(plan.output_canonical_shape));
    ttnn::Tensor output_tensor = ttnn::empty(canonical_output_shape,
                                             output_dtype,
                                             ttnn::TILE_LAYOUT,
                                             mesh_device,
                                             ttnn::DRAM_MEMORY_CONFIG);

    ttnn::Tensor result = ttnn::matmul(
        /*input_tensor_a=*/lhs,
        /*input_tensor_b=*/rhs,
        /*transpose_a=*/false,
        /*transpose_b=*/false,
        /*memory_config=*/ttnn::DRAM_MEMORY_CONFIG,
        /*dtype=*/output_dtype,
        /*program_config=*/std::nullopt,
        /*activation=*/std::nullopt,
        /*compute_kernel_config=*/std::nullopt,
        /*core_grid=*/std::nullopt,
        /*output_tile=*/std::nullopt,
        /*optional_output_tensor=*/output_tensor);
    if (TensorShapeVector(result) != output_dims) {
      ttnn::Shape output_shape;
      if (PJRT_Error* error = ShapeFromTensorDesc(output_desc, &output_shape)) {
        return error;
      }
      result = ttnn::reshape(result, output_shape, ttnn::DRAM_MEMORY_CONFIG);
    }
    out->emplace(std::move(result));
    if (PJRT_Error* error = ValidateTensorMatchesDesc(out->value(), output_desc,
                                                      "matmul output")) {
      return error;
    }
  } catch (const std::exception& e) {
    return Internal(std::string("TTNN matmul failed: ") + e.what());
  } catch (...) {
    return Internal("TTNN matmul failed with unknown exception");
  }
  return nullptr;
}

PJRT_Error* ExecuteTopK(const tt::TopKOp& top_k,
                        const ttnn::Tensor& input,
                        const tt::TensorDesc& values_desc,
                        const tt::TensorDesc& indices_desc,
                        tt::tt_metal::distributed::MeshDevice* mesh_device,
  std::optional<ttnn::Tensor>* values_out,
  std::optional<ttnn::Tensor>* indices_out) {
  try {
    ttnn::Tensor tiled_input = ToDeviceTensor(input, mesh_device, ttnn::TILE_LAYOUT);
    if (tiled_input.dtype() != tt::tt_metal::DataType::BFLOAT16 &&
        tiled_input.dtype() != tt::tt_metal::DataType::BFLOAT8_B) {
      tiled_input = CastTensorIfNeeded(tiled_input, tt::tt_metal::DataType::BFLOAT16,
                                       mesh_device);
    }
    std::vector<ttnn::Tensor> result = ttnn::topk(
        tiled_input, top_k.k(), -1, true, true, ttnn::DRAM_MEMORY_CONFIG,
        std::nullopt);
    if (result.size() != 2) {
      return Internal("TTNN topk did not return values and indices");
    }
    PJRT_Buffer_Type values_type = PJRT_Buffer_Type_INVALID;
    if (PJRT_Error* error = TensorDescBufferType(values_desc, &values_type)) {
      return error;
    }
    const std::optional<tt::tt_metal::DataType> values_dtype =
        TtnnDataTypeForPjrtBufferType(values_type);
    if (!values_dtype.has_value()) {
      return Unimplemented("top_k values dtype is not supported");
    }
    PJRT_Buffer_Type indices_type = PJRT_Buffer_Type_INVALID;
    if (PJRT_Error* error = TensorDescBufferType(indices_desc, &indices_type)) {
      return error;
    }
    const std::optional<tt::tt_metal::DataType> indices_dtype =
        TtnnDataTypeForPjrtBufferType(indices_type);
    if (!indices_dtype.has_value()) {
      return Unimplemented("top_k indices dtype is not supported");
    }
    values_out->emplace(CastTensorIfNeeded(result[0], *values_dtype, mesh_device));
    indices_out->emplace(CastTensorIfNeeded(result[1], *indices_dtype, mesh_device));
    if (PJRT_Error* error = ReshapeTensorToDescIfSameVolume(&values_out->value(),
                                                            values_desc,
                                                            "top_k values")) {
      return error;
    }
    if (PJRT_Error* error = ReshapeTensorToDescIfSameVolume(&indices_out->value(),
                                                            indices_desc,
                                                            "top_k indices")) {
      return error;
    }
    if (PJRT_Error* error = ValidateTensorMatchesDesc(values_out->value(), values_desc, "top_k values")) {
      return error;
    }
    if (PJRT_Error* error = ValidateTensorMatchesDesc(indices_out->value(), indices_desc, "top_k indices")) {
      return error;
    }
  } catch (const std::exception& e) {
    return Internal(std::string("TTNN top_k failed: ") + e.what());
  } catch (...) {
    return Internal("TTNN top_k failed with unknown exception");
  }
  return nullptr;
}

PJRT_Error* ExecuteFusedElementwise(const tt::FusedElementwiseOp& fused,
                                    const tt::TensorDesc& output_desc,
                                    const std::vector<std::optional<ttnn::Tensor>>& values,
                                    tt::tt_metal::distributed::MeshDevice* mesh_device,
                                    std::optional<ttnn::Tensor>* out) {
  if (fused.nodes_size() == 0) {
    return InvalidArgument("fused_elementwise op has no nodes");
  }
  std::vector<std::optional<ttnn::Tensor>> node_values(static_cast<size_t>(fused.nodes_size()));
  auto node_input = [&](uint32_t node_id, const ttnn::Tensor** tensor) -> PJRT_Error* {
    if (node_id >= static_cast<uint32_t>(node_values.size()) ||
        !node_values[static_cast<size_t>(node_id)].has_value()) {
      return Internal("fused_elementwise node input was not produced");
    }
    *tensor = &*node_values[static_cast<size_t>(node_id)];
    return nullptr;
  };

  try {
    for (int i = 0; i < fused.nodes_size(); ++i) {
      const tt::FusedElementwiseOp::Node& node = fused.nodes(i);
      std::optional<ttnn::Tensor> result;
      switch (node.kind()) {
        case tt::FusedElementwiseOp::Node::KIND_INPUT: {
          if (node.input_index() >= static_cast<uint32_t>(fused.input_ids_size())) {
            return Internal("fused_elementwise input node index is out of bounds");
          }
          const uint32_t input_id = fused.input_ids(static_cast<int>(node.input_index()));
          const ttnn::Tensor* input = nullptr;
          if (PJRT_Error* error = GetValueTensor(values, input_id, "fused_elementwise", &input)) {
            return error;
          }
          if (node.single_tile_broadcast()) {
            if (PJRT_Error* error = BroadcastTrailingDims(*input, output_desc,
                                                          mesh_device,
                                                          &result)) {
              return error;
            }
          } else {
            result = *input;
            if (PJRT_Error* error = ReshapeTensorToDescIfSameVolume(&*result, output_desc,
                                                                    "fused_elementwise input")) {
              return error;
            }
          }
          break;
        }
        case tt::FusedElementwiseOp::Node::KIND_CONSTANT: {
          tt::TensorDesc node_desc = output_desc;
          node_desc.set_element_type(node.element_type());
          tt::ConstantOp constant;
          constant.set_packed_value(node.packed_value());
          if (PJRT_Error* error = CreateConstantTensor(constant, node_desc, mesh_device, &result)) {
            return error;
          }
          break;
        }
        case tt::FusedElementwiseOp::Node::KIND_ADD:
        case tt::FusedElementwiseOp::Node::KIND_SUBTRACT:
        case tt::FusedElementwiseOp::Node::KIND_MULTIPLY:
        case tt::FusedElementwiseOp::Node::KIND_DIVIDE:
        case tt::FusedElementwiseOp::Node::KIND_MAX:
        case tt::FusedElementwiseOp::Node::KIND_POWER:
        case tt::FusedElementwiseOp::Node::KIND_COMPARE: {
          if (node.input_nodes_size() != 2) {
            return InvalidArgument("binary fused_elementwise node must have two inputs");
          }
          const ttnn::Tensor* lhs = nullptr;
          const ttnn::Tensor* rhs = nullptr;
          if (PJRT_Error* error = node_input(node.input_nodes(0), &lhs)) {
            return error;
          }
          if (PJRT_Error* error = node_input(node.input_nodes(1), &rhs)) {
            return error;
          }
          switch (node.kind()) {
            case tt::FusedElementwiseOp::Node::KIND_ADD:
              result = ttnn::add(*lhs, *rhs, std::nullopt, ttnn::DRAM_MEMORY_CONFIG);
              break;
            case tt::FusedElementwiseOp::Node::KIND_SUBTRACT:
              result = ttnn::subtract(*lhs, *rhs, std::nullopt, ttnn::DRAM_MEMORY_CONFIG);
              break;
            case tt::FusedElementwiseOp::Node::KIND_MULTIPLY:
              result = ttnn::multiply(*lhs, *rhs, std::nullopt, ttnn::DRAM_MEMORY_CONFIG);
              break;
            case tt::FusedElementwiseOp::Node::KIND_DIVIDE:
              result = ttnn::divide(*lhs, *rhs, std::nullopt, ttnn::DRAM_MEMORY_CONFIG);
              break;
            case tt::FusedElementwiseOp::Node::KIND_MAX:
              result = ttnn::maximum(*lhs, *rhs, std::nullopt, ttnn::DRAM_MEMORY_CONFIG);
              break;
            case tt::FusedElementwiseOp::Node::KIND_POWER:
              result = ttnn::pow(*lhs, *rhs, std::nullopt, ttnn::DRAM_MEMORY_CONFIG);
              break;
            case tt::FusedElementwiseOp::Node::KIND_COMPARE:
              switch (node.compare_direction()) {
                case tt::FusedElementwiseOp::Node::DIRECTION_EQ:
                  result = ttnn::eq(*lhs, *rhs, std::nullopt, ttnn::DRAM_MEMORY_CONFIG);
                  break;
                case tt::FusedElementwiseOp::Node::DIRECTION_NE:
                  result = ttnn::ne(*lhs, *rhs, std::nullopt, ttnn::DRAM_MEMORY_CONFIG);
                  break;
                case tt::FusedElementwiseOp::Node::DIRECTION_GE:
                  result = ttnn::ge(*lhs, *rhs, std::nullopt, ttnn::DRAM_MEMORY_CONFIG);
                  break;
                case tt::FusedElementwiseOp::Node::DIRECTION_GT:
                  result = ttnn::gt(*lhs, *rhs, std::nullopt, ttnn::DRAM_MEMORY_CONFIG);
                  break;
                case tt::FusedElementwiseOp::Node::DIRECTION_LE:
                  result = ttnn::le(*lhs, *rhs, std::nullopt, ttnn::DRAM_MEMORY_CONFIG);
                  break;
                case tt::FusedElementwiseOp::Node::DIRECTION_LT:
                  result = ttnn::lt(*lhs, *rhs, std::nullopt, ttnn::DRAM_MEMORY_CONFIG);
                  break;
                default:
                  return InvalidArgument("unsupported fused_elementwise compare direction");
              }
              break;
            default:
              break;
          }
          break;
        }
        case tt::FusedElementwiseOp::Node::KIND_NEGATE:
        case tt::FusedElementwiseOp::Node::KIND_EXPONENTIAL:
        case tt::FusedElementwiseOp::Node::KIND_RSQRT:
        case tt::FusedElementwiseOp::Node::KIND_COSINE:
        case tt::FusedElementwiseOp::Node::KIND_SINE:
        case tt::FusedElementwiseOp::Node::KIND_CONVERT:
        case tt::FusedElementwiseOp::Node::KIND_LOG: {
          if (node.input_nodes_size() != 1) {
            return InvalidArgument("unary fused_elementwise node must have one input");
          }
          const ttnn::Tensor* input = nullptr;
          if (PJRT_Error* error = node_input(node.input_nodes(0), &input)) {
            return error;
          }
          switch (node.kind()) {
            case tt::FusedElementwiseOp::Node::KIND_NEGATE:
              result = ttnn::neg(*input, ttnn::DRAM_MEMORY_CONFIG);
              break;
            case tt::FusedElementwiseOp::Node::KIND_EXPONENTIAL:
              result = ttnn::exp(*input, false, ttnn::DRAM_MEMORY_CONFIG);
              break;
            case tt::FusedElementwiseOp::Node::KIND_RSQRT:
              result = ttnn::rsqrt(*input, false, ttnn::DRAM_MEMORY_CONFIG);
              break;
            case tt::FusedElementwiseOp::Node::KIND_COSINE:
              result = ttnn::cos(*input, ttnn::DRAM_MEMORY_CONFIG);
              break;
            case tt::FusedElementwiseOp::Node::KIND_SINE:
              result = ttnn::sin(*input, ttnn::DRAM_MEMORY_CONFIG);
              break;
            case tt::FusedElementwiseOp::Node::KIND_CONVERT: {
              PJRT_Buffer_Type type = PjrtBufferTypeFromProto(node.element_type());
              const std::optional<tt::tt_metal::DataType> dtype =
                  TtnnDataTypeForPjrtBufferType(type);
              if (!dtype.has_value()) {
                return Unimplemented("fused_elementwise convert dtype is not supported");
              }
              result = CastTensorIfNeeded(*input, *dtype, mesh_device);
              break;
            }
            case tt::FusedElementwiseOp::Node::KIND_LOG:
              result = ttnn::log(*input, false, ttnn::DRAM_MEMORY_CONFIG);
              break;
            default:
              break;
          }
          break;
        }
        default:
          return Unimplemented("unsupported fused_elementwise node kind");
      }
      if (!result.has_value()) {
        return Internal("fused_elementwise node did not produce a tensor");
      }
      node_values[static_cast<size_t>(i)] = std::move(*result);
    }
    PJRT_Buffer_Type output_type = PJRT_Buffer_Type_INVALID;
    if (PJRT_Error* error = TensorDescBufferType(output_desc, &output_type)) {
      return error;
    }
    const std::optional<tt::tt_metal::DataType> output_dtype =
        TtnnDataTypeForPjrtBufferType(output_type);
    if (!output_dtype.has_value()) {
      return Unimplemented("fused_elementwise output dtype is not supported");
    }
    out->emplace(CastTensorIfNeeded(*node_values.back(), *output_dtype, mesh_device));
    if (PJRT_Error* error = ReshapeTensorToDescIfSameVolume(&out->value(), output_desc,
                                                            "fused_elementwise output")) {
      return error;
    }
    std::vector<int64_t> output_dims;
    if (PJRT_Error* error = TensorDescDims(output_desc, &output_dims)) {
      return error;
    }
    if (TensorShapeVector(out->value()) != output_dims) {
      return InvalidArgument("fused_elementwise output shape does not match executable metadata: got " +
                             DimsToString(TensorShapeVector(out->value())) +
                             ", expected " + TensorDescShapeString(output_desc) +
                             FusedInputShapes(fused, values));
    }
    if (PJRT_Error* error = ValidateTensorMatchesDesc(out->value(), output_desc,
                                                      "fused_elementwise output")) {
      return error;
    }
  } catch (const std::exception& e) {
    return Internal(std::string("TTNN fused_elementwise failed: ") + e.what());
  } catch (...) {
    return Internal("TTNN fused_elementwise failed with unknown exception");
  }
  return nullptr;
}

bool RepeatedFieldEquals(const google::protobuf::RepeatedField<int64_t>& lhs,
                         const std::vector<int64_t>& rhs) {
  if (static_cast<size_t>(lhs.size()) != rhs.size()) {
    return false;
  }
  for (int i = 0; i < lhs.size(); ++i) {
    if (lhs.Get(i) != rhs[static_cast<size_t>(i)]) {
      return false;
    }
  }
  return true;
}

bool IsEmbeddingGather(const tt::GatherOp& gather,
                       const std::vector<int64_t>& operand_dims,
                       const std::vector<int64_t>& start_indices_dims,
                       const std::vector<int64_t>& output_dims) {
  return operand_dims.size() == 2 &&
         start_indices_dims.size() == 2 &&
         start_indices_dims[1] == 1 &&
         output_dims == std::vector<int64_t>{start_indices_dims[0], operand_dims[1]} &&
         RepeatedFieldEquals(gather.offset_dims(), {1}) &&
         RepeatedFieldEquals(gather.collapsed_slice_dims(), {0}) &&
         RepeatedFieldEquals(gather.start_index_map(), {0}) &&
         gather.operand_batching_dims_size() == 0 &&
         gather.start_indices_batching_dims_size() == 0 &&
         gather.index_vector_dim() == 1 &&
         RepeatedFieldEquals(gather.slice_sizes(), {1, operand_dims[1]});
}

bool IsSingleAxisGather(const tt::GatherOp& gather,
                        const std::vector<int64_t>& operand_dims,
                        const std::vector<int64_t>& start_indices_dims,
                        const std::vector<int64_t>& output_dims,
                        int64_t* gather_dim) {
  if (operand_dims.empty() || operand_dims.size() > 4 ||
      output_dims.size() != operand_dims.size() ||
      (start_indices_dims.size() != 1 &&
       !(start_indices_dims.size() == 2 && start_indices_dims[1] == 1)) ||
      gather.collapsed_slice_dims_size() != 1 ||
      gather.start_index_map_size() != 1) {
    return false;
  }

  *gather_dim = gather.collapsed_slice_dims(0);
  if (*gather_dim < 0 ||
      static_cast<size_t>(*gather_dim) >= operand_dims.size() ||
      gather.start_index_map(0) != *gather_dim ||
      gather.offset_dims_size() != static_cast<int>(operand_dims.size() - 1) ||
      gather.slice_sizes_size() != static_cast<int>(operand_dims.size())) {
    return false;
  }

  std::vector<int64_t> expected_output = operand_dims;
  expected_output[static_cast<size_t>(*gather_dim)] = start_indices_dims[0];
  std::vector<int64_t> expected_offset_dims;
  expected_offset_dims.reserve(operand_dims.size() - 1);
  std::vector<int64_t> expected_slice_sizes = operand_dims;
  expected_slice_sizes[static_cast<size_t>(*gather_dim)] = 1;
  for (size_t dim = 0; dim < operand_dims.size(); ++dim) {
    if (dim != static_cast<size_t>(*gather_dim)) {
      expected_offset_dims.push_back(static_cast<int64_t>(dim));
    }
  }

  return output_dims == expected_output &&
         RepeatedFieldEquals(gather.offset_dims(), expected_offset_dims) &&
         RepeatedFieldEquals(gather.slice_sizes(), expected_slice_sizes) &&
         gather.operand_batching_dims_size() == 0 &&
         gather.start_indices_batching_dims_size() == 0 &&
         gather.index_vector_dim() == 1;
}

PJRT_Error* GatherSingleAxis(const ttnn::Tensor& input,
                             const ttnn::Tensor& start_indices,
                             const std::vector<int64_t>& operand_dims,
                             const std::vector<int64_t>& output_dims,
                             int64_t gather_dim,
                             tt::tt_metal::DataType output_dtype,
                             tt::tt_metal::distributed::MeshDevice* mesh_device,
                             ttnn::Tensor* out) {
  ttnn::Tensor input_tensor = input;
  input_tensor = ToDeviceTensor(input_tensor, mesh_device, ttnn::TILE_LAYOUT);
  std::vector<int64_t> padded_operand_dims(4, 1);
  std::vector<int64_t> padded_output_dims(4, 1);
  const size_t offset = 4 - operand_dims.size();
  for (size_t dim = 0; dim < operand_dims.size(); ++dim) {
    padded_operand_dims[offset + dim] = operand_dims[dim];
    padded_output_dims[offset + dim] = output_dims[dim];
  }
  input_tensor = ttnn::reshape(input_tensor,
                               ttnn::Shape(U32SmallVector(padded_operand_dims)),
                               ttnn::DRAM_MEMORY_CONFIG);

  const size_t padded_gather_dim = offset + static_cast<size_t>(gather_dim);
  ttnn::Tensor indices = ToDeviceTensor(start_indices,
                                        mesh_device,
                                        ttnn::TILE_LAYOUT);
  indices = CastTensorIfNeeded(indices, tt::tt_metal::DataType::UINT32, mesh_device);
  std::vector<int64_t> index_shape(4, 1);
  index_shape[padded_gather_dim] = output_dims[static_cast<size_t>(gather_dim)];
  indices = ttnn::reshape(indices,
                          ttnn::Shape(U32SmallVector(index_shape)),
                          ttnn::DRAM_MEMORY_CONFIG);

  ttnn::SmallVector<uint32_t> repetitions;
  repetitions.reserve(4);
  bool needs_repeat = false;
  for (size_t dim = 0; dim < padded_output_dims.size(); ++dim) {
    if (index_shape[dim] <= 0 ||
        padded_output_dims[dim] % index_shape[dim] != 0) {
      return InvalidArgument("gather index shape cannot be broadcast to output shape");
    }
    const uint32_t repeat =
        static_cast<uint32_t>(padded_output_dims[dim] / index_shape[dim]);
    repetitions.push_back(repeat);
    needs_repeat = needs_repeat || repeat != 1;
  }
  if (needs_repeat) {
    indices = ttnn::repeat(indices, repetitions, ttnn::DRAM_MEMORY_CONFIG);
  }

  ttnn::Tensor result =
      ttnn::gather(input_tensor,
                   static_cast<int8_t>(padded_gather_dim),
                   indices,
                   /*sparse_grad=*/false,
                   ttnn::DRAM_MEMORY_CONFIG);
  *out = CastTensorIfNeeded(result, output_dtype, mesh_device);
  return nullptr;
}

std::string GatherMetadataString(const tt::GatherOp& gather,
                                 const std::vector<int64_t>& operand_dims,
                                 const std::vector<int64_t>& start_indices_dims,
                                 const std::vector<int64_t>& output_dims) {
  return "operand=" + DimsToString(operand_dims) +
         " start_indices=" + DimsToString(start_indices_dims) +
         " output=" + DimsToString(output_dims) +
         " offset_dims=" + RepeatedDimsToString(gather.offset_dims()) +
         " collapsed_slice_dims=" +
         RepeatedDimsToString(gather.collapsed_slice_dims()) +
         " start_index_map=" +
         RepeatedDimsToString(gather.start_index_map()) +
         " index_vector_dim=" + std::to_string(gather.index_vector_dim()) +
         " slice_sizes=" + RepeatedDimsToString(gather.slice_sizes());
}

PJRT_Error* ExecuteGather(const tt::GatherOp& gather,
                          const ttnn::Tensor& operand,
                          const ttnn::Tensor& start_indices,
                          const tt::TensorDesc& operand_desc,
                          const tt::TensorDesc& output_desc,
                          tt::tt_metal::distributed::MeshDevice* mesh_device,
                          std::optional<ttnn::Tensor>* out) {
  const std::vector<int64_t> start_indices_dims = TensorShapeVector(start_indices);
  std::vector<int64_t> operand_dims;
  std::vector<int64_t> output_dims;
  if (PJRT_Error* error = TensorDescDims(operand_desc, &operand_dims)) {
    return error;
  }
  if (PJRT_Error* error = TensorDescDims(output_desc, &output_dims)) {
    return error;
  }
  PJRT_Buffer_Type output_type = PJRT_Buffer_Type_INVALID;
  if (PJRT_Error* error = TensorDescBufferType(output_desc, &output_type)) {
    return error;
  }
  const std::optional<tt::tt_metal::DataType> output_dtype =
      TtnnDataTypeForPjrtBufferType(output_type);
  if (!output_dtype.has_value()) {
    return Unimplemented("gather output dtype is not supported");
  }
  const bool is_embedding =
      IsEmbeddingGather(gather, operand_dims, start_indices_dims, output_dims);
  int64_t gather_dim = -1;
  const bool is_single_axis =
      IsSingleAxisGather(gather, operand_dims, start_indices_dims, output_dims,
                         &gather_dim);
  if (!is_embedding && !is_single_axis) {
    return Unimplemented("TTNN gather does not support this StableHLO pattern: " +
                         GatherMetadataString(gather, operand_dims,
                                              start_indices_dims, output_dims));
  }

  try {
    if (is_single_axis && !is_embedding) {
      ttnn::Tensor result;
      if (PJRT_Error* error = GatherSingleAxis(operand,
                                               start_indices,
                                               operand_dims,
                                               output_dims,
                                               gather_dim,
                                               *output_dtype,
                                               mesh_device,
                                               &result)) {
        return error;
      }
      return EmplaceTensorMatchingDesc(std::move(result),
                                       output_desc,
                                       "gather output",
                                       out);
    }

    ttnn::Tensor input_tensor = operand;
    if (TensorShapeVector(input_tensor) != operand_dims) {
      uint64_t operand_elements = 0;
      if (PJRT_Error* error = TensorDescElementCount(operand_desc, &operand_elements)) {
        return error;
      }
      if (input_tensor.logical_volume() != operand_elements) {
        return Internal("gather operand shape does not match metadata: got " +
                        DimsToString(TensorShapeVector(input_tensor)) +
                        ", expected " + DimsToString(operand_dims));
      }
      ttnn::Shape operand_shape(U32SmallVector(operand_dims));
      input_tensor = ReshapeTensorForOutputDType(input_tensor,
                                                 operand_shape,
                                                 input_tensor.dtype(),
                                                 mesh_device);
    }
    if (input_tensor.dtype() != tt::tt_metal::DataType::BFLOAT16) {
      return Unimplemented("TTNN embedding gather currently requires a BF16 operand");
    }

    ttnn::Tensor indices = ToDeviceTensor(start_indices,
                                          mesh_device,
                                          ttnn::ROW_MAJOR_LAYOUT);
    indices = CastTensorIfNeeded(indices, tt::tt_metal::DataType::UINT32, mesh_device);
    ttnn::Tensor weight = ToDeviceTensor(input_tensor,
                                         mesh_device,
                                         ttnn::ROW_MAJOR_LAYOUT);
    ttnn::Tensor result = ttnn::embedding(indices,
                                          weight,
                                          std::nullopt,
                                          ttnn::TILE_LAYOUT,
                                          ttnn::prim::EmbeddingsType::GENERIC,
                                          *output_dtype,
                                          ttnn::DRAM_MEMORY_CONFIG);
    return EmplaceTensorMatchingDesc(std::move(result),
                                     output_desc,
                                     "gather output",
                                     out);
  } catch (const std::exception& e) {
    return Internal(std::string("TTNN gather failed: ") + e.what());
  } catch (...) {
    return Internal("TTNN gather failed with unknown exception");
  }
  return nullptr;
}

PJRT_Error* ValidateSetScatterDimensionNumbers(const tt::ScatterOp& scatter,
                                               size_t rank,
                                               int64_t* scatter_dim) {
  if (scatter.scatter_dims_to_operand_dims_size() != 1) {
    return Unimplemented("TTNN scatter requires one scatter_dims_to_operand_dims entry");
  }
  *scatter_dim = scatter.scatter_dims_to_operand_dims(0);
  if (*scatter_dim < 0 || static_cast<size_t>(*scatter_dim) >= rank) {
    return InvalidArgument("scatter dimension is out of bounds");
  }
  std::vector<int64_t> expected_update_window_dims;
  expected_update_window_dims.reserve(rank - 1);
  for (size_t dim = 0; dim < rank; ++dim) {
    if (dim != static_cast<size_t>(*scatter_dim)) {
      expected_update_window_dims.push_back(static_cast<int64_t>(dim));
    }
  }
  if (RepeatedDimsToString(scatter.update_window_dims()) !=
      DimsToString(expected_update_window_dims)) {
    return Unimplemented("TTNN scatter only supports set updates with update_window_dims " +
                         DimsToString(expected_update_window_dims) + ", got " +
                         RepeatedDimsToString(scatter.update_window_dims()));
  }
  if (scatter.inserted_window_dims_size() != 1 ||
      scatter.inserted_window_dims(0) != *scatter_dim) {
    return Unimplemented("TTNN scatter only supports inserted_window_dims matching the scatter dimension");
  }
  if (scatter.input_batching_dims_size() != 0 ||
      scatter.scatter_indices_batching_dims_size() != 0) {
    return Unimplemented("TTNN scatter does not support scatter batching dimensions");
  }
  if (scatter.index_vector_dim() != 1) {
    return Unimplemented("TTNN scatter requires index_vector_dim 1");
  }
  return nullptr;
}

PJRT_Error* ExecuteScatter(const tt::ScatterOp& scatter,
                           const ttnn::Tensor& operand,
                           const ttnn::Tensor& start_indices,
                           const ttnn::Tensor& updates,
                           const tt::TensorDesc& operand_desc,
                           const tt::TensorDesc& start_indices_desc,
                           const tt::TensorDesc& updates_desc,
                           const tt::TensorDesc& output_desc,
                           tt::tt_metal::distributed::MeshDevice* mesh_device,
                           std::optional<ttnn::Tensor>* out) {
  std::vector<int64_t> operand_dims;
  std::vector<int64_t> start_indices_dims;
  std::vector<int64_t> updates_dims;
  std::vector<int64_t> output_dims;
  if (PJRT_Error* error = TensorDescDims(operand_desc, &operand_dims)) {
    return error;
  }
  if (PJRT_Error* error = TensorDescDims(start_indices_desc, &start_indices_dims)) {
    return error;
  }
  if (PJRT_Error* error = TensorDescDims(updates_desc, &updates_dims)) {
    return error;
  }
  if (PJRT_Error* error = TensorDescDims(output_desc, &output_dims)) {
    return error;
  }
  if (operand_dims != output_dims) {
    return InvalidArgument("scatter output shape must match operand shape");
  }
  if (operand_dims.size() != updates_dims.size()) {
    return InvalidArgument("scatter operand/update ranks must match");
  }

  int64_t scatter_dim = 0;
  if (PJRT_Error* error = ValidateSetScatterDimensionNumbers(
          scatter, operand_dims.size(), &scatter_dim)) {
    return error;
  }
  for (size_t dim = 0; dim < operand_dims.size(); ++dim) {
    if (dim == static_cast<size_t>(scatter_dim)) {
      continue;
    }
    if (updates_dims[dim] != operand_dims[dim]) {
      return InvalidArgument("scatter update shape must match operand shape outside scatter dimension");
    }
  }

  if (start_indices_dims.size() != 2 ||
      start_indices_dims[0] != updates_dims[static_cast<size_t>(scatter_dim)] ||
      start_indices_dims[1] != 1) {
    return Unimplemented("TTNN scatter currently supports start_indices shape [updates[scatter_dim], 1]");
  }

  PJRT_Buffer_Type operand_type = PJRT_Buffer_Type_INVALID;
  PJRT_Buffer_Type updates_type = PJRT_Buffer_Type_INVALID;
  PJRT_Buffer_Type start_indices_type = PJRT_Buffer_Type_INVALID;
  if (PJRT_Error* error = TensorDescBufferType(operand_desc, &operand_type)) {
    return error;
  }
  if (PJRT_Error* error = TensorDescBufferType(updates_desc, &updates_type)) {
    return error;
  }
  if (PJRT_Error* error = TensorDescBufferType(start_indices_desc, &start_indices_type)) {
    return error;
  }
  if (updates_type != operand_type) {
    return InvalidArgument("scatter updates dtype must match operand dtype");
  }
  if (start_indices_type != PJRT_Buffer_Type_S32) {
    return Unimplemented("TTNN scatter currently supports s32 start_indices");
  }

  try {
    ttnn::Tensor operand_tensor = ToDeviceTensor(operand, mesh_device, ttnn::ROW_MAJOR_LAYOUT);
    ttnn::Tensor updates_tensor = ToDeviceTensor(updates, mesh_device, ttnn::ROW_MAJOR_LAYOUT);
    ttnn::Tensor indices_tensor = ToDeviceTensor(start_indices, mesh_device, ttnn::ROW_MAJOR_LAYOUT);

    std::vector<int64_t> index_shape(updates_dims.size(), 1);
    index_shape[static_cast<size_t>(scatter_dim)] =
        updates_dims[static_cast<size_t>(scatter_dim)];
    indices_tensor = ttnn::reshape(indices_tensor,
                                   ttnn::Shape(U32SmallVector(index_shape)),
                                   ttnn::DRAM_MEMORY_CONFIG);
    ttnn::SmallVector<uint32_t> repetitions;
    repetitions.reserve(updates_dims.size());
    for (size_t dim = 0; dim < updates_dims.size(); ++dim) {
      if (index_shape[dim] <= 0 ||
          updates_dims[dim] % index_shape[dim] != 0) {
        return InvalidArgument("scatter index shape cannot be broadcast to updates shape");
      }
      repetitions.push_back(static_cast<uint32_t>(updates_dims[dim] / index_shape[dim]));
    }
    indices_tensor = ttnn::repeat(indices_tensor, repetitions, ttnn::DRAM_MEMORY_CONFIG);
    out->emplace(ttnn::scatter(operand_tensor,
                               static_cast<int32_t>(scatter_dim),
                               indices_tensor,
                               updates_tensor,
                               ttnn::DRAM_MEMORY_CONFIG));
    if (PJRT_Error* error = ValidateTensorMatchesDesc(out->value(), output_desc,
                                                      "scatter output")) {
      return error;
    }
  } catch (const std::exception& e) {
    return Internal(std::string("TTNN scatter failed: ") + e.what());
  } catch (...) {
    return Internal("TTNN scatter failed with unknown exception");
  }
  return nullptr;
}

PJRT_Error* ExecuteReduce(const tt::ReduceOp& reduce,
                          const ttnn::Tensor& input,
                          const tt::TensorDesc& output_desc,
                          tt::tt_metal::distributed::MeshDevice* mesh_device,
                          std::optional<ttnn::Tensor>* out) {
  if (reduce.input_ids_size() != 1 || reduce.init_value_ids_size() > 1) {
    return Unimplemented("TTNN reduce currently supports one input");
  }
  ttnn::SmallVector<int> dims;
  dims.reserve(static_cast<size_t>(reduce.dimensions_size()));
  for (int64_t dim : reduce.dimensions()) {
    if (dim < std::numeric_limits<int>::min() || dim > std::numeric_limits<int>::max()) {
      return InvalidArgument("reduce dimension is out of int range");
    }
    dims.push_back(static_cast<int>(dim));
  }
  if (PJRT_Error* error = NormalizeReduceDims(input.logical_shape().rank(), &dims)) {
    return error;
  }
  if (PJRT_Error* error = CompleteReduceDimsFromOutputShape(input, output_desc, &dims)) {
    return error;
  }
  if (PJRT_Error* error = NormalizeReduceDims(input.logical_shape().rank(), &dims)) {
    return error;
  }
  if (PJRT_Error* error = ValidateReduceOutputShape(input, dims, output_desc)) {
    return error;
  }
  if (dims.empty()) {
    out->emplace(input);
    return ValidateTensorMatchesDesc(out->value(), output_desc, "reduce output");
  }
  std::optional<std::variant<int, int64_t, ttnn::SmallVector<int>>> dim_arg;
  if (dims.size() == 1) {
    dim_arg = dims[0];
  } else if (!dims.empty()) {
    dim_arg = dims;
  }
  try {
    PJRT_Buffer_Type output_type = PJRT_Buffer_Type_INVALID;
    if (PJRT_Error* error = TensorDescBufferType(output_desc, &output_type)) {
      return error;
    }
    const std::optional<tt::tt_metal::DataType> output_dtype =
        TtnnDataTypeForPjrtBufferType(output_type);
    if (!output_dtype.has_value()) {
      return Unimplemented("reduce output dtype is not supported");
    }
    switch (reduce.reducer()) {
      case tt::ReduceOp::REDUCER_ADD:
        if (input.dtype() == tt::tt_metal::DataType::INT32) {
          ttnn::Tensor reduce_input =
              ToDeviceTensor(input, mesh_device, ttnn::TILE_LAYOUT);
          reduce_input = CastTensorIfNeeded(reduce_input,
                                            tt::tt_metal::DataType::FLOAT32,
                                            mesh_device);
          out->emplace(ttnn::sum(reduce_input, dim_arg, false,
                                 ttnn::DRAM_MEMORY_CONFIG));
          out->emplace(CastTensorIfNeeded(out->value(), *output_dtype,
                                          mesh_device));
        } else {
          out->emplace(ttnn::sum(input, dim_arg, false, ttnn::DRAM_MEMORY_CONFIG));
        }
        break;
      case tt::ReduceOp::REDUCER_MAX:
        out->emplace(ttnn::max(input, dim_arg, false, ttnn::DRAM_MEMORY_CONFIG));
        break;
      case tt::ReduceOp::REDUCER_MIN:
        out->emplace(ttnn::min(input, dim_arg, false, ttnn::DRAM_MEMORY_CONFIG));
        break;
      case tt::ReduceOp::REDUCER_MUL:
        if (dims.size() != 1) {
          return Unimplemented("TTNN prod reduce currently supports one dimension");
        }
        out->emplace(ttnn::prod(input, static_cast<int64_t>(dims[0]), false,
                                ttnn::DRAM_MEMORY_CONFIG));
        break;
      case tt::ReduceOp::REDUCER_AND: {
        if (input.dtype() != tt::tt_metal::DataType::UINT8 ||
            *output_dtype != tt::tt_metal::DataType::UINT8) {
          return Unimplemented("TTNN logical AND reduce currently supports PRED/U8 tensors only");
        }
        ttnn::Tensor reduce_input =
            ToDeviceTensor(input, mesh_device, ttnn::TILE_LAYOUT);
        reduce_input = CastTensorIfNeeded(reduce_input,
                                          tt::tt_metal::DataType::FLOAT32,
                                          mesh_device);
        out->emplace(ttnn::min(reduce_input, dim_arg, false,
                               ttnn::DRAM_MEMORY_CONFIG));
        out->emplace(CastTensorIfNeeded(out->value(), *output_dtype, mesh_device));
        break;
      }
      default:
        return Unimplemented("TTNN reduce reducer is not implemented");
    }
    if (PJRT_Error* error = ReshapeSingletonReduceOutput(input, dims, output_desc, out)) {
      return error;
    }
    if (PJRT_Error* error = ValidateTensorMatchesDesc(out->value(), output_desc, "reduce output")) {
      return error;
    }
  } catch (const std::exception& e) {
    return Internal(std::string("TTNN reduce failed: ") + e.what());
  } catch (...) {
    return Internal("TTNN reduce failed with unknown exception");
  }
  return nullptr;
}

PJRT_Error* ExecuteRmsNorm(const tt::RmsNormOp& rms_norm,
                           const ttnn::Tensor& input,
                           const ttnn::Tensor& weight,
                           const tt::TensorDesc& input_desc,
                           tt::tt_metal::distributed::MeshDevice* mesh_device,
                           std::optional<ttnn::Tensor>* out) {
  const float scale = F32FromBits(rms_norm.scale_bits());
  if (input_desc.dims_size() == 0) {
    return InvalidArgument("rms_norm input rank must be at least one");
  }
  const float expected_scale = 1.0f / static_cast<float>(input_desc.dims(input_desc.dims_size() - 1));
  if (scale != expected_scale) {
    return Unimplemented("TTNN rms_norm requires mean scale 1 / hidden_dim");
  }
  try {
    ttnn::Tensor input_tensor = ToDeviceTensor(input, mesh_device, ttnn::TILE_LAYOUT);
    ttnn::Tensor weight_tensor = ToDeviceTensor(weight, mesh_device, ttnn::TILE_LAYOUT);
    out->emplace(ttnn::rms_norm(input_tensor,
                                F32FromBits(rms_norm.bias_bits()),
                                weight_tensor,
                                std::nullopt,
                                std::nullopt,
                                ttnn::DRAM_MEMORY_CONFIG));
  } catch (const std::exception& e) {
    return Internal(std::string("TTNN rms_norm failed: ") + e.what());
  } catch (...) {
    return Internal("TTNN rms_norm failed with unknown exception");
  }
  return nullptr;
}

PJRT_Error* ExecuteRope(const tt::RopeOp& rope,
                        const ttnn::Tensor& input,
                        const ttnn::Tensor& cos,
                        const ttnn::Tensor& sin,
                        tt::tt_metal::distributed::MeshDevice* mesh_device,
                        std::optional<ttnn::Tensor>* out) {
  try {
    ttnn::Tensor input_tensor = ToDeviceTensor(input, mesh_device, ttnn::TILE_LAYOUT);
    ttnn::Tensor cos_tensor = ToDeviceTensor(cos, mesh_device, ttnn::TILE_LAYOUT);
    ttnn::Tensor sin_tensor = ToDeviceTensor(sin, mesh_device, ttnn::TILE_LAYOUT);
    const std::vector<int64_t> input_shape = TensorShapeVector(input_tensor);
    if (input_shape.empty()) {
      return InvalidArgument("rope input must have rank at least 1");
    }
    const int64_t head_dim = input_shape.back();
    if (head_dim <= 0) {
      return InvalidArgument("rope input head dimension must be positive");
    }
    ttnn::Tensor ttnn_cos;
    ttnn::Tensor ttnn_sin;
    if (PJRT_Error* error = PrepareRopeCacheForTtnn(cos_tensor, head_dim, &ttnn_cos)) {
      return error;
    }
    if (PJRT_Error* error = PrepareRopeCacheForTtnn(sin_tensor, head_dim, &ttnn_sin)) {
      return error;
    }

    if (input_shape.size() == 3) {
      const std::vector<int64_t> ttnn_input_shape = {
          1, input_shape[1], input_shape[0], input_shape[2]};
      ttnn::Tensor ttnn_input = ttnn::permute(input_tensor,
                                             I64SmallVector({1, 0, 2}),
                                             ttnn::DRAM_MEMORY_CONFIG);
      ttnn_input = ttnn::reshape(ttnn_input,
                                 ttnn::Shape(U32SmallVector(ttnn_input_shape)),
                                 ttnn::DRAM_MEMORY_CONFIG);
      ttnn::Tensor ttnn_output = ttnn::experimental::rotary_embedding(
          ttnn_input, ttnn_cos, ttnn_sin, std::nullopt, ttnn::DRAM_MEMORY_CONFIG);
      ttnn_output = ttnn::permute(ttnn_output, I64SmallVector({0, 2, 1, 3}),
                                  ttnn::DRAM_MEMORY_CONFIG);
      out->emplace(ttnn::reshape(ttnn_output,
                                 ttnn::Shape(U32SmallVector(input_shape)),
                                 ttnn::DRAM_MEMORY_CONFIG));
      return nullptr;
    }

    out->emplace(ttnn::experimental::rotary_embedding(
        input_tensor, ttnn_cos, ttnn_sin, std::nullopt, ttnn::DRAM_MEMORY_CONFIG));
  } catch (const std::exception& e) {
    return Internal(std::string("TTNN rope failed: ") + e.what());
  } catch (...) {
    return Internal("TTNN rope failed with unknown exception");
  }
  return nullptr;
}

PJRT_Error* PreparePagedSdpaDecodeQuery(
    const ttnn::Tensor& q,
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    ttnn::Tensor* out) {
  const std::vector<int64_t> dims = TensorShapeVector(q);
  if (dims.size() != 4) {
    return InvalidArgument("paged_sdpa_decode q must be rank 4, got " +
                           DimsToString(dims));
  }
  if (dims[0] != 1) {
    return Unimplemented("paged_sdpa_decode currently supports one decode query, got " +
                         DimsToString(dims));
  }
  *out = ToDeviceTensor(q, mesh_device, ttnn::TILE_LAYOUT);
  return nullptr;
}

PJRT_Error* SplitPagedFusedKvCache(
    const ttnn::Tensor& fused_kv,
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    ttnn::Tensor* k_cache,
    ttnn::Tensor* v_cache) {
  const std::vector<int64_t> dims = TensorShapeVector(fused_kv);
  if (dims.size() != 5) {
    return InvalidArgument("paged_sdpa_decode fused KV cache must be rank 5, got " +
                           DimsToString(dims));
  }
  const int64_t pages = dims[0];
  const int64_t page_size = dims[1];
  const int64_t kv_heads = dims[2];
  const int64_t packing = dims[3];
  const int64_t head_dim = dims[4];
  if (pages < 0 || page_size < 0 || kv_heads < 0 || head_dim < 0) {
    return InvalidArgument("paged_sdpa_decode fused KV dimensions must be non-negative");
  }
  if (packing != 2) {
    return InvalidArgument("paged_sdpa_decode fused KV cache expects packing dimension 2, got " +
                           std::to_string(packing));
  }
  if (page_size != 0 && pages > std::numeric_limits<int64_t>::max() / page_size) {
    return ResourceExhausted("paged_sdpa_decode fused KV token count overflow");
  }
  const int64_t cache_tokens = pages * page_size;

  ttnn::Tensor compact = ToDeviceTensor(fused_kv, mesh_device, ttnn::TILE_LAYOUT);
  compact = ttnn::reshape(
      compact,
      ttnn::Shape(U32SmallVector({cache_tokens, kv_heads * packing, head_dim})),
      ttnn::DRAM_MEMORY_CONFIG);

  ttnn::Tensor k = ttnn::slice(
      compact,
      I32SmallVector({0, 0, 0}),
      I32SmallVector({cache_tokens, kv_heads * packing, head_dim}),
      I32SmallVector({1, 2, 1}),
      ttnn::DRAM_MEMORY_CONFIG);
  ttnn::Tensor v = ttnn::slice(
      compact,
      I32SmallVector({0, 1, 0}),
      I32SmallVector({cache_tokens, kv_heads * packing, head_dim}),
      I32SmallVector({1, 2, 1}),
      ttnn::DRAM_MEMORY_CONFIG);

  k = ttnn::reshape(k,
                    ttnn::Shape(U32SmallVector({pages, page_size, kv_heads, head_dim})),
                    ttnn::DRAM_MEMORY_CONFIG);
  v = ttnn::reshape(v,
                    ttnn::Shape(U32SmallVector({pages, page_size, kv_heads, head_dim})),
                    ttnn::DRAM_MEMORY_CONFIG);
  *k_cache = ttnn::permute(k, I64SmallVector({0, 2, 1, 3}), ttnn::DRAM_MEMORY_CONFIG);
  *v_cache = ttnn::permute(v, I64SmallVector({0, 2, 1, 3}), ttnn::DRAM_MEMORY_CONFIG);
  return nullptr;
}

PJRT_Error* ExecutePagedSdpaDecode(
    const tt::PagedSdpaDecodeOp& sdpa,
    const ttnn::Tensor& q,
    const ttnn::Tensor& fused_kv,
    const ttnn::Tensor& page_table,
    const ttnn::Tensor& cur_pos,
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    std::optional<ttnn::Tensor>* out) {
  try {
    ttnn::Tensor q_tensor;
    if (PJRT_Error* error = PreparePagedSdpaDecodeQuery(q, mesh_device, &q_tensor)) {
      return error;
    }

    ttnn::Tensor k_cache;
    ttnn::Tensor v_cache;
    if (PJRT_Error* error = SplitPagedFusedKvCache(
            fused_kv, mesh_device, &k_cache, &v_cache)) {
      return error;
    }

    ttnn::Tensor page_table_tensor =
        ToDeviceTensor(page_table, mesh_device, ttnn::ROW_MAJOR_LAYOUT);
    page_table_tensor = CastTensorIfNeeded(page_table_tensor,
                                           tt::tt_metal::DataType::INT32,
                                           mesh_device);
    ttnn::Tensor cur_pos_tensor =
        ToDeviceTensor(cur_pos, mesh_device, ttnn::ROW_MAJOR_LAYOUT);
    cur_pos_tensor = CastTensorIfNeeded(cur_pos_tensor,
                                        tt::tt_metal::DataType::INT32,
                                        mesh_device);

    const std::vector<int64_t> fused_dims = TensorShapeVector(fused_kv);
    ttnn::operations::transformer::SDPAProgramConfig program_config{
        q_tensor.device()->compute_with_storage_grid_size(),
        std::nullopt,
        32,
        32,
        std::nullopt,
        1};
    ttnn::DeviceComputeKernelConfig compute_kernel_config =
        ttnn::init_device_compute_kernel_config(
            q_tensor.device()->arch(),
            std::nullopt,
            tt::tt_metal::MathFidelity::HiFi2,
            true,
            false,
            false);

    std::optional<const ttnn::Tensor> cur_pos_arg(cur_pos_tensor);
    std::optional<uint32_t> sliding_window_size = std::nullopt;
    if (sdpa.sliding_window_size() != 0) {
      sliding_window_size = sdpa.sliding_window_size();
    }

    out->emplace(ttnn::transformer::paged_scaled_dot_product_attention_decode(
        q_tensor,
        k_cache,
        v_cache,
        page_table_tensor,
        /*is_causal=*/true,
        /*attn_mask=*/std::nullopt,
        cur_pos_arg,
        /*attention_sink=*/std::nullopt,
        sdpa.scale(),
        sliding_window_size,
        ttnn::DRAM_MEMORY_CONFIG,
        program_config,
        compute_kernel_config,
        static_cast<uint32_t>(fused_dims[1]),
        static_cast<uint32_t>(fused_dims[2]),
        std::nullopt));
  } catch (const std::exception& e) {
    return Internal(std::string("TTNN paged_sdpa_decode failed: ") + e.what());
  } catch (...) {
    return Internal("TTNN paged_sdpa_decode failed with unknown exception");
  }
  return nullptr;
}

PJRT_Error* ExecuteBitwiseBinary(const tt::BitwiseBinaryOp& bitwise,
                                 const ttnn::Tensor& lhs,
                                 const ttnn::Tensor& rhs,
                                 const tt::TensorDesc& output_desc,
                                 tt::tt_metal::distributed::MeshDevice* mesh_device,
                                 std::optional<ttnn::Tensor>* out) {
  PJRT_Buffer_Type output_type = PJRT_Buffer_Type_INVALID;
  if (PJRT_Error* error = TensorDescBufferType(output_desc, &output_type)) {
    return error;
  }
  const std::optional<tt::tt_metal::DataType> output_dtype =
      TtnnDataTypeForPjrtBufferType(output_type);
  if (!output_dtype.has_value()) {
    return Unimplemented("bitwise_binary output dtype is not supported");
  }

  try {
    ttnn::Tensor lhs_tensor = ToDeviceTensor(lhs, mesh_device, ttnn::TILE_LAYOUT);
    ttnn::Tensor rhs_tensor = ToDeviceTensor(rhs, mesh_device, ttnn::TILE_LAYOUT);
    if (lhs_tensor.dtype() == tt::tt_metal::DataType::UINT8 &&
        rhs_tensor.dtype() == tt::tt_metal::DataType::UINT8) {
      lhs_tensor = CastTensorIfNeeded(lhs_tensor, tt::tt_metal::DataType::UINT32, mesh_device);
      rhs_tensor = CastTensorIfNeeded(rhs_tensor, tt::tt_metal::DataType::UINT32, mesh_device);
    }
    switch (bitwise.kind()) {
      case tt::BitwiseBinaryOp::KIND_AND:
        out->emplace(ttnn::bitwise_and(lhs_tensor, rhs_tensor, ttnn::DRAM_MEMORY_CONFIG));
        break;
      case tt::BitwiseBinaryOp::KIND_OR:
        out->emplace(ttnn::bitwise_or(lhs_tensor, rhs_tensor, ttnn::DRAM_MEMORY_CONFIG));
        break;
      case tt::BitwiseBinaryOp::KIND_XOR:
        out->emplace(ttnn::bitwise_xor(lhs_tensor, rhs_tensor, ttnn::DRAM_MEMORY_CONFIG));
        break;
      case tt::BitwiseBinaryOp::KIND_SHIFT_LEFT:
        out->emplace(ttnn::bitwise_left_shift(lhs_tensor, rhs_tensor, ttnn::DRAM_MEMORY_CONFIG));
        break;
      case tt::BitwiseBinaryOp::KIND_SHIFT_RIGHT_LOGICAL:
      case tt::BitwiseBinaryOp::KIND_SHIFT_RIGHT_ARITHMETIC:
        out->emplace(ttnn::bitwise_right_shift(lhs_tensor, rhs_tensor, ttnn::DRAM_MEMORY_CONFIG));
        break;
      default:
        return Unimplemented("unknown bitwise_binary kind");
    }
    out->emplace(CastTensorIfNeeded(out->value(), *output_dtype, mesh_device));
    if (PJRT_Error* error = ValidateTensorMatchesDesc(out->value(), output_desc,
                                                      "bitwise_binary output")) {
      return error;
    }
  } catch (const std::exception& e) {
    return Internal(std::string("TTNN bitwise_binary failed: ") + e.what());
  } catch (...) {
    return Internal("TTNN bitwise_binary failed with unknown exception");
  }
  return nullptr;
}

PJRT_Error* ExecuteProgram(const PJRT_LoadedExecutable* executable,
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
  if (target_device == nullptr || target_device->default_memory == nullptr) {
    return InvalidArgument("no execute device available");
  }

  std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_device =
      GetTtMetalMeshDevice(target_device->local_hardware_id);
  std::vector<std::optional<ttnn::Tensor>> values(static_cast<size_t>(program.values_size()));
  for (const tt::Op& op : program.ops()) {
    if (op.output_id() >= static_cast<uint32_t>(values.size())) {
      return Internal("executable op output id is out of bounds");
    }
    const tt::ValueDesc& output_value_desc = program.values(op.output_id());
    if (!output_value_desc.has_tensor()) {
      return Internal("executable op output is missing tensor metadata");
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
        if (argument->IsDeleted()) {
          return FailedPrecondition("argument buffer has been deleted");
        }
        if (target_device != nullptr && argument->device != nullptr && argument->device != target_device) {
          return InvalidArgument("all input buffers and execute_device must be on the same device");
        }
        if (PJRT_Error* error = ArgumentTensorForParameter(
                *argument, output_value_desc.tensor(), mesh_device.get(),
                &values[op.output_id()])) {
          return error;
        }
        break;
      }
      case tt::Op::kConstant: {
        if (PJRT_Error* error = CreateConstantTensor(op.constant(), output_value_desc.tensor(),
                                                     mesh_device.get(),
                                                     &values[op.output_id()])) {
          return error;
        }
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
        const ttnn::Tensor* input = nullptr;
        if (PJRT_Error* error = GetValueTensor(values, input_id, "custom_call", &input)) {
          return error;
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
        values[op.output_id()] = *input;
        break;
      }
      case tt::Op::kMatmul: {
        const tt::MatmulOp& matmul = op.matmul();
        const uint32_t lhs_id = matmul.lhs_id();
        const uint32_t rhs_id = matmul.rhs_id();
        const ttnn::Tensor* lhs = nullptr;
        const ttnn::Tensor* rhs = nullptr;
        if (PJRT_Error* error = GetValueTensor(values, lhs_id, "matmul", &lhs)) {
          return error;
        }
        if (PJRT_Error* error = GetValueTensor(values, rhs_id, "matmul", &rhs)) {
          return error;
        }
        if (lhs_id >= static_cast<uint32_t>(program.values_size()) ||
            rhs_id >= static_cast<uint32_t>(program.values_size())) {
          return Internal("matmul input metadata id is out of bounds");
        }
        const tt::ValueDesc& lhs_desc = program.values(lhs_id);
        const tt::ValueDesc& rhs_desc = program.values(rhs_id);
        const tt::ValueDesc& output_desc = program.values(op.output_id());
        if (!lhs_desc.has_tensor() || !rhs_desc.has_tensor() || !output_desc.has_tensor()) {
          return Internal("matmul value is missing tensor metadata");
        }

        std::optional<ttnn::Tensor> matmul_result;
        if (PJRT_Error* error = ExecuteTtnnMatmul(matmul, *lhs, *rhs,
                                                  lhs_desc.tensor(), rhs_desc.tensor(),
                                                  output_desc.tensor(), mesh_device.get(),
                                                  &matmul_result)) {
          return error;
        }
        if (matmul.has_top_k_epilogue()) {
          const tt::MatmulTopKEpilogue& epilogue = matmul.top_k_epilogue();
          if (epilogue.indices_id() >= static_cast<uint32_t>(program.values_size())) {
            return Internal("matmul top_k epilogue indices metadata id is out of bounds");
          }
          const tt::ValueDesc& indices_desc = program.values(epilogue.indices_id());
          if (!indices_desc.has_tensor()) {
            return Internal("matmul top_k indices value is missing tensor metadata");
          }
          tt::TopKOp top_k;
          top_k.set_operand_id(epilogue.matmul_output_id());
          top_k.set_indices_id(epilogue.indices_id());
          top_k.set_k(epilogue.k());
          if (PJRT_Error* error = ExecuteTopK(top_k, *matmul_result, output_value_desc.tensor(),
                                              indices_desc.tensor(), mesh_device.get(),
                                              &values[op.output_id()],
                                              &values[epilogue.indices_id()])) {
            return error;
          }
        } else {
          values[op.output_id()] = std::move(*matmul_result);
        }
        break;
      }
      case tt::Op::kTopK: {
        const tt::TopKOp& top_k = op.top_k();
        const ttnn::Tensor* input = nullptr;
        if (PJRT_Error* error = GetValueTensor(values, top_k.operand_id(), "top_k", &input)) {
          return error;
        }
        if (top_k.indices_id() >= static_cast<uint32_t>(program.values_size())) {
          return Internal("top_k indices metadata id is out of bounds");
        }
        const tt::ValueDesc& indices_desc = program.values(top_k.indices_id());
        if (!indices_desc.has_tensor()) {
          return Internal("top_k indices value is missing tensor metadata");
        }
        if (PJRT_Error* error = ExecuteTopK(top_k, *input, output_value_desc.tensor(),
                                            indices_desc.tensor(), mesh_device.get(),
                                            &values[op.output_id()],
                                            &values[top_k.indices_id()])) {
          return error;
        }
        break;
      }
      case tt::Op::kFusedElementwise: {
        if (PJRT_Error* error = ExecuteFusedElementwise(op.fused_elementwise(),
                                                        output_value_desc.tensor(),
                                                        values, mesh_device.get(),
                                                        &values[op.output_id()])) {
          return error;
        }
        break;
      }
      case tt::Op::kReshape: {
        const ttnn::Tensor* input = nullptr;
        if (PJRT_Error* error = GetValueTensor(values, op.reshape().operand_id(), "reshape", &input)) {
          return error;
        }
        ttnn::Shape shape;
        if (PJRT_Error* error = ShapeFromTensorDesc(output_value_desc.tensor(), &shape)) {
          return error;
        }
        PJRT_Buffer_Type output_type = PJRT_Buffer_Type_INVALID;
        if (PJRT_Error* error = TensorDescBufferType(output_value_desc.tensor(), &output_type)) {
          return error;
        }
        const std::optional<tt::tt_metal::DataType> output_dtype =
            TtnnDataTypeForPjrtBufferType(output_type);
        if (!output_dtype.has_value()) {
          return Unimplemented("reshape output dtype is not supported");
        }
        try {
          ttnn::Tensor reshape_input = *input;
          if (input->dtype() == tt::tt_metal::DataType::UINT8 &&
              *output_dtype == tt::tt_metal::DataType::UINT8) {
            reshape_input =
                ToDeviceTensor(*input, mesh_device.get(), ttnn::TILE_LAYOUT);
            reshape_input = CastTensorIfNeeded(reshape_input,
                                               tt::tt_metal::DataType::UINT32,
                                               mesh_device.get());
          }
          ttnn::Tensor reshaped =
              ttnn::reshape(reshape_input, shape, ttnn::DRAM_MEMORY_CONFIG);
          values[op.output_id()] =
              CastTensorIfNeeded(reshaped, *output_dtype, mesh_device.get());
          if (PJRT_Error* error = ValidateTensorMatchesDesc(values[op.output_id()].value(),
                                                            output_value_desc.tensor(),
                                                            "reshape output")) {
            return error;
          }
        } catch (const std::exception& e) {
          return Internal(std::string("TTNN reshape failed from ") +
                          DimsToString(TensorShapeVector(*input)) + " to " +
                          TensorDescShapeString(output_value_desc.tensor()) + ": " +
                          e.what());
        }
        break;
      }
      case tt::Op::kSlice: {
        const tt::SliceOp& slice = op.slice();
        const ttnn::Tensor* input = nullptr;
        if (PJRT_Error* error = GetValueTensor(values, slice.operand_id(), "slice", &input)) {
          return error;
        }
        try {
          values[op.output_id()] = ttnn::slice(*input,
                                               I32SmallVector(slice.start_indices()),
                                               I32SmallVector(slice.limit_indices()),
                                               I32SmallVector(slice.strides()),
                                               ttnn::DRAM_MEMORY_CONFIG);
        } catch (const std::exception& e) {
          return Internal(std::string("TTNN slice failed: ") + e.what());
        }
        break;
      }
      case tt::Op::kTranspose: {
        const tt::TransposeOp& transpose = op.transpose();
        const ttnn::Tensor* input = nullptr;
        if (PJRT_Error* error = GetValueTensor(values, transpose.operand_id(), "transpose", &input)) {
          return error;
        }
        try {
          values[op.output_id()] = ttnn::permute(*input,
                                                 Int64SmallVector(transpose.permutation()),
                                                 ttnn::DRAM_MEMORY_CONFIG);
        } catch (const std::exception& e) {
          return Internal(std::string("TTNN transpose/permute failed: ") + e.what());
        }
        break;
      }
      case tt::Op::kConcatenate: {
        const tt::ConcatenateOp& concatenate = op.concatenate();
        std::vector<ttnn::Tensor> inputs;
        inputs.reserve(static_cast<size_t>(concatenate.input_ids_size()));
        for (uint32_t input_id : concatenate.input_ids()) {
          const ttnn::Tensor* input = nullptr;
          if (PJRT_Error* error = GetValueTensor(values, input_id, "concatenate", &input)) {
            return error;
          }
          inputs.push_back(*input);
        }
        try {
          values[op.output_id()] = ttnn::concat(inputs,
                                                static_cast<int>(concatenate.dimension()),
                                                ttnn::DRAM_MEMORY_CONFIG);
        } catch (const std::exception& e) {
          return Internal(std::string("TTNN concatenate failed: ") + e.what());
        }
        break;
      }
      case tt::Op::kBroadcastInDim: {
        const ttnn::Tensor* input = nullptr;
        if (PJRT_Error* error = GetValueTensor(values, op.broadcast_in_dim().operand_id(),
                                               "broadcast_in_dim", &input)) {
          return error;
        }
        if (PJRT_Error* error = BroadcastTensorInDim(*input, op.broadcast_in_dim(),
                                                     output_value_desc.tensor(),
                                                     mesh_device.get(),
                                                     &values[op.output_id()])) {
          return error;
        }
        break;
      }
      case tt::Op::kSelect: {
        const tt::SelectOp& select = op.select();
        const ttnn::Tensor* pred = nullptr;
        const ttnn::Tensor* on_true = nullptr;
        const ttnn::Tensor* on_false = nullptr;
        if (PJRT_Error* error = GetValueTensor(values, select.pred_id(), "select", &pred)) {
          return error;
        }
        if (PJRT_Error* error = GetValueTensor(values, select.on_true_id(), "select", &on_true)) {
          return error;
        }
        if (PJRT_Error* error = GetValueTensor(values, select.on_false_id(), "select", &on_false)) {
          return error;
        }
        try {
          PJRT_Buffer_Type output_type = PJRT_Buffer_Type_INVALID;
          if (PJRT_Error* error = TensorDescBufferType(output_value_desc.tensor(), &output_type)) {
            return error;
          }
          const std::optional<tt::tt_metal::DataType> output_dtype =
              TtnnDataTypeForPjrtBufferType(output_type);
          if (!output_dtype.has_value()) {
            return Unimplemented("select output dtype is not supported");
          }
          ttnn::Tensor true_tensor =
              ToDeviceTensor(*on_true, mesh_device.get(), ttnn::TILE_LAYOUT);
          true_tensor = CastTensorIfNeeded(true_tensor, *output_dtype, mesh_device.get());
          ttnn::Tensor false_tensor =
              ToDeviceTensor(*on_false, mesh_device.get(), ttnn::TILE_LAYOUT);
          false_tensor = CastTensorIfNeeded(false_tensor, *output_dtype, mesh_device.get());
          ttnn::Tensor pred_tensor = ToDeviceTensor(*pred, mesh_device.get(), ttnn::TILE_LAYOUT);
          pred_tensor = CastTensorIfNeeded(pred_tensor, *output_dtype, mesh_device.get());
          values[op.output_id()] = ttnn::where(pred_tensor, true_tensor, false_tensor,
                                               ttnn::DRAM_MEMORY_CONFIG);
          if (PJRT_Error* error = ValidateTensorMatchesDesc(values[op.output_id()].value(),
                                                            output_value_desc.tensor(),
                                                            "select output")) {
            return error;
          }
        } catch (const std::exception& e) {
          return Internal(std::string("TTNN select failed: ") + e.what());
        }
        break;
      }
      case tt::Op::kGather: {
        const tt::GatherOp& gather = op.gather();
        const ttnn::Tensor* operand = nullptr;
        const ttnn::Tensor* start_indices = nullptr;
        if (PJRT_Error* error = GetValueTensor(values, gather.operand_id(), "gather", &operand)) {
          return error;
        }
        if (PJRT_Error* error = GetValueTensor(values, gather.start_indices_id(), "gather", &start_indices)) {
          return error;
        }
        if (gather.operand_id() >= static_cast<uint32_t>(program.values_size())) {
          return Internal("gather operand metadata id is out of bounds");
        }
        const tt::ValueDesc& operand_desc = program.values(gather.operand_id());
        if (!operand_desc.has_tensor()) {
          return Internal("gather operand is missing tensor metadata");
        }
        if (PJRT_Error* error = ExecuteGather(gather, *operand, *start_indices,
                                              operand_desc.tensor(),
                                              output_value_desc.tensor(),
                                              mesh_device.get(),
                                              &values[op.output_id()])) {
          return error;
        }
        break;
      }
      case tt::Op::kScatter: {
        const tt::ScatterOp& scatter = op.scatter();
        const ttnn::Tensor* operand = nullptr;
        const ttnn::Tensor* start_indices = nullptr;
        const ttnn::Tensor* updates = nullptr;
        if (PJRT_Error* error = GetValueTensor(values, scatter.operand_id(), "scatter", &operand)) {
          return error;
        }
        if (PJRT_Error* error = GetValueTensor(values, scatter.start_indices_id(), "scatter", &start_indices)) {
          return error;
        }
        if (PJRT_Error* error = GetValueTensor(values, scatter.updates_id(), "scatter", &updates)) {
          return error;
        }
        if (scatter.operand_id() >= static_cast<uint32_t>(program.values_size()) ||
            scatter.start_indices_id() >= static_cast<uint32_t>(program.values_size()) ||
            scatter.updates_id() >= static_cast<uint32_t>(program.values_size())) {
          return Internal("scatter operand metadata id is out of bounds");
        }
        const tt::ValueDesc& operand_desc = program.values(scatter.operand_id());
        const tt::ValueDesc& start_indices_desc = program.values(scatter.start_indices_id());
        const tt::ValueDesc& updates_desc = program.values(scatter.updates_id());
        if (!operand_desc.has_tensor() || !start_indices_desc.has_tensor() ||
            !updates_desc.has_tensor()) {
          return Internal("scatter operand metadata is missing tensor metadata");
        }
        if (PJRT_Error* error = ExecuteScatter(scatter, *operand, *start_indices, *updates,
                                               operand_desc.tensor(),
                                               start_indices_desc.tensor(),
                                               updates_desc.tensor(),
                                               output_value_desc.tensor(),
                                               mesh_device.get(),
                                               &values[op.output_id()])) {
          return error;
        }
        break;
      }
      case tt::Op::kIota: {
        const uint64_t dim = op.iota().iota_dimension();
        std::vector<int64_t> output_dims;
        if (PJRT_Error* error = TensorDescDims(output_value_desc.tensor(), &output_dims)) {
          return error;
        }
        if (dim >= output_dims.size()) {
          return InvalidArgument("iota dimension is out of bounds");
        }
        PJRT_Buffer_Type output_type = PJRT_Buffer_Type_INVALID;
        if (PJRT_Error* error = TensorDescBufferType(output_value_desc.tensor(), &output_type)) {
          return error;
        }
        const std::optional<tt::tt_metal::DataType> dtype =
            TtnnDataTypeForPjrtBufferType(output_type);
        if (!dtype.has_value()) {
          return Unimplemented("iota dtype is not supported");
        }
        try {
          ttnn::Tensor range = ttnn::arange(0,
                                            output_dims[static_cast<size_t>(dim)],
                                            1,
                                            *dtype,
                                            std::ref(*mesh_device),
                                            ttnn::DRAM_MEMORY_CONFIG);
          if (output_dims.size() == 1) {
            values[op.output_id()] = range;
            break;
          }
          std::vector<int64_t> reshaped_dims(output_dims.size(), 1);
          reshaped_dims[static_cast<size_t>(dim)] = output_dims[static_cast<size_t>(dim)];
          ttnn::Tensor reshaped = ttnn::reshape(range,
                                                ttnn::Shape(U32SmallVector(reshaped_dims)),
                                                ttnn::DRAM_MEMORY_CONFIG);
          std::optional<ttnn::Tensor> broadcasted;
          if (PJRT_Error* error = BroadcastTrailingDims(reshaped,
                                                        output_value_desc.tensor(),
                                                        mesh_device.get(),
                                                        &broadcasted)) {
            return error;
          }
          values[op.output_id()] = std::move(*broadcasted);
        } catch (const std::exception& e) {
          return Internal(std::string("TTNN iota failed: ") + e.what());
        }
        break;
      }
      case tt::Op::kReduce: {
        const tt::ReduceOp& reduce = op.reduce();
        if (reduce.input_ids_size() != 1) {
          return Unimplemented("reduce with multiple inputs is not implemented");
        }
        const ttnn::Tensor* input = nullptr;
        if (PJRT_Error* error = GetValueTensor(values, reduce.input_ids(0), "reduce", &input)) {
          return error;
        }
        if (PJRT_Error* error = ExecuteReduce(reduce, *input,
                                              output_value_desc.tensor(),
                                              mesh_device.get(),
                                              &values[op.output_id()])) {
          return error;
        }
        break;
      }
      case tt::Op::kReduceWindow: {
        const tt::ReduceWindowOp& reduce_window = op.reduce_window();
        if (reduce_window.input_ids_size() != 1) {
          return Unimplemented("reduce_window with multiple inputs is not implemented");
        }
        const uint32_t input_id = reduce_window.input_ids(0);
        const ttnn::Tensor* input = nullptr;
        if (PJRT_Error* error = GetValueTensor(values, input_id, "reduce_window", &input)) {
          return error;
        }
        if (input_id >= static_cast<uint32_t>(program.values_size())) {
          return Internal("reduce_window input metadata id is out of bounds");
        }
        const tt::ValueDesc& input_desc = program.values(input_id);
        if (!input_desc.has_tensor()) {
          return Internal("reduce_window input is missing tensor metadata");
        }
        std::vector<int64_t> input_dims;
        std::vector<int64_t> output_dims;
        if (PJRT_Error* error = TensorDescDims(input_desc.tensor(), &input_dims)) {
          return error;
        }
        if (PJRT_Error* error = TensorDescDims(output_value_desc.tensor(), &output_dims)) {
          return error;
        }
        const std::vector<int64_t> ones(input_dims.size(), 1);
        const std::vector<int64_t> zeros(input_dims.size(), 0);
        if (reduce_window.reducer() == tt::ReduceOp::REDUCER_ADD &&
            input_dims == output_dims &&
            RepeatedFieldEquals(reduce_window.window_dimensions(), ones) &&
            RepeatedFieldEquals(reduce_window.window_strides(), ones) &&
            RepeatedFieldEquals(reduce_window.base_dilations(), ones) &&
            RepeatedFieldEquals(reduce_window.window_dilations(), ones) &&
            RepeatedFieldEquals(reduce_window.padding_low(), zeros) &&
            RepeatedFieldEquals(reduce_window.padding_high(), zeros)) {
          values[op.output_id()] = *input;
          break;
        }
        if (reduce_window.reducer() == tt::ReduceOp::REDUCER_ADD &&
            input_dims == output_dims &&
            RepeatedFieldEquals(reduce_window.window_strides(), ones) &&
            RepeatedFieldEquals(reduce_window.base_dilations(), ones) &&
            RepeatedFieldEquals(reduce_window.window_dilations(), ones) &&
            RepeatedFieldEquals(reduce_window.padding_high(), zeros) &&
            reduce_window.window_dimensions_size() ==
                static_cast<int>(input_dims.size()) &&
            reduce_window.padding_low_size() ==
                static_cast<int>(input_dims.size())) {
          int32_t cumsum_dim = -1;
          for (size_t dim = 0; dim < input_dims.size(); ++dim) {
            const int64_t window_dim =
                reduce_window.window_dimensions(static_cast<int>(dim));
            const int64_t pad_low =
                reduce_window.padding_low(static_cast<int>(dim));
            if (window_dim == 1 && pad_low == 0) {
              continue;
            }
            if (window_dim == input_dims[dim] &&
                pad_low == input_dims[dim] - 1 &&
                cumsum_dim < 0) {
              cumsum_dim = static_cast<int32_t>(dim);
              continue;
            }
            cumsum_dim = -1;
            break;
          }
          if (cumsum_dim >= 0) {
            PJRT_Buffer_Type output_type = PJRT_Buffer_Type_INVALID;
            if (PJRT_Error* error =
                    TensorDescBufferType(output_value_desc.tensor(), &output_type)) {
              return error;
            }
            const std::optional<tt::tt_metal::DataType> output_dtype =
                TtnnDataTypeForPjrtBufferType(output_type);
            if (!output_dtype.has_value()) {
              return Unimplemented("reduce_window cumsum output dtype is not supported");
            }
            try {
              ttnn::Tensor cumsum_input =
                  ToDeviceTensor(*input, mesh_device.get(), ttnn::TILE_LAYOUT);
              tt::tt_metal::DataType cumsum_dtype = cumsum_input.dtype();
              if (cumsum_dtype == tt::tt_metal::DataType::INT32) {
                cumsum_input = CastTensorIfNeeded(
                    cumsum_input, tt::tt_metal::DataType::FLOAT32,
                    mesh_device.get());
                cumsum_dtype = tt::tt_metal::DataType::FLOAT32;
              }
              ttnn::Tensor result =
                  ttnn::cumsum(cumsum_input, cumsum_dim, cumsum_dtype, false,
                               std::nullopt, ttnn::DRAM_MEMORY_CONFIG);
              values[op.output_id()] =
                  CastTensorIfNeeded(result, *output_dtype, mesh_device.get());
              if (PJRT_Error* error =
                      ValidateTensorMatchesDesc(values[op.output_id()].value(),
                                                output_value_desc.tensor(),
                                                "reduce_window output")) {
                return error;
              }
              break;
            } catch (const std::exception& e) {
              return Internal(std::string("TTNN reduce_window cumsum failed: ") +
                              e.what());
            }
          }
        }
        return Unimplemented(
            "C++ execution does not support reduce_window: input=" +
            TensorDescShapeString(input_desc.tensor()) +
            " output=" + TensorDescShapeString(output_value_desc.tensor()) +
            " window_dimensions=" +
            RepeatedDimsToString(reduce_window.window_dimensions()) +
            " window_strides=" +
            RepeatedDimsToString(reduce_window.window_strides()) +
            " base_dilations=" +
            RepeatedDimsToString(reduce_window.base_dilations()) +
            " window_dilations=" +
            RepeatedDimsToString(reduce_window.window_dilations()) +
            " padding_low=" +
            RepeatedDimsToString(reduce_window.padding_low()) +
            " padding_high=" +
            RepeatedDimsToString(reduce_window.padding_high()) +
            " reducer=" + std::to_string(reduce_window.reducer()));
      }
      case tt::Op::kRmsNorm: {
        const tt::RmsNormOp& rms_norm = op.rms_norm();
        const ttnn::Tensor* input = nullptr;
        const ttnn::Tensor* weight = nullptr;
        if (PJRT_Error* error = GetValueTensor(values, rms_norm.input_id(), "rms_norm", &input)) {
          return error;
        }
        if (PJRT_Error* error = GetValueTensor(values, rms_norm.weight_id(), "rms_norm", &weight)) {
          return error;
        }
        if (rms_norm.input_id() >= static_cast<uint32_t>(program.values_size())) {
          return Internal("rms_norm input metadata id is out of bounds");
        }
        const tt::ValueDesc& input_desc = program.values(rms_norm.input_id());
        if (!input_desc.has_tensor()) {
          return Internal("rms_norm input is missing tensor metadata");
        }
        if (PJRT_Error* error = ExecuteRmsNorm(rms_norm, *input, *weight,
                                               input_desc.tensor(), mesh_device.get(),
                                               &values[op.output_id()])) {
          return error;
        }
        break;
      }
      case tt::Op::kRope: {
        const tt::RopeOp& rope = op.rope();
        const ttnn::Tensor* input = nullptr;
        const ttnn::Tensor* cos = nullptr;
        const ttnn::Tensor* sin = nullptr;
        if (PJRT_Error* error = GetValueTensor(values, rope.input_id(), "rope", &input)) {
          return error;
        }
        if (PJRT_Error* error = GetValueTensor(values, rope.cos_id(), "rope", &cos)) {
          return error;
        }
        if (PJRT_Error* error = GetValueTensor(values, rope.sin_id(), "rope", &sin)) {
          return error;
        }
        if (PJRT_Error* error = ExecuteRope(rope, *input, *cos, *sin,
                                            mesh_device.get(),
                                            &values[op.output_id()])) {
          return error;
        }
        break;
      }
      case tt::Op::kPagedSdpaDecode: {
        const tt::PagedSdpaDecodeOp& sdpa = op.paged_sdpa_decode();
        const ttnn::Tensor* q = nullptr;
        const ttnn::Tensor* fused_kv = nullptr;
        const ttnn::Tensor* page_table = nullptr;
        const ttnn::Tensor* cur_pos = nullptr;
        if (PJRT_Error* error = GetValueTensor(values, sdpa.q_id(), "paged_sdpa_decode", &q)) {
          return error;
        }
        if (PJRT_Error* error = GetValueTensor(values, sdpa.fused_kv_cache_id(),
                                               "paged_sdpa_decode", &fused_kv)) {
          return error;
        }
        if (PJRT_Error* error = GetValueTensor(values, sdpa.page_table_id(),
                                               "paged_sdpa_decode", &page_table)) {
          return error;
        }
        if (PJRT_Error* error = GetValueTensor(values, sdpa.cur_pos_id(),
                                               "paged_sdpa_decode", &cur_pos)) {
          return error;
        }
        if (PJRT_Error* error = ExecutePagedSdpaDecode(
                sdpa, *q, *fused_kv, *page_table, *cur_pos, mesh_device.get(),
                &values[op.output_id()])) {
          return error;
        }
        break;
      }
      case tt::Op::kBitwiseBinary: {
        const tt::BitwiseBinaryOp& bitwise = op.bitwise_binary();
        const ttnn::Tensor* lhs = nullptr;
        const ttnn::Tensor* rhs = nullptr;
        if (PJRT_Error* error = GetValueTensor(values, bitwise.lhs_id(), "bitwise_binary", &lhs)) {
          return error;
        }
        if (PJRT_Error* error = GetValueTensor(values, bitwise.rhs_id(), "bitwise_binary", &rhs)) {
          return error;
        }
        if (PJRT_Error* error = ExecuteBitwiseBinary(bitwise, *lhs, *rhs,
                                                     output_value_desc.tensor(),
                                                     mesh_device.get(),
                                                     &values[op.output_id()])) {
          return error;
        }
        break;
      }
      default:
        return Unimplemented("C++ execution does not support op kind: " +
                             OpKindName(op.kind_case()));
    }
  }

  for (int i = 0; i < program.output_ids_size(); ++i) {
    const uint32_t output_id = program.output_ids(i);
    if (output_id >= static_cast<uint32_t>(values.size()) || !values[output_id].has_value()) {
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

    const ttnn::Tensor* output_tensor = &*values[output_id];
    if (PJRT_Error* error = CreatePjrtBufferFromTtnnTensor(
            output_type, output_dims, target_device, target_device->default_memory, *output_tensor, &outputs[i])) {
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
  auto event = std::make_unique<PJRT_Event>();
  event->ready = false;
  event->error = std::nullopt;
  args->event = event.release();
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
  auto executable = std::make_unique<PJRT_LoadedExecutable>();
  executable->metadata = std::move(metadata);
  executable->addressable_devices = args->client->addressable_device_ptrs;
  executable->deleted = false;
  args->executable = executable.release();
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
  auto executable = std::make_unique<PJRT_Executable>();
  executable->metadata = std::move(metadata);
  args->executable = executable.release();
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
  auto executable = std::make_unique<PJRT_Executable>();
  executable->metadata = args->loaded_executable->metadata;
  args->executable = executable.release();
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
  if (PJRT_Error* error = ExecuteProgram(args->executable, arguments, args->num_args,
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
  auto attributes = std::make_unique<PJRT_Device_Attributes>();
  args->device_attributes = attributes.release();
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
  size_t size = 0;
  if (PJRT_Error* error = TtnnTensorPhysicalByteSize(*args->buffer, &size)) {
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
  args->buffer->Delete();
  return nullptr;
}

extern "C" PJRT_Error* TT_Buffer_IsDeleted(PJRT_Buffer_IsDeleted_Args* args) {
  if (args == nullptr || args->buffer == nullptr) {
    return InvalidArgument("buffer must not be null");
  }
  args->is_deleted = args->buffer->IsDeleted();
  return nullptr;
}

extern "C" PJRT_Error* TT_Buffer_ToHostBuffer(PJRT_Buffer_ToHostBuffer_Args* args) {
  if (args == nullptr || args->src == nullptr) {
    return InvalidArgument("src must not be null");
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
  args->event = args->buffer->IsDeleted()
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
  auto context = std::make_unique<PJRT_ExecuteContext>();
  args->context = context.release();
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
