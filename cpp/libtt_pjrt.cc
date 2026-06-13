#include "cpp/libtt_pjrt.h"

#include "cpp/pjrt_buffer.h"
#include "cpp/tt_metal_matmul_runtime.h"
#include "mlir/executable.pb.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
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

std::vector<int64_t> RepeatedI64ToVector(const google::protobuf::RepeatedField<int64_t>& field) {
  return std::vector<int64_t>(field.begin(), field.end());
}

PJRT_Error* CheckedElementCount(const std::vector<int64_t>& dims, size_t* out) {
  size_t count = 1;
  for (int64_t dim : dims) {
    if (dim < 0) {
      return InvalidArgument("shape dimensions must be >= 0");
    }
    const size_t value = static_cast<size_t>(dim);
    if (value != 0 && count > std::numeric_limits<size_t>::max() / value) {
      return ResourceExhausted("tensor element count overflows size_t");
    }
    count *= value;
  }
  *out = count;
  return nullptr;
}

float Bf16ToFloat(uint16_t value) {
  uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

uint16_t FloatToBf16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t rounding_bias = 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>((bits + rounding_bias) >> 16);
}

PJRT_Error* ReadElementAsDouble(PJRT_Buffer_Type type, const std::vector<std::byte>& data,
                                size_t index, double* out) {
  const size_t element_size = BytesPerElement(type);
  if (element_size == 0) {
    return Unimplemented("unsupported element type in fused elementwise op");
  }
  const size_t offset = index * element_size;
  if (offset > data.size() || element_size > data.size() - offset) {
    return InvalidArgument("fused elementwise input index is out of bounds");
  }
  switch (type) {
    case PJRT_Buffer_Type_PRED:
    case PJRT_Buffer_Type_U8: {
      uint8_t value = 0;
      std::memcpy(&value, data.data() + offset, sizeof(value));
      *out = value;
      return nullptr;
    }
    case PJRT_Buffer_Type_S8: {
      int8_t value = 0;
      std::memcpy(&value, data.data() + offset, sizeof(value));
      *out = value;
      return nullptr;
    }
    case PJRT_Buffer_Type_U16: {
      uint16_t value = 0;
      std::memcpy(&value, data.data() + offset, sizeof(value));
      *out = value;
      return nullptr;
    }
    case PJRT_Buffer_Type_BF16: {
      uint16_t value = 0;
      std::memcpy(&value, data.data() + offset, sizeof(value));
      *out = Bf16ToFloat(value);
      return nullptr;
    }
    case PJRT_Buffer_Type_S32: {
      int32_t value = 0;
      std::memcpy(&value, data.data() + offset, sizeof(value));
      *out = value;
      return nullptr;
    }
    case PJRT_Buffer_Type_U32: {
      uint32_t value = 0;
      std::memcpy(&value, data.data() + offset, sizeof(value));
      *out = value;
      return nullptr;
    }
    case PJRT_Buffer_Type_F32: {
      float value = 0.0f;
      std::memcpy(&value, data.data() + offset, sizeof(value));
      *out = value;
      return nullptr;
    }
    default:
      return Unimplemented("unsupported element type in fused elementwise op");
  }
}

PJRT_Error* WriteElementFromDouble(PJRT_Buffer_Type type, double value,
                                   std::vector<std::byte>* data, size_t index) {
  const size_t element_size = BytesPerElement(type);
  if (element_size == 0) {
    return Unimplemented("unsupported element type in fused elementwise op");
  }
  const size_t offset = index * element_size;
  if (offset > data->size() || element_size > data->size() - offset) {
    return InvalidArgument("fused elementwise output index is out of bounds");
  }
  switch (type) {
    case PJRT_Buffer_Type_PRED: {
      const uint8_t stored = static_cast<uint8_t>(value != 0.0);
      std::memcpy(data->data() + offset, &stored, sizeof(stored));
      return nullptr;
    }
    case PJRT_Buffer_Type_U8: {
      const uint8_t stored = static_cast<uint8_t>(value);
      std::memcpy(data->data() + offset, &stored, sizeof(stored));
      return nullptr;
    }
    case PJRT_Buffer_Type_S8: {
      const int8_t stored = static_cast<int8_t>(value);
      std::memcpy(data->data() + offset, &stored, sizeof(stored));
      return nullptr;
    }
    case PJRT_Buffer_Type_U16: {
      const uint16_t stored = static_cast<uint16_t>(value);
      std::memcpy(data->data() + offset, &stored, sizeof(stored));
      return nullptr;
    }
    case PJRT_Buffer_Type_BF16: {
      const uint16_t stored = FloatToBf16(static_cast<float>(value));
      std::memcpy(data->data() + offset, &stored, sizeof(stored));
      return nullptr;
    }
    case PJRT_Buffer_Type_S32: {
      const int32_t stored = static_cast<int32_t>(value);
      std::memcpy(data->data() + offset, &stored, sizeof(stored));
      return nullptr;
    }
    case PJRT_Buffer_Type_U32: {
      const uint32_t stored = static_cast<uint32_t>(value);
      std::memcpy(data->data() + offset, &stored, sizeof(stored));
      return nullptr;
    }
    case PJRT_Buffer_Type_F32: {
      const float stored = static_cast<float>(value);
      std::memcpy(data->data() + offset, &stored, sizeof(stored));
      return nullptr;
    }
    default:
      return Unimplemented("unsupported element type in fused elementwise op");
  }
}

PJRT_Error* PackedValueAsDouble(tt::TensorDesc::ElementType element_type,
                                uint32_t packed_value, double* out) {
  const PJRT_Buffer_Type type = PjrtBufferTypeFromProto(element_type);
  if (type == PJRT_Buffer_Type_INVALID) {
    return Unimplemented("unsupported fused elementwise constant element type");
  }
  const size_t element_size = BytesPerElement(type);
  std::vector<std::byte> data(element_size);
  if (element_size != 0) {
    std::memcpy(data.data(), &packed_value, element_size);
  }
  return ReadElementAsDouble(type, data, 0, out);
}

bool CompareValues(double lhs, double rhs,
                   tt::FusedElementwiseOp::Node::CompareDirection direction) {
  using Direction = tt::FusedElementwiseOp::Node::CompareDirection;
  switch (direction) {
    case Direction::FusedElementwiseOp_Node_CompareDirection_DIRECTION_EQ:
      return lhs == rhs;
    case Direction::FusedElementwiseOp_Node_CompareDirection_DIRECTION_NE:
      return lhs != rhs;
    case Direction::FusedElementwiseOp_Node_CompareDirection_DIRECTION_GE:
      return lhs >= rhs;
    case Direction::FusedElementwiseOp_Node_CompareDirection_DIRECTION_GT:
      return lhs > rhs;
    case Direction::FusedElementwiseOp_Node_CompareDirection_DIRECTION_LE:
      return lhs <= rhs;
    case Direction::FusedElementwiseOp_Node_CompareDirection_DIRECTION_LT:
      return lhs < rhs;
  }
  return false;
}

PJRT_Error* ValueTensorDesc(const tt::Executable& program, uint32_t value_id,
                            const tt::TensorDesc** out) {
  if (value_id >= static_cast<uint32_t>(program.values_size())) {
    return Internal("executable value metadata id is out of bounds");
  }
  const tt::ValueDesc& value_desc = program.values(value_id);
  if (!value_desc.has_tensor()) {
    return Internal("executable value is missing tensor metadata");
  }
  *out = &value_desc.tensor();
  return nullptr;
}

PJRT_Error* CreateBufferForValue(const tt::Executable& program, uint32_t value_id,
                                 PJRT_Device* target_device,
                                 std::vector<std::byte> data,
                                 PJRT_Buffer** out) {
  const tt::TensorDesc* tensor = nullptr;
  if (PJRT_Error* error = ValueTensorDesc(program, value_id, &tensor)) {
    return error;
  }
  PJRT_Buffer_Type type = PJRT_Buffer_Type_INVALID;
  if (PJRT_Error* error = TensorDescBufferType(*tensor, &type)) {
    return error;
  }
  std::vector<int64_t> dims;
  if (PJRT_Error* error = TensorDescDims(*tensor, &dims)) {
    return error;
  }
  size_t expected_size = 0;
  if (PJRT_Error* error = HostByteSize(type, dims, &expected_size)) {
    return error;
  }
  if (data.size() != expected_size) {
    return InvalidArgument("buffer byte size does not match executable value shape");
  }
  const void* bytes = data.empty() ? nullptr : data.data();
  PJRT_Memory* memory = target_device == nullptr ? nullptr : target_device->default_memory;
  return CreatePjrtBufferFromHostBytes(type, dims, target_device, memory, bytes, data.size(), out);
}

PJRT_Error* ConstantBytes(PJRT_Buffer_Type type, const std::vector<int64_t>& dims,
                          const tt::ConstantOp& constant,
                          std::vector<std::byte>* out) {
  size_t byte_size = 0;
  if (PJRT_Error* error = HostByteSize(type, dims, &byte_size)) {
    return error;
  }
  if (!constant.data().empty()) {
    if (constant.data().size() != byte_size) {
      return InvalidArgument("constant payload byte size does not match tensor shape");
    }
    out->resize(byte_size);
    if (byte_size != 0) {
      std::memcpy(out->data(), constant.data().data(), byte_size);
    }
    return nullptr;
  }

  const size_t element_size = BytesPerElement(type);
  if (element_size == 0) {
    return Unimplemented("constant uses unsupported element type");
  }
  out->assign(byte_size, std::byte{0});
  const uint32_t packed = constant.packed_value();
  for (size_t offset = 0; offset < byte_size; offset += element_size) {
    std::memcpy(out->data() + offset, &packed, element_size);
  }
  return nullptr;
}

PJRT_Error* ExecuteFusedElementwiseOp(const tt::Executable& program,
                                      const tt::FusedElementwiseOp& fused,
                                      uint32_t output_id,
                                      const std::vector<PJRT_Buffer*>& values,
                                      PJRT_Device* target_device,
                                      PJRT_Buffer** out) {
  const tt::TensorDesc* output_desc = nullptr;
  if (PJRT_Error* error = ValueTensorDesc(program, output_id, &output_desc)) {
    return error;
  }
  PJRT_Buffer_Type output_type = PJRT_Buffer_Type_INVALID;
  if (PJRT_Error* error = TensorDescBufferType(*output_desc, &output_type)) {
    return error;
  }
  std::vector<int64_t> output_dims;
  if (PJRT_Error* error = TensorDescDims(*output_desc, &output_dims)) {
    return error;
  }
  size_t element_count = 0;
  if (PJRT_Error* error = CheckedElementCount(output_dims, &element_count)) {
    return error;
  }

  struct InputData {
    PJRT_Buffer_Type type = PJRT_Buffer_Type_INVALID;
    size_t elements = 0;
    std::vector<std::byte> bytes;
  };
  std::vector<InputData> inputs;
  inputs.reserve(static_cast<size_t>(fused.input_ids_size()));
  for (uint32_t input_id : fused.input_ids()) {
    if (input_id >= values.size() || values[input_id] == nullptr) {
      return Internal("fused elementwise input value was not produced");
    }
    const tt::TensorDesc* input_desc = nullptr;
    if (PJRT_Error* error = ValueTensorDesc(program, input_id, &input_desc)) {
      return error;
    }
    InputData input;
    if (PJRT_Error* error = TensorDescBufferType(*input_desc, &input.type)) {
      return error;
    }
    std::vector<int64_t> input_dims;
    if (PJRT_Error* error = TensorDescDims(*input_desc, &input_dims)) {
      return error;
    }
    if (PJRT_Error* error = CheckedElementCount(input_dims, &input.elements)) {
      return error;
    }
    if (input.elements != element_count && input.elements != 1) {
      return Unimplemented("fused elementwise only supports same-shape or scalar inputs");
    }
    if (PJRT_Error* error = ReadBufferLogicalBytes(*values[input_id], &input.bytes)) {
      return error;
    }
    inputs.push_back(std::move(input));
  }

  std::vector<std::vector<double>> node_values;
  node_values.reserve(static_cast<size_t>(fused.nodes_size()));
  using Node = tt::FusedElementwiseOp::Node;
  for (const Node& node : fused.nodes()) {
    std::vector<double> result(element_count, 0.0);
    switch (node.kind()) {
      case Node::KIND_INPUT: {
        if (node.input_index() >= inputs.size()) {
          return Internal("fused elementwise input node index is out of bounds");
        }
        const InputData& input = inputs[node.input_index()];
        for (size_t i = 0; i < element_count; ++i) {
          const size_t source_index =
              (node.single_tile_broadcast() || input.elements == 1) ? 0 : i;
          if (PJRT_Error* error =
                  ReadElementAsDouble(input.type, input.bytes, source_index, &result[i])) {
            return error;
          }
        }
        break;
      }
      case Node::KIND_CONSTANT: {
        double value = 0.0;
        if (PJRT_Error* error =
                PackedValueAsDouble(node.element_type(), node.packed_value(), &value)) {
          return error;
        }
        std::fill(result.begin(), result.end(), value);
        break;
      }
      case Node::KIND_ADD:
      case Node::KIND_SUBTRACT:
      case Node::KIND_MULTIPLY:
      case Node::KIND_DIVIDE:
      case Node::KIND_MAX:
      case Node::KIND_POWER:
      case Node::KIND_COMPARE: {
        if (node.input_nodes_size() != 2) {
          return Internal("binary fused elementwise node must have two inputs");
        }
        const uint32_t lhs_index = node.input_nodes(0);
        const uint32_t rhs_index = node.input_nodes(1);
        if (lhs_index >= node_values.size() || rhs_index >= node_values.size()) {
          return Internal("fused elementwise node input is out of bounds");
        }
        const auto& lhs = node_values[lhs_index];
        const auto& rhs = node_values[rhs_index];
        for (size_t i = 0; i < element_count; ++i) {
          switch (node.kind()) {
            case Node::KIND_ADD:
              result[i] = lhs[i] + rhs[i];
              break;
            case Node::KIND_SUBTRACT:
              result[i] = lhs[i] - rhs[i];
              break;
            case Node::KIND_MULTIPLY:
              result[i] = lhs[i] * rhs[i];
              break;
            case Node::KIND_DIVIDE:
              result[i] = lhs[i] / rhs[i];
              break;
            case Node::KIND_MAX:
              result[i] = std::max(lhs[i], rhs[i]);
              break;
            case Node::KIND_POWER:
              result[i] = std::pow(lhs[i], rhs[i]);
              break;
            case Node::KIND_COMPARE:
              result[i] = CompareValues(lhs[i], rhs[i], node.compare_direction()) ? 1.0 : 0.0;
              break;
            default:
              break;
          }
        }
        break;
      }
      case Node::KIND_NEGATE:
      case Node::KIND_EXPONENTIAL:
      case Node::KIND_RSQRT:
      case Node::KIND_COSINE:
      case Node::KIND_SINE:
      case Node::KIND_CONVERT:
      case Node::KIND_LOG: {
        if (node.input_nodes_size() != 1) {
          return Internal("unary fused elementwise node must have one input");
        }
        const uint32_t input_index = node.input_nodes(0);
        if (input_index >= node_values.size()) {
          return Internal("fused elementwise node input is out of bounds");
        }
        const auto& input = node_values[input_index];
        for (size_t i = 0; i < element_count; ++i) {
          switch (node.kind()) {
            case Node::KIND_NEGATE:
              result[i] = -input[i];
              break;
            case Node::KIND_EXPONENTIAL:
              result[i] = std::exp(input[i]);
              break;
            case Node::KIND_RSQRT:
              result[i] = 1.0 / std::sqrt(input[i]);
              break;
            case Node::KIND_COSINE:
              result[i] = std::cos(input[i]);
              break;
            case Node::KIND_SINE:
              result[i] = std::sin(input[i]);
              break;
            case Node::KIND_CONVERT:
              result[i] = input[i];
              break;
            case Node::KIND_LOG:
              result[i] = std::log(input[i]);
              break;
            default:
              break;
          }
        }
        break;
      }
      default:
        return Unimplemented("unsupported fused elementwise node kind");
    }
    node_values.push_back(std::move(result));
  }

  if (node_values.empty()) {
    return Internal("fused elementwise op has no nodes");
  }
  size_t output_size = 0;
  if (PJRT_Error* error = HostByteSize(output_type, output_dims, &output_size)) {
    return error;
  }
  std::vector<std::byte> output_data(output_size);
  const auto& result = node_values.back();
  for (size_t i = 0; i < element_count; ++i) {
    if (PJRT_Error* error = WriteElementFromDouble(output_type, result[i], &output_data, i)) {
      return error;
    }
  }
  return CreateBufferForValue(program, output_id, target_device, std::move(output_data), out);
}

PJRT_Error* ExecuteSelectOp(const tt::Executable& program, const tt::SelectOp& select,
                            uint32_t output_id, const std::vector<PJRT_Buffer*>& values,
                            PJRT_Device* target_device, PJRT_Buffer** out) {
  const tt::TensorDesc* output_desc = nullptr;
  if (PJRT_Error* error = ValueTensorDesc(program, output_id, &output_desc)) {
    return error;
  }
  PJRT_Buffer_Type output_type = PJRT_Buffer_Type_INVALID;
  if (PJRT_Error* error = TensorDescBufferType(*output_desc, &output_type)) {
    return error;
  }
  std::vector<int64_t> output_dims;
  if (PJRT_Error* error = TensorDescDims(*output_desc, &output_dims)) {
    return error;
  }
  size_t element_count = 0;
  if (PJRT_Error* error = CheckedElementCount(output_dims, &element_count)) {
    return error;
  }
  const size_t element_size = BytesPerElement(output_type);
  if (element_size == 0) {
    return Unimplemented("select uses unsupported output element type");
  }

  auto require_value = [&](uint32_t id, const char* name) -> PJRT_Error* {
    if (id >= values.size() || values[id] == nullptr) {
      return Internal(std::string("select ") + name + " value was not produced");
    }
    return nullptr;
  };
  if (PJRT_Error* error = require_value(select.pred_id(), "predicate")) {
    return error;
  }
  if (PJRT_Error* error = require_value(select.on_true_id(), "true")) {
    return error;
  }
  if (PJRT_Error* error = require_value(select.on_false_id(), "false")) {
    return error;
  }

  const auto* pred_buffer = values[select.pred_id()];
  const auto* true_buffer = values[select.on_true_id()];
  const auto* false_buffer = values[select.on_false_id()];
  if (pred_buffer->buffer_type != PJRT_Buffer_Type_PRED) {
    return InvalidArgument("select predicate must have PRED element type");
  }
  if (true_buffer->buffer_type != output_type || false_buffer->buffer_type != output_type) {
    return InvalidArgument("select data operand types must match output type");
  }

  size_t pred_elements = 0;
  size_t true_elements = 0;
  size_t false_elements = 0;
  if (PJRT_Error* error = CheckedElementCount(pred_buffer->dims, &pred_elements)) {
    return error;
  }
  if (PJRT_Error* error = CheckedElementCount(true_buffer->dims, &true_elements)) {
    return error;
  }
  if (PJRT_Error* error = CheckedElementCount(false_buffer->dims, &false_elements)) {
    return error;
  }
  if ((pred_elements != element_count && pred_elements != 1) ||
      (true_elements != element_count && true_elements != 1) ||
      (false_elements != element_count && false_elements != 1)) {
    return Unimplemented("select only supports same-shape or scalar operands");
  }

  std::vector<std::byte> pred_data;
  std::vector<std::byte> true_data;
  std::vector<std::byte> false_data;
  if (PJRT_Error* error = ReadBufferLogicalBytes(*pred_buffer, &pred_data)) {
    return error;
  }
  if (PJRT_Error* error = ReadBufferLogicalBytes(*true_buffer, &true_data)) {
    return error;
  }
  if (PJRT_Error* error = ReadBufferLogicalBytes(*false_buffer, &false_data)) {
    return error;
  }

  size_t output_size = 0;
  if (PJRT_Error* error = HostByteSize(output_type, output_dims, &output_size)) {
    return error;
  }
  std::vector<std::byte> output_data(output_size);
  for (size_t i = 0; i < element_count; ++i) {
    double pred = 0.0;
    if (PJRT_Error* error =
            ReadElementAsDouble(PJRT_Buffer_Type_PRED, pred_data,
                                pred_elements == 1 ? 0 : i, &pred)) {
      return error;
    }
    const std::vector<std::byte>& source = pred != 0.0 ? true_data : false_data;
    const size_t source_elements = pred != 0.0 ? true_elements : false_elements;
    const size_t source_index = source_elements == 1 ? 0 : i;
    std::memcpy(output_data.data() + i * element_size,
                source.data() + source_index * element_size, element_size);
  }
  return CreateBufferForValue(program, output_id, target_device, std::move(output_data), out);
}

std::vector<size_t> RowMajorStrides(const std::vector<int64_t>& dims) {
  std::vector<size_t> strides(dims.size(), 1);
  size_t stride = 1;
  for (size_t i = dims.size(); i > 0; --i) {
    strides[i - 1] = stride;
    stride *= static_cast<size_t>(dims[i - 1]);
  }
  return strides;
}

std::vector<size_t> RowMajorCoords(size_t linear_index,
                                   const std::vector<int64_t>& dims,
                                   const std::vector<size_t>& strides) {
  std::vector<size_t> coords(dims.size(), 0);
  for (size_t dim = 0; dim < dims.size(); ++dim) {
    coords[dim] = strides.empty() ? 0 : (linear_index / strides[dim]) %
                                       static_cast<size_t>(dims[dim]);
  }
  return coords;
}

PJRT_Error* ExecuteReshapeOp(const tt::Executable& program,
                             const tt::ReshapeOp& reshape,
                             uint32_t output_id,
                             const std::vector<PJRT_Buffer*>& values,
                             PJRT_Device* target_device,
                             PJRT_Buffer** out) {
  if (reshape.operand_id() >= values.size() || values[reshape.operand_id()] == nullptr) {
    return Internal("reshape input value was not produced");
  }
  const PJRT_Buffer* input = values[reshape.operand_id()];
  const tt::TensorDesc* output_desc = nullptr;
  if (PJRT_Error* error = ValueTensorDesc(program, output_id, &output_desc)) {
    return error;
  }
  PJRT_Buffer_Type output_type = PJRT_Buffer_Type_INVALID;
  if (PJRT_Error* error = TensorDescBufferType(*output_desc, &output_type)) {
    return error;
  }
  if (input->buffer_type != output_type) {
    return InvalidArgument("reshape input and output element types must match");
  }
  std::vector<int64_t> output_dims;
  if (PJRT_Error* error = TensorDescDims(*output_desc, &output_dims)) {
    return error;
  }
  size_t input_elements = 0;
  size_t output_elements = 0;
  if (PJRT_Error* error = CheckedElementCount(input->dims, &input_elements)) {
    return error;
  }
  if (PJRT_Error* error = CheckedElementCount(output_dims, &output_elements)) {
    return error;
  }
  if (input_elements != output_elements) {
    return InvalidArgument("reshape input and output element counts must match");
  }

  std::vector<std::byte> data;
  if (PJRT_Error* error = ReadBufferLogicalBytes(*input, &data)) {
    return error;
  }
  return CreateBufferForValue(program, output_id, target_device, std::move(data), out);
}

PJRT_Error* ExecuteSliceOp(const tt::Executable& program, const tt::SliceOp& slice,
                           uint32_t output_id, const std::vector<PJRT_Buffer*>& values,
                           PJRT_Device* target_device, PJRT_Buffer** out) {
  if (slice.operand_id() >= values.size() || values[slice.operand_id()] == nullptr) {
    return Internal("slice input value was not produced");
  }
  const PJRT_Buffer* input = values[slice.operand_id()];
  const size_t rank = input->dims.size();
  if (slice.start_indices_size() != static_cast<int>(rank) ||
      slice.limit_indices_size() != static_cast<int>(rank) ||
      slice.strides_size() != static_cast<int>(rank)) {
    return InvalidArgument("slice index metadata rank must match operand rank");
  }

  const tt::TensorDesc* output_desc = nullptr;
  if (PJRT_Error* error = ValueTensorDesc(program, output_id, &output_desc)) {
    return error;
  }
  PJRT_Buffer_Type output_type = PJRT_Buffer_Type_INVALID;
  if (PJRT_Error* error = TensorDescBufferType(*output_desc, &output_type)) {
    return error;
  }
  if (input->buffer_type != output_type) {
    return InvalidArgument("slice input and output element types must match");
  }
  std::vector<int64_t> output_dims;
  if (PJRT_Error* error = TensorDescDims(*output_desc, &output_dims)) {
    return error;
  }
  if (output_dims.size() != rank) {
    return InvalidArgument("slice output rank must match operand rank");
  }

  for (size_t dim = 0; dim < rank; ++dim) {
    const int64_t start = slice.start_indices(static_cast<int>(dim));
    const int64_t limit = slice.limit_indices(static_cast<int>(dim));
    const int64_t stride = slice.strides(static_cast<int>(dim));
    if (start < 0 || limit < start || limit > input->dims[dim] || stride <= 0) {
      return InvalidArgument("slice indices are out of bounds");
    }
    const int64_t expected = (limit - start + stride - 1) / stride;
    if (output_dims[dim] != expected) {
      return InvalidArgument("slice output shape does not match slice metadata");
    }
  }

  const size_t element_size = BytesPerElement(output_type);
  if (element_size == 0) {
    return Unimplemented("slice uses unsupported element type");
  }
  size_t output_elements = 0;
  if (PJRT_Error* error = CheckedElementCount(output_dims, &output_elements)) {
    return error;
  }
  size_t output_size = 0;
  if (PJRT_Error* error = HostByteSize(output_type, output_dims, &output_size)) {
    return error;
  }

  std::vector<std::byte> input_data;
  if (PJRT_Error* error = ReadBufferLogicalBytes(*input, &input_data)) {
    return error;
  }
  std::vector<std::byte> output_data(output_size);
  const std::vector<size_t> input_strides = RowMajorStrides(input->dims);
  const std::vector<size_t> output_strides = RowMajorStrides(output_dims);
  for (size_t output_index = 0; output_index < output_elements; ++output_index) {
    const std::vector<size_t> coords =
        RowMajorCoords(output_index, output_dims, output_strides);
    size_t input_index = 0;
    for (size_t dim = 0; dim < rank; ++dim) {
      const size_t input_coord =
          static_cast<size_t>(slice.start_indices(static_cast<int>(dim))) +
          coords[dim] * static_cast<size_t>(slice.strides(static_cast<int>(dim)));
      input_index += input_coord * input_strides[dim];
    }
    std::memcpy(output_data.data() + output_index * element_size,
                input_data.data() + input_index * element_size, element_size);
  }
  return CreateBufferForValue(program, output_id, target_device, std::move(output_data), out);
}

PJRT_Error* ExecuteTransposeOp(const tt::Executable& program,
                               const tt::TransposeOp& transpose,
                               uint32_t output_id,
                               const std::vector<PJRT_Buffer*>& values,
                               PJRT_Device* target_device,
                               PJRT_Buffer** out) {
  if (transpose.operand_id() >= values.size() || values[transpose.operand_id()] == nullptr) {
    return Internal("transpose input value was not produced");
  }
  const PJRT_Buffer* input = values[transpose.operand_id()];
  const size_t rank = input->dims.size();
  if (transpose.permutation_size() != static_cast<int>(rank)) {
    return InvalidArgument("transpose permutation rank must match operand rank");
  }
  std::vector<bool> seen(rank, false);
  for (int64_t dim : transpose.permutation()) {
    if (dim < 0 || dim >= static_cast<int64_t>(rank) || seen[static_cast<size_t>(dim)]) {
      return InvalidArgument("transpose permutation must be a rank permutation");
    }
    seen[static_cast<size_t>(dim)] = true;
  }

  const tt::TensorDesc* output_desc = nullptr;
  if (PJRT_Error* error = ValueTensorDesc(program, output_id, &output_desc)) {
    return error;
  }
  PJRT_Buffer_Type output_type = PJRT_Buffer_Type_INVALID;
  if (PJRT_Error* error = TensorDescBufferType(*output_desc, &output_type)) {
    return error;
  }
  if (input->buffer_type != output_type) {
    return InvalidArgument("transpose input and output element types must match");
  }
  std::vector<int64_t> output_dims;
  if (PJRT_Error* error = TensorDescDims(*output_desc, &output_dims)) {
    return error;
  }
  if (output_dims.size() != rank) {
    return InvalidArgument("transpose output rank must match operand rank");
  }
  for (size_t dim = 0; dim < rank; ++dim) {
    if (output_dims[dim] != input->dims[static_cast<size_t>(transpose.permutation(static_cast<int>(dim)))]) {
      return InvalidArgument("transpose output shape does not match permutation");
    }
  }

  const size_t element_size = BytesPerElement(output_type);
  if (element_size == 0) {
    return Unimplemented("transpose uses unsupported element type");
  }
  size_t output_elements = 0;
  if (PJRT_Error* error = CheckedElementCount(output_dims, &output_elements)) {
    return error;
  }
  size_t output_size = 0;
  if (PJRT_Error* error = HostByteSize(output_type, output_dims, &output_size)) {
    return error;
  }

  std::vector<std::byte> input_data;
  if (PJRT_Error* error = ReadBufferLogicalBytes(*input, &input_data)) {
    return error;
  }
  std::vector<std::byte> output_data(output_size);
  const std::vector<size_t> input_strides = RowMajorStrides(input->dims);
  const std::vector<size_t> output_strides = RowMajorStrides(output_dims);
  for (size_t output_index = 0; output_index < output_elements; ++output_index) {
    const std::vector<size_t> output_coords =
        RowMajorCoords(output_index, output_dims, output_strides);
    size_t input_index = 0;
    for (size_t output_dim = 0; output_dim < rank; ++output_dim) {
      const size_t input_dim =
          static_cast<size_t>(transpose.permutation(static_cast<int>(output_dim)));
      input_index += output_coords[output_dim] * input_strides[input_dim];
    }
    std::memcpy(output_data.data() + output_index * element_size,
                input_data.data() + input_index * element_size, element_size);
  }
  return CreateBufferForValue(program, output_id, target_device, std::move(output_data), out);
}

PJRT_Error* ExecuteConcatenateOp(const tt::Executable& program,
                                 const tt::ConcatenateOp& concatenate,
                                 uint32_t output_id,
                                 const std::vector<PJRT_Buffer*>& values,
                                 PJRT_Device* target_device,
                                 PJRT_Buffer** out) {
  const tt::TensorDesc* output_desc = nullptr;
  if (PJRT_Error* error = ValueTensorDesc(program, output_id, &output_desc)) {
    return error;
  }
  PJRT_Buffer_Type output_type = PJRT_Buffer_Type_INVALID;
  if (PJRT_Error* error = TensorDescBufferType(*output_desc, &output_type)) {
    return error;
  }
  std::vector<int64_t> output_dims;
  if (PJRT_Error* error = TensorDescDims(*output_desc, &output_dims)) {
    return error;
  }
  const size_t rank = output_dims.size();
  if (concatenate.dimension() >= rank) {
    return InvalidArgument("concatenate dimension is out of bounds");
  }
  const size_t concat_dim = static_cast<size_t>(concatenate.dimension());

  struct ConcatInput {
    const PJRT_Buffer* buffer = nullptr;
    std::vector<std::byte> data;
    std::vector<size_t> strides;
    size_t offset = 0;
  };
  std::vector<ConcatInput> inputs;
  inputs.reserve(static_cast<size_t>(concatenate.input_ids_size()));
  int64_t concat_extent = 0;
  for (uint32_t input_id : concatenate.input_ids()) {
    if (input_id >= values.size() || values[input_id] == nullptr) {
      return Internal("concatenate input value was not produced");
    }
    const PJRT_Buffer* input = values[input_id];
    if (input->buffer_type != output_type || input->dims.size() != rank) {
      return InvalidArgument("concatenate inputs must match output rank and element type");
    }
    for (size_t dim = 0; dim < rank; ++dim) {
      if (dim != concat_dim && input->dims[dim] != output_dims[dim]) {
        return InvalidArgument("concatenate non-concatenated dimensions must match");
      }
    }
    ConcatInput input_data;
    input_data.buffer = input;
    input_data.offset = static_cast<size_t>(concat_extent);
    input_data.strides = RowMajorStrides(input->dims);
    if (PJRT_Error* error = ReadBufferLogicalBytes(*input, &input_data.data)) {
      return error;
    }
    concat_extent += input->dims[concat_dim];
    inputs.push_back(std::move(input_data));
  }
  if (concat_extent != output_dims[concat_dim]) {
    return InvalidArgument("concatenate input extents do not match output shape");
  }

  const size_t element_size = BytesPerElement(output_type);
  if (element_size == 0) {
    return Unimplemented("concatenate uses unsupported element type");
  }
  size_t output_elements = 0;
  if (PJRT_Error* error = CheckedElementCount(output_dims, &output_elements)) {
    return error;
  }
  size_t output_size = 0;
  if (PJRT_Error* error = HostByteSize(output_type, output_dims, &output_size)) {
    return error;
  }
  std::vector<std::byte> output_data(output_size);
  const std::vector<size_t> output_strides = RowMajorStrides(output_dims);
  for (size_t output_index = 0; output_index < output_elements; ++output_index) {
    std::vector<size_t> coords = RowMajorCoords(output_index, output_dims, output_strides);
    ConcatInput const* selected = nullptr;
    for (const ConcatInput& input : inputs) {
      const size_t begin = input.offset;
      const size_t end = begin + static_cast<size_t>(input.buffer->dims[concat_dim]);
      if (coords[concat_dim] >= begin && coords[concat_dim] < end) {
        selected = &input;
        coords[concat_dim] -= begin;
        break;
      }
    }
    if (selected == nullptr) {
      return Internal("concatenate failed to map output coordinate to input");
    }
    size_t input_index = 0;
    for (size_t dim = 0; dim < rank; ++dim) {
      input_index += coords[dim] * selected->strides[dim];
    }
    std::memcpy(output_data.data() + output_index * element_size,
                selected->data.data() + input_index * element_size, element_size);
  }
  return CreateBufferForValue(program, output_id, target_device, std::move(output_data), out);
}

PJRT_Error* ExecuteIotaOp(const tt::Executable& program, const tt::IotaOp& iota,
                          uint32_t output_id, PJRT_Device* target_device,
                          PJRT_Buffer** out) {
  const tt::TensorDesc* output_desc = nullptr;
  if (PJRT_Error* error = ValueTensorDesc(program, output_id, &output_desc)) {
    return error;
  }
  PJRT_Buffer_Type output_type = PJRT_Buffer_Type_INVALID;
  if (PJRT_Error* error = TensorDescBufferType(*output_desc, &output_type)) {
    return error;
  }
  std::vector<int64_t> output_dims;
  if (PJRT_Error* error = TensorDescDims(*output_desc, &output_dims)) {
    return error;
  }
  if (iota.iota_dimension() >= output_dims.size()) {
    return InvalidArgument("iota dimension is out of bounds");
  }
  size_t output_elements = 0;
  if (PJRT_Error* error = CheckedElementCount(output_dims, &output_elements)) {
    return error;
  }
  size_t output_size = 0;
  if (PJRT_Error* error = HostByteSize(output_type, output_dims, &output_size)) {
    return error;
  }
  std::vector<std::byte> output_data(output_size);
  const std::vector<size_t> output_strides = RowMajorStrides(output_dims);
  for (size_t output_index = 0; output_index < output_elements; ++output_index) {
    const std::vector<size_t> coords =
        RowMajorCoords(output_index, output_dims, output_strides);
    if (PJRT_Error* error = WriteElementFromDouble(
            output_type, static_cast<double>(coords[static_cast<size_t>(iota.iota_dimension())]),
            &output_data, output_index)) {
      return error;
    }
  }
  return CreateBufferForValue(program, output_id, target_device, std::move(output_data), out);
}

double ApplyReducer(double accumulator, double value, tt::ReduceOp::Reducer reducer) {
  switch (reducer) {
    case tt::ReduceOp::REDUCER_ADD:
      return accumulator + value;
    case tt::ReduceOp::REDUCER_MAX:
      return std::max(accumulator, value);
    case tt::ReduceOp::REDUCER_MUL:
      return accumulator * value;
    case tt::ReduceOp::REDUCER_MIN:
      return std::min(accumulator, value);
    case tt::ReduceOp::REDUCER_AND:
      return (accumulator != 0.0 && value != 0.0) ? 1.0 : 0.0;
    case tt::ReduceOp::REDUCER_OR:
      return (accumulator != 0.0 || value != 0.0) ? 1.0 : 0.0;
  }
  return value;
}

PJRT_Error* ExecuteReduceOp(const tt::Executable& program, const tt::ReduceOp& reduce,
                            uint32_t output_id, const std::vector<PJRT_Buffer*>& values,
                            PJRT_Device* target_device, PJRT_Buffer** out) {
  if (reduce.input_ids_size() != 1 || reduce.init_value_ids_size() != 1) {
    return Unimplemented("reduce currently supports one input and one init value");
  }
  const uint32_t input_id = reduce.input_ids(0);
  const uint32_t init_id = reduce.init_value_ids(0);
  if (input_id >= values.size() || values[input_id] == nullptr ||
      init_id >= values.size() || values[init_id] == nullptr) {
    return Internal("reduce input or init value was not produced");
  }

  const PJRT_Buffer* input = values[input_id];
  const PJRT_Buffer* init = values[init_id];
  const tt::TensorDesc* output_desc = nullptr;
  if (PJRT_Error* error = ValueTensorDesc(program, output_id, &output_desc)) {
    return error;
  }
  PJRT_Buffer_Type output_type = PJRT_Buffer_Type_INVALID;
  if (PJRT_Error* error = TensorDescBufferType(*output_desc, &output_type)) {
    return error;
  }
  if (input->buffer_type != output_type || init->buffer_type != output_type) {
    return InvalidArgument("reduce input, init, and output element types must match");
  }
  std::vector<int64_t> output_dims;
  if (PJRT_Error* error = TensorDescDims(*output_desc, &output_dims)) {
    return error;
  }

  const size_t input_rank = input->dims.size();
  std::vector<bool> is_reduced(input_rank, false);
  for (int64_t dim : reduce.dimensions()) {
    if (dim < 0 || dim >= static_cast<int64_t>(input_rank) ||
        is_reduced[static_cast<size_t>(dim)]) {
      return InvalidArgument("reduce dimensions must be unique and in bounds");
    }
    is_reduced[static_cast<size_t>(dim)] = true;
  }
  std::vector<int64_t> expected_output_dims;
  expected_output_dims.reserve(input_rank);
  for (size_t dim = 0; dim < input_rank; ++dim) {
    if (!is_reduced[dim]) {
      expected_output_dims.push_back(input->dims[dim]);
    }
  }
  if (expected_output_dims != output_dims) {
    return InvalidArgument("reduce output shape does not match reduced input shape");
  }

  size_t input_elements = 0;
  size_t output_elements = 0;
  size_t init_elements = 0;
  if (PJRT_Error* error = CheckedElementCount(input->dims, &input_elements)) {
    return error;
  }
  if (PJRT_Error* error = CheckedElementCount(output_dims, &output_elements)) {
    return error;
  }
  if (PJRT_Error* error = CheckedElementCount(init->dims, &init_elements)) {
    return error;
  }
  if (init_elements != 1 && init_elements != output_elements) {
    return Unimplemented("reduce init value must be scalar or match the output shape");
  }

  std::vector<std::byte> input_data;
  std::vector<std::byte> init_data;
  if (PJRT_Error* error = ReadBufferLogicalBytes(*input, &input_data)) {
    return error;
  }
  if (PJRT_Error* error = ReadBufferLogicalBytes(*init, &init_data)) {
    return error;
  }

  std::vector<double> accumulators(output_elements == 0 ? 1 : output_elements, 0.0);
  for (size_t i = 0; i < accumulators.size(); ++i) {
    if (PJRT_Error* error = ReadElementAsDouble(
            init->buffer_type, init_data, init_elements == 1 ? 0 : i, &accumulators[i])) {
      return error;
    }
  }

  const std::vector<size_t> input_strides = RowMajorStrides(input->dims);
  const std::vector<size_t> output_strides = RowMajorStrides(output_dims);
  for (size_t input_index = 0; input_index < input_elements; ++input_index) {
    const std::vector<size_t> input_coords =
        RowMajorCoords(input_index, input->dims, input_strides);
    size_t output_index = 0;
    size_t output_dim = 0;
    for (size_t dim = 0; dim < input_rank; ++dim) {
      if (is_reduced[dim]) {
        continue;
      }
      output_index += input_coords[dim] * output_strides[output_dim];
      ++output_dim;
    }
    double value = 0.0;
    if (PJRT_Error* error =
            ReadElementAsDouble(input->buffer_type, input_data, input_index, &value)) {
      return error;
    }
    accumulators[output_index] =
        ApplyReducer(accumulators[output_index], value, reduce.reducer());
  }

  size_t output_size = 0;
  if (PJRT_Error* error = HostByteSize(output_type, output_dims, &output_size)) {
    return error;
  }
  std::vector<std::byte> output_data(output_size);
  for (size_t i = 0; i < output_elements; ++i) {
    if (PJRT_Error* error =
            WriteElementFromDouble(output_type, accumulators[i], &output_data, i)) {
      return error;
    }
  }
  return CreateBufferForValue(program, output_id, target_device, std::move(output_data), out);
}

PJRT_Error* ExecuteBroadcastInDimOp(const tt::Executable& program,
                                    const tt::BroadcastInDimOp& broadcast,
                                    uint32_t output_id,
                                    const std::vector<PJRT_Buffer*>& values,
                                    PJRT_Device* target_device,
                                    PJRT_Buffer** out) {
  if (broadcast.operand_id() >= values.size() || values[broadcast.operand_id()] == nullptr) {
    return Internal("broadcast input value was not produced");
  }
  const tt::TensorDesc* output_desc = nullptr;
  if (PJRT_Error* error = ValueTensorDesc(program, output_id, &output_desc)) {
    return error;
  }
  PJRT_Buffer_Type output_type = PJRT_Buffer_Type_INVALID;
  if (PJRT_Error* error = TensorDescBufferType(*output_desc, &output_type)) {
    return error;
  }
  std::vector<int64_t> output_dims;
  if (PJRT_Error* error = TensorDescDims(*output_desc, &output_dims)) {
    return error;
  }
  size_t output_elements = 0;
  if (PJRT_Error* error = CheckedElementCount(output_dims, &output_elements)) {
    return error;
  }

  const PJRT_Buffer* input = values[broadcast.operand_id()];
  if (input->buffer_type != output_type) {
    return InvalidArgument("broadcast input and output element types must match");
  }
  if (broadcast.broadcast_dimensions_size() != static_cast<int>(input->dims.size())) {
    return InvalidArgument("broadcast_dimensions rank must match operand rank");
  }
  for (int64_t dim : broadcast.broadcast_dimensions()) {
    if (dim < 0 || dim >= static_cast<int64_t>(output_dims.size())) {
      return InvalidArgument("broadcast dimension is out of output rank bounds");
    }
  }

  std::vector<std::byte> input_data;
  if (PJRT_Error* error = ReadBufferLogicalBytes(*input, &input_data)) {
    return error;
  }
  size_t output_size = 0;
  if (PJRT_Error* error = HostByteSize(output_type, output_dims, &output_size)) {
    return error;
  }
  const size_t element_size = BytesPerElement(output_type);
  std::vector<std::byte> output_data(output_size);
  const std::vector<size_t> input_strides = RowMajorStrides(input->dims);
  const std::vector<size_t> output_strides = RowMajorStrides(output_dims);

  for (size_t output_index = 0; output_index < output_elements; ++output_index) {
    size_t input_index = 0;
    for (size_t input_dim = 0; input_dim < input->dims.size(); ++input_dim) {
      const size_t output_dim =
          static_cast<size_t>(broadcast.broadcast_dimensions(static_cast<int>(input_dim)));
      const size_t output_coord =
          output_strides.empty() ? 0 : (output_index / output_strides[output_dim]) %
                                      static_cast<size_t>(output_dims[output_dim]);
      const size_t input_coord = input->dims[input_dim] == 1 ? 0 : output_coord;
      input_index += input_coord * input_strides[input_dim];
    }
    std::memcpy(output_data.data() + output_index * element_size,
                input_data.data() + input_index * element_size, element_size);
  }
  return CreateBufferForValue(program, output_id, target_device, std::move(output_data), out);
}

bool IsEmbeddingGather(const tt::GatherOp& gather, const std::vector<int64_t>& operand_dims,
                       const std::vector<int64_t>& indices_dims,
                       const std::vector<int64_t>& output_dims) {
  return operand_dims.size() == 2 && indices_dims.size() == 2 && output_dims.size() == 2 &&
         gather.offset_dims_size() == 1 && gather.offset_dims(0) == 1 &&
         gather.collapsed_slice_dims_size() == 1 && gather.collapsed_slice_dims(0) == 0 &&
         gather.operand_batching_dims_size() == 0 &&
         gather.start_indices_batching_dims_size() == 0 &&
         gather.start_index_map_size() == 1 && gather.start_index_map(0) == 0 &&
         gather.index_vector_dim() == 1 &&
         gather.slice_sizes_size() == 2 &&
         gather.slice_sizes(0) == 1 &&
         gather.slice_sizes(1) == operand_dims[1] &&
         indices_dims[1] == 1 &&
         output_dims[0] == indices_dims[0] &&
         output_dims[1] == operand_dims[1];
}

PJRT_Error* ExecuteGatherOp(const tt::Executable& program, const tt::GatherOp& gather,
                            uint32_t output_id, const std::vector<PJRT_Buffer*>& values,
                            PJRT_Device* target_device, PJRT_Buffer** out) {
  if (gather.operand_id() >= values.size() || values[gather.operand_id()] == nullptr ||
      gather.start_indices_id() >= values.size() || values[gather.start_indices_id()] == nullptr) {
    return Internal("gather input value was not produced");
  }
  const tt::TensorDesc* operand_desc = nullptr;
  const tt::TensorDesc* indices_desc = nullptr;
  const tt::TensorDesc* output_desc = nullptr;
  if (PJRT_Error* error = ValueTensorDesc(program, gather.operand_id(), &operand_desc)) {
    return error;
  }
  if (PJRT_Error* error = ValueTensorDesc(program, gather.start_indices_id(), &indices_desc)) {
    return error;
  }
  if (PJRT_Error* error = ValueTensorDesc(program, output_id, &output_desc)) {
    return error;
  }

  PJRT_Buffer_Type operand_type = PJRT_Buffer_Type_INVALID;
  PJRT_Buffer_Type indices_type = PJRT_Buffer_Type_INVALID;
  PJRT_Buffer_Type output_type = PJRT_Buffer_Type_INVALID;
  if (PJRT_Error* error = TensorDescBufferType(*operand_desc, &operand_type)) {
    return error;
  }
  if (PJRT_Error* error = TensorDescBufferType(*indices_desc, &indices_type)) {
    return error;
  }
  if (PJRT_Error* error = TensorDescBufferType(*output_desc, &output_type)) {
    return error;
  }
  std::vector<int64_t> operand_dims;
  std::vector<int64_t> indices_dims;
  std::vector<int64_t> output_dims;
  if (PJRT_Error* error = TensorDescDims(*operand_desc, &operand_dims)) {
    return error;
  }
  if (PJRT_Error* error = TensorDescDims(*indices_desc, &indices_dims)) {
    return error;
  }
  if (PJRT_Error* error = TensorDescDims(*output_desc, &output_dims)) {
    return error;
  }
  if (!IsEmbeddingGather(gather, operand_dims, indices_dims, output_dims)) {
    return Unimplemented("only embedding-style gather is currently supported");
  }

  TtMetalEmbeddingRequest request;
  request.local_hardware_id = target_device == nullptr ? 0 : target_device->local_hardware_id;
  request.indices.type = indices_type;
  request.indices.dims = {1, 1, 1, indices_dims[0]};
  request.table.type = operand_type;
  request.table.dims = {1, 1, operand_dims[0], operand_dims[1]};
  request.output_type = output_type;
  request.output_dims = output_dims;
  if (PJRT_Error* error = ReadBufferLogicalBytes(*values[gather.start_indices_id()],
                                                 &request.indices.data)) {
    return error;
  }
  if (PJRT_Error* error = ReadBufferLogicalBytes(*values[gather.operand_id()],
                                                 &request.table.data)) {
    return error;
  }

  std::vector<std::byte> result_data;
  if (PJRT_Error* error = ExecuteTtMetalEmbedding(request, &result_data)) {
    return error;
  }
  return CreateBufferForValue(program, output_id, target_device, std::move(result_data), out);
}

PJRT_Error* ExecuteTopKOp(const tt::Executable& program, const tt::TopKOp& top_k,
                          uint32_t values_output_id,
                          const std::vector<PJRT_Buffer*>& values,
                          PJRT_Device* target_device,
                          PJRT_Buffer** values_out,
                          PJRT_Buffer** indices_out) {
  if (top_k.operand_id() >= values.size() || values[top_k.operand_id()] == nullptr) {
    return Internal("top_k input value was not produced");
  }
  if (top_k.indices_id() >= static_cast<uint32_t>(program.values_size())) {
    return Internal("top_k indices metadata id is out of bounds");
  }

  const tt::TensorDesc* input_desc = nullptr;
  const tt::TensorDesc* values_output_desc = nullptr;
  const tt::TensorDesc* indices_output_desc = nullptr;
  if (PJRT_Error* error = ValueTensorDesc(program, top_k.operand_id(), &input_desc)) {
    return error;
  }
  if (PJRT_Error* error = ValueTensorDesc(program, values_output_id, &values_output_desc)) {
    return error;
  }
  if (PJRT_Error* error = ValueTensorDesc(program, top_k.indices_id(), &indices_output_desc)) {
    return error;
  }

  TtMetalTopKRequest request;
  request.local_hardware_id = target_device == nullptr ? 0 : target_device->local_hardware_id;
  request.k = top_k.k();
  if (PJRT_Error* error = TensorDescBufferType(*input_desc, &request.input.type)) {
    return error;
  }
  if (PJRT_Error* error =
          TensorDescBufferType(*values_output_desc, &request.values_output_type)) {
    return error;
  }
  if (PJRT_Error* error =
          TensorDescBufferType(*indices_output_desc, &request.indices_output_type)) {
    return error;
  }
  if (PJRT_Error* error = TensorDescDims(*input_desc, &request.input.dims)) {
    return error;
  }
  if (PJRT_Error* error = TensorDescDims(*values_output_desc, &request.values_output_dims)) {
    return error;
  }
  if (PJRT_Error* error =
          TensorDescDims(*indices_output_desc, &request.indices_output_dims)) {
    return error;
  }
  if (PJRT_Error* error =
          ReadBufferLogicalBytes(*values[top_k.operand_id()], &request.input.data)) {
    return error;
  }

  std::vector<std::byte> values_data;
  std::vector<std::byte> indices_data;
  if (PJRT_Error* error = ExecuteTtMetalTopK(request, &values_data, &indices_data)) {
    return error;
  }
  if (PJRT_Error* error = CreateBufferForValue(program, values_output_id, target_device,
                                               std::move(values_data), values_out)) {
    return error;
  }
  if (PJRT_Error* error = CreateBufferForValue(program, top_k.indices_id(), target_device,
                                               std::move(indices_data), indices_out)) {
    return error;
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

  std::vector<PJRT_Buffer*> values(static_cast<size_t>(program.values_size()), nullptr);
  std::vector<std::unique_ptr<PJRT_Buffer>> owned_values;
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
      case tt::Op::kConstant: {
        const tt::TensorDesc* output_desc = nullptr;
        if (PJRT_Error* error = ValueTensorDesc(program, op.output_id(), &output_desc)) {
          return error;
        }
        PJRT_Buffer_Type output_type = PJRT_Buffer_Type_INVALID;
        if (PJRT_Error* error = TensorDescBufferType(*output_desc, &output_type)) {
          return error;
        }
        std::vector<int64_t> output_dims;
        if (PJRT_Error* error = TensorDescDims(*output_desc, &output_dims)) {
          return error;
        }
        std::vector<std::byte> data;
        if (PJRT_Error* error = ConstantBytes(output_type, output_dims, op.constant(), &data)) {
          return error;
        }
        PJRT_Buffer* result_buffer = nullptr;
        if (PJRT_Error* error = CreateBufferForValue(program, op.output_id(), target_device,
                                                     std::move(data), &result_buffer)) {
          return error;
        }
        values[op.output_id()] = result_buffer;
        owned_values.emplace_back(result_buffer);
        break;
      }
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
      case tt::Op::kMatmul: {
        const tt::MatmulOp& matmul = op.matmul();
        if (matmul.has_top_k_epilogue()) {
          return Unimplemented("tt-metal matmul top_k epilogue is not implemented yet");
        }
        const uint32_t lhs_id = matmul.lhs_id();
        const uint32_t rhs_id = matmul.rhs_id();
        if (lhs_id >= static_cast<uint32_t>(values.size()) || values[lhs_id] == nullptr ||
            rhs_id >= static_cast<uint32_t>(values.size()) || values[rhs_id] == nullptr) {
          return Internal("matmul input value was not produced");
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

        TtMetalMatmulRequest request;
        request.local_hardware_id = target_device == nullptr ? 0 : target_device->local_hardware_id;
        if (PJRT_Error* error = TensorDescBufferType(lhs_desc.tensor(), &request.lhs.type)) {
          return error;
        }
        if (PJRT_Error* error = TensorDescBufferType(rhs_desc.tensor(), &request.rhs.type)) {
          return error;
        }
        if (PJRT_Error* error = TensorDescBufferType(output_desc.tensor(), &request.output_type)) {
          return error;
        }
        if (PJRT_Error* error = TensorDescDims(lhs_desc.tensor(), &request.lhs.dims)) {
          return error;
        }
        if (PJRT_Error* error = TensorDescDims(rhs_desc.tensor(), &request.rhs.dims)) {
          return error;
        }
        if (PJRT_Error* error = TensorDescDims(output_desc.tensor(), &request.output_dims)) {
          return error;
        }
        if (PJRT_Error* error = ReadBufferLogicalBytes(*values[lhs_id], &request.lhs.data)) {
          return error;
        }
        if (PJRT_Error* error = ReadBufferLogicalBytes(*values[rhs_id], &request.rhs.data)) {
          return error;
        }
        request.lhs_batching_dimensions =
            RepeatedI64ToVector(matmul.lhs_batching_dimensions());
        request.rhs_batching_dimensions =
            RepeatedI64ToVector(matmul.rhs_batching_dimensions());
        request.lhs_contracting_dimensions =
            RepeatedI64ToVector(matmul.lhs_contracting_dimensions());
        request.rhs_contracting_dimensions =
            RepeatedI64ToVector(matmul.rhs_contracting_dimensions());

        std::vector<std::byte> result_data;
        if (PJRT_Error* error = ExecuteTtMetalMatmul(request, &result_data)) {
          return error;
        }
        PJRT_Buffer* result_buffer = nullptr;
        const void* data = result_data.empty() ? nullptr : result_data.data();
        PJRT_Memory* output_memory =
            target_device == nullptr ? nullptr : target_device->default_memory;
        if (PJRT_Error* error = CreatePjrtBufferFromHostBytes(
                request.output_type, request.output_dims, target_device, output_memory, data,
                result_data.size(), &result_buffer)) {
          return error;
        }
        values[op.output_id()] = result_buffer;
        owned_values.emplace_back(result_buffer);
        break;
      }
      case tt::Op::kFusedElementwise: {
        PJRT_Buffer* result_buffer = nullptr;
        if (PJRT_Error* error = ExecuteFusedElementwiseOp(
                program, op.fused_elementwise(), op.output_id(), values, target_device,
                &result_buffer)) {
          return error;
        }
        values[op.output_id()] = result_buffer;
        owned_values.emplace_back(result_buffer);
        break;
      }
      case tt::Op::kSelect: {
        PJRT_Buffer* result_buffer = nullptr;
        if (PJRT_Error* error = ExecuteSelectOp(program, op.select(), op.output_id(), values,
                                                target_device, &result_buffer)) {
          return error;
        }
        values[op.output_id()] = result_buffer;
        owned_values.emplace_back(result_buffer);
        break;
      }
      case tt::Op::kBroadcastInDim: {
        PJRT_Buffer* result_buffer = nullptr;
        if (PJRT_Error* error = ExecuteBroadcastInDimOp(
                program, op.broadcast_in_dim(), op.output_id(), values, target_device,
                &result_buffer)) {
          return error;
        }
        values[op.output_id()] = result_buffer;
        owned_values.emplace_back(result_buffer);
        break;
      }
      case tt::Op::kIota: {
        PJRT_Buffer* result_buffer = nullptr;
        if (PJRT_Error* error = ExecuteIotaOp(program, op.iota(), op.output_id(),
                                              target_device, &result_buffer)) {
          return error;
        }
        values[op.output_id()] = result_buffer;
        owned_values.emplace_back(result_buffer);
        break;
      }
      case tt::Op::kConcatenate: {
        PJRT_Buffer* result_buffer = nullptr;
        if (PJRT_Error* error = ExecuteConcatenateOp(program, op.concatenate(),
                                                     op.output_id(), values,
                                                     target_device, &result_buffer)) {
          return error;
        }
        values[op.output_id()] = result_buffer;
        owned_values.emplace_back(result_buffer);
        break;
      }
      case tt::Op::kReduce: {
        PJRT_Buffer* result_buffer = nullptr;
        if (PJRT_Error* error = ExecuteReduceOp(program, op.reduce(), op.output_id(),
                                                values, target_device, &result_buffer)) {
          return error;
        }
        values[op.output_id()] = result_buffer;
        owned_values.emplace_back(result_buffer);
        break;
      }
      case tt::Op::kReshape: {
        PJRT_Buffer* result_buffer = nullptr;
        if (PJRT_Error* error = ExecuteReshapeOp(program, op.reshape(), op.output_id(),
                                                 values, target_device, &result_buffer)) {
          return error;
        }
        values[op.output_id()] = result_buffer;
        owned_values.emplace_back(result_buffer);
        break;
      }
      case tt::Op::kSlice: {
        PJRT_Buffer* result_buffer = nullptr;
        if (PJRT_Error* error = ExecuteSliceOp(program, op.slice(), op.output_id(),
                                               values, target_device, &result_buffer)) {
          return error;
        }
        values[op.output_id()] = result_buffer;
        owned_values.emplace_back(result_buffer);
        break;
      }
      case tt::Op::kTranspose: {
        PJRT_Buffer* result_buffer = nullptr;
        if (PJRT_Error* error = ExecuteTransposeOp(program, op.transpose(), op.output_id(),
                                                   values, target_device, &result_buffer)) {
          return error;
        }
        values[op.output_id()] = result_buffer;
        owned_values.emplace_back(result_buffer);
        break;
      }
      case tt::Op::kTopK: {
        PJRT_Buffer* values_buffer = nullptr;
        PJRT_Buffer* indices_buffer = nullptr;
        if (PJRT_Error* error = ExecuteTopKOp(program, op.top_k(), op.output_id(), values,
                                              target_device, &values_buffer, &indices_buffer)) {
          return error;
        }
        if (op.top_k().indices_id() >= static_cast<uint32_t>(values.size())) {
          return Internal("top_k indices output id is out of bounds");
        }
        values[op.output_id()] = values_buffer;
        values[op.top_k().indices_id()] = indices_buffer;
        owned_values.emplace_back(values_buffer);
        owned_values.emplace_back(indices_buffer);
        break;
      }
      case tt::Op::kGather: {
        PJRT_Buffer* result_buffer = nullptr;
        if (PJRT_Error* error = ExecuteGatherOp(program, op.gather(), op.output_id(), values,
                                                target_device, &result_buffer)) {
          return error;
        }
        values[op.output_id()] = result_buffer;
        owned_values.emplace_back(result_buffer);
        break;
      }
      default:
        return Unimplemented("C++ execution does not support op kind: " +
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
  DeletePjrtBufferStorage(args->buffer);
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
