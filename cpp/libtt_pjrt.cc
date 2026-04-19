#include "cpp/libtt_pjrt.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "cpp/libtt_rust_ffi.h"

struct PJRT_Error {
  PJRT_Error_Code code;
  std::string message;
};

struct EventError {
  PJRT_Error_Code code;
  std::string message;
};

struct PJRT_Event {
  std::optional<EventError> error;
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
  TTRustBufferHandle* rust_buffer;
  size_t local_hardware_id;
  bool deleted;
  size_t external_reference_count;
};

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
constexpr const char* kPlatformVersion = "libtt cpp+rustrt 0.1.0";

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

PJRT_Error* Internal(std::string message) {
  return MakePjrtError(PJRT_Error_Code_INTERNAL, std::move(message));
}

PJRT_Error* FromRustError(TTRustError* error) {
  if (error == nullptr) {
    return nullptr;
  }
  const std::string message = error->message != nullptr ? error->message : "unknown Rust error";
  const auto code = static_cast<PJRT_Error_Code>(error->code);
  tt_rust_error_destroy(error);
  return MakePjrtError(code, message);
}

PJRT_Event* ReadyEvent() { return new PJRT_Event{std::nullopt}; }

PJRT_Event* EventWithError(PJRT_Error_Code code, std::string message) {
  return new PJRT_Event{EventError{code, std::move(message)}};
}

PJRT_Error* CloneEventError(const PJRT_Event* event) {
  if (event == nullptr || !event->error.has_value()) {
    return nullptr;
  }
  return MakePjrtError(event->error->code, event->error->message);
}

std::string CopyCString(const char* value) { return value == nullptr ? "" : value; }

size_t BytesPerElement(PJRT_Buffer_Type type) {
  switch (type) {
    case PJRT_Buffer_Type_S32:
    case PJRT_Buffer_Type_U32:
    case PJRT_Buffer_Type_F32:
      return 4;
    case PJRT_Buffer_Type_U16:
    case PJRT_Buffer_Type_F16:
    case PJRT_Buffer_Type_BF16:
      return 2;
    case PJRT_Buffer_Type_S8:
    case PJRT_Buffer_Type_U8:
      return 1;
    default:
      return 0;
  }
}

bool IsSupportedBufferType(PJRT_Buffer_Type type) { return BytesPerElement(type) != 0; }

PJRT_Error* CopyDims(const int64_t* dims, size_t num_dims, std::vector<int64_t>* out) {
  if (num_dims == 0) {
    out->clear();
    return nullptr;
  }
  if (dims == nullptr) {
    return InvalidArgument("dims must not be null when num_dims > 0");
  }
  out->assign(dims, dims + num_dims);
  return nullptr;
}

PJRT_Error* DimsToSizeT(const std::vector<int64_t>& dims, std::vector<size_t>* out) {
  out->clear();
  out->reserve(dims.size());
  for (int64_t dim : dims) {
    if (dim < 0) {
      return InvalidArgument("shape dimensions must be >= 0");
    }
    out->push_back(static_cast<size_t>(dim));
  }
  return nullptr;
}

PJRT_Error* HostByteSize(PJRT_Buffer_Type type, const std::vector<size_t>& dims, size_t* out) {
  const size_t bytes_per_element = BytesPerElement(type);
  if (bytes_per_element == 0) {
    return Unimplemented("unsupported PJRT buffer type");
  }
  size_t elements = 1;
  for (size_t dim : dims) {
    if (dim != 0 && elements > static_cast<size_t>(-1) / dim) {
      return MakePjrtError(PJRT_Error_Code_RESOURCE_EXHAUSTED, "host buffer size overflow");
    }
    elements *= dim;
  }
  if (bytes_per_element != 0 && elements > static_cast<size_t>(-1) / bytes_per_element) {
    return MakePjrtError(PJRT_Error_Code_RESOURCE_EXHAUSTED, "host buffer size overflow");
  }
  *out = elements * bytes_per_element;
  return nullptr;
}

PJRT_Error* ValidateDenseRowMajorStrides(PJRT_Buffer_Type type, const std::vector<size_t>& dims,
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
      return InvalidArgument("byte strides must be >= 0");
    }
    if (static_cast<size_t>(stride) != expected) {
      return Unimplemented("only dense row-major host buffers are supported");
    }
    const size_t dim = std::max<size_t>(dims[i - 1], 1);
    if (dim != 0 && expected > static_cast<size_t>(-1) / dim) {
      return MakePjrtError(PJRT_Error_Code_RESOURCE_EXHAUSTED, "byte stride overflow");
    }
    expected *= dim;
  }
  return nullptr;
}

PJRT_Client* CreateClient() {
  auto* client = new PJRT_Client;
  client->platform_name = kPlatformName;
  client->platform_version = kPlatformVersion;

  TTRustDiscovery* discovery = tt_rust_discovery_create();
  const size_t device_count = discovery == nullptr ? 0 : tt_rust_discovery_len(discovery);
  const TTDeviceInfo* infos = discovery == nullptr ? nullptr : tt_rust_discovery_devices(discovery);

  client->device_descriptions_storage.reserve(device_count);
  client->memories_storage.reserve(device_count);
  client->devices_storage.reserve(device_count);
  client->device_ptrs.reserve(device_count);
  client->memory_ptrs.reserve(device_count);

  for (size_t i = 0; i < device_count; ++i) {
    const TTDeviceInfo& info = infos[i];
    client->device_descriptions_storage.push_back(PJRT_DeviceDescription{
        info.id,
        0,
        CopyCString(info.device_kind),
        CopyCString(info.device_debug_string),
        CopyCString(info.device_to_string),
    });
    client->memories_storage.push_back(PJRT_Memory{
        info.id,
        "dram",
        CopyCString(info.memory_debug_string),
        CopyCString(info.memory_to_string),
        {},
    });
  }

  for (size_t i = 0; i < client->memories_storage.size(); ++i) {
    client->memory_ptrs.push_back(&client->memories_storage[i]);
  }

  for (size_t i = 0; i < device_count; ++i) {
    const TTDeviceInfo& info = infos[i];
    client->devices_storage.push_back(PJRT_Device{
        info.id,
        info.local_hardware_id,
        &client->device_descriptions_storage[i],
        true,
        &client->memories_storage[i],
        {&client->memories_storage[i]},
    });
  }

  for (size_t i = 0; i < client->devices_storage.size(); ++i) {
    client->device_ptrs.push_back(&client->devices_storage[i]);
    client->addressable_device_ptrs.push_back(&client->devices_storage[i]);
    client->memories_storage[i].device_ptrs.push_back(&client->devices_storage[i]);
  }

  client->topology.platform_name = client->platform_name;
  client->topology.platform_version = client->platform_version;
  client->topology.device_description_ptrs.reserve(client->device_descriptions_storage.size());
  for (auto& description : client->device_descriptions_storage) {
    client->topology.device_description_ptrs.push_back(&description);
  }

  if (discovery != nullptr) {
    tt_rust_discovery_destroy(discovery);
  }
  return client;
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

extern "C" PJRT_Error* TT_Plugin_Initialize(PJRT_Plugin_Initialize_Args* args) {
  if (args == nullptr) {
    return InvalidArgument("args must not be null");
  }
  return nullptr;
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
  if (args == nullptr) {
    return InvalidArgument("args must not be null");
  }
  if (args->event == nullptr) {
    return InvalidArgument("event must not be null");
  }
  args->is_ready = true;
  return nullptr;
}

extern "C" PJRT_Error* TT_Event_Error(PJRT_Event_Error_Args* args) {
  if (args == nullptr) {
    return InvalidArgument("args must not be null");
  }
  if (args->event == nullptr) {
    return InvalidArgument("event must not be null");
  }
  return CloneEventError(args->event);
}

extern "C" PJRT_Error* TT_Event_Await(PJRT_Event_Await_Args* args) {
  if (args == nullptr) {
    return InvalidArgument("args must not be null");
  }
  if (args->event == nullptr) {
    return InvalidArgument("event must not be null");
  }
  return CloneEventError(args->event);
}

extern "C" PJRT_Error* TT_Event_OnReady(PJRT_Event_OnReady_Args* args) {
  if (args == nullptr) {
    return InvalidArgument("args must not be null");
  }
  if (args->event == nullptr) {
    return InvalidArgument("event must not be null");
  }
  if (args->callback == nullptr) {
    return InvalidArgument("callback must not be null");
  }
  args->callback(CloneEventError(args->event), args->user_arg);
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
  args->platform_name = args->client->platform_name.data();
  args->platform_name_size = args->client->platform_name.size();
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
  args->platform_version = args->client->platform_version.data();
  args->platform_version_size = args->client->platform_version.size();
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
  args->addressable_devices =
      args->client->addressable_device_ptrs.empty()
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

extern "C" PJRT_Error* TT_Client_DefaultDeviceAssignment(
    PJRT_Client_DefaultDeviceAssignment_Args* args) {
  if (args == nullptr || args->client == nullptr) {
    return InvalidArgument("client must not be null");
  }
  if (args->num_replicas < 0 || args->num_partitions < 0) {
    return InvalidArgument("num_replicas and num_partitions must be >= 0");
  }
  const size_t required =
      static_cast<size_t>(args->num_replicas) * static_cast<size_t>(args->num_partitions);
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

  std::vector<int64_t> dims_i64;
  if (PJRT_Error* error = CopyDims(args->dims, args->num_dims, &dims_i64)) {
    return error;
  }
  std::vector<size_t> dims;
  if (PJRT_Error* error = DimsToSizeT(dims_i64, &dims)) {
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

  PJRT_Device* target_device = args->device;
  if (target_device == nullptr && args->memory != nullptr && !args->memory->device_ptrs.empty()) {
    target_device = args->memory->device_ptrs.front();
  }
  if (target_device == nullptr && !args->client->addressable_device_ptrs.empty()) {
    target_device = args->client->addressable_device_ptrs.front();
  }
  if (target_device == nullptr) {
    return InvalidArgument("no target device available");
  }
  PJRT_Memory* target_memory =
      args->memory != nullptr ? args->memory : target_device->default_memory;

  TTRustBufferHandle* rust_buffer = nullptr;
  TTRustError* rust_error = tt_rust_buffer_from_host(
      static_cast<size_t>(target_device->local_hardware_id), static_cast<int32_t>(args->type),
      dims_i64.data(), dims_i64.size(), args->data, byte_size, &rust_buffer);
  if (rust_error != nullptr) {
    return FromRustError(rust_error);
  }

  args->done_with_host_buffer = ReadyEvent();
  args->buffer = new PJRT_Buffer{
      args->type,
      dims_i64,
      target_device,
      target_memory,
      rust_buffer,
      static_cast<size_t>(target_device->local_hardware_id),
      false,
      0,
  };
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
  args->device_kind = args->device_description->device_kind.data();
  args->device_kind_size = args->device_description->device_kind.size();
  return nullptr;
}

extern "C" PJRT_Error* TT_DeviceDescription_DebugString(
    PJRT_DeviceDescription_DebugString_Args* args) {
  if (args == nullptr || args->device_description == nullptr) {
    return InvalidArgument("device_description must not be null");
  }
  args->debug_string = args->device_description->debug_string.data();
  args->debug_string_size = args->device_description->debug_string.size();
  return nullptr;
}

extern "C" PJRT_Error* TT_DeviceDescription_ToString(
    PJRT_DeviceDescription_ToString_Args* args) {
  if (args == nullptr || args->device_description == nullptr) {
    return InvalidArgument("device_description must not be null");
  }
  args->to_string = args->device_description->to_string.data();
  args->to_string_size = args->device_description->to_string.size();
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
  args->kind = args->memory->kind.data();
  args->kind_size = args->memory->kind.size();
  return nullptr;
}

extern "C" PJRT_Error* TT_Memory_DebugString(PJRT_Memory_DebugString_Args* args) {
  if (args == nullptr || args->memory == nullptr) {
    return InvalidArgument("memory must not be null");
  }
  args->debug_string = args->memory->debug_string.data();
  args->debug_string_size = args->memory->debug_string.size();
  return nullptr;
}

extern "C" PJRT_Error* TT_Memory_ToString(PJRT_Memory_ToString_Args* args) {
  if (args == nullptr || args->memory == nullptr) {
    return InvalidArgument("memory must not be null");
  }
  args->to_string = args->memory->to_string.data();
  args->to_string_size = args->memory->to_string.size();
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
  if (args->buffer != nullptr) {
    tt_rust_buffer_destroy(args->buffer->rust_buffer);
    delete args->buffer;
    args->buffer = nullptr;
  }
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
  if (args->buffer->deleted || tt_rust_buffer_is_deleted(args->buffer->rust_buffer)) {
    return FailedPrecondition("buffer has been deleted");
  }
  args->on_device_size_in_bytes = tt_rust_buffer_size(args->buffer->rust_buffer);
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
  tt_rust_buffer_delete(args->buffer->rust_buffer);
  args->buffer->deleted = true;
  return nullptr;
}

extern "C" PJRT_Error* TT_Buffer_IsDeleted(PJRT_Buffer_IsDeleted_Args* args) {
  if (args == nullptr || args->buffer == nullptr) {
    return InvalidArgument("buffer must not be null");
  }
  args->is_deleted = args->buffer->deleted || tt_rust_buffer_is_deleted(args->buffer->rust_buffer);
  return nullptr;
}

extern "C" PJRT_Error* TT_Buffer_ToHostBuffer(PJRT_Buffer_ToHostBuffer_Args* args) {
  if (args == nullptr || args->src == nullptr) {
    return InvalidArgument("src must not be null");
  }
  if (args->host_layout != nullptr) {
    return Unimplemented("custom host layouts are not supported");
  }
  if (args->src->deleted || tt_rust_buffer_is_deleted(args->src->rust_buffer)) {
    return FailedPrecondition("buffer has been deleted");
  }

  std::vector<size_t> dims;
  if (PJRT_Error* error = DimsToSizeT(args->src->dims, &dims)) {
    return error;
  }
  size_t byte_size = 0;
  if (PJRT_Error* error = HostByteSize(args->src->buffer_type, dims, &byte_size)) {
    return error;
  }
  if (args->dst == nullptr) {
    args->dst_size = byte_size;
    args->event = ReadyEvent();
    return nullptr;
  }
  if (args->dst_size < byte_size) {
    return InvalidArgument("dst buffer is too small");
  }

  size_t out_len = 0;
  if (TTRustError* error =
          tt_rust_buffer_read(args->src->rust_buffer, args->dst, args->dst_size, &out_len)) {
    return FromRustError(error);
  }
  args->dst_size = out_len;
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
  args->event = (args->buffer->deleted || tt_rust_buffer_is_deleted(args->buffer->rust_buffer))
                    ? EventWithError(PJRT_Error_Code_FAILED_PRECONDITION,
                                     "buffer has been deleted")
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

extern "C" PJRT_Error* TT_TopologyDescription_PlatformName(
    PJRT_TopologyDescription_PlatformName_Args* args) {
  if (args == nullptr || args->topology == nullptr) {
    return InvalidArgument("topology must not be null");
  }
  args->platform_name = args->topology->platform_name.data();
  args->platform_name_size = args->topology->platform_name.size();
  return nullptr;
}

extern "C" PJRT_Error* TT_TopologyDescription_PlatformVersion(
    PJRT_TopologyDescription_PlatformVersion_Args* args) {
  if (args == nullptr || args->topology == nullptr) {
    return InvalidArgument("topology must not be null");
  }
  args->platform_version = args->topology->platform_version.data();
  args->platform_version_size = args->topology->platform_version.size();
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
      .struct_size = PJRT_Api_Version_STRUCT_SIZE,
      .extension_start = nullptr,
      .major_version = PJRT_API_MAJOR,
      .minor_version = PJRT_API_MINOR,
  };

  api.PJRT_Error_Destroy = TT_Error_Destroy;
  api.PJRT_Error_Message = TT_Error_Message;
  api.PJRT_Error_GetCode = TT_Error_GetCode;

  api.PJRT_Plugin_Initialize = TT_Plugin_Initialize;
  api.PJRT_Plugin_Attributes = TT_Plugin_Attributes;

  api.PJRT_Event_Destroy = TT_Event_Destroy;
  api.PJRT_Event_IsReady = TT_Event_IsReady;
  api.PJRT_Event_Error = TT_Event_Error;
  api.PJRT_Event_Await = TT_Event_Await;
  api.PJRT_Event_OnReady = TT_Event_OnReady;

  api.PJRT_Client_Create = TT_Client_Create;
  api.PJRT_Client_Destroy = TT_Client_Destroy;
  api.PJRT_Client_PlatformName = TT_Client_PlatformName;
  api.PJRT_Client_ProcessIndex = TT_Client_ProcessIndex;
  api.PJRT_Client_PlatformVersion = TT_Client_PlatformVersion;
  api.PJRT_Client_Devices = TT_Client_Devices;
  api.PJRT_Client_AddressableDevices = TT_Client_AddressableDevices;
  api.PJRT_Client_LookupDevice = TT_Client_LookupDevice;
  api.PJRT_Client_LookupAddressableDevice = TT_Client_LookupAddressableDevice;
  api.PJRT_Client_AddressableMemories = TT_Client_AddressableMemories;
  api.PJRT_Client_Compile = nullptr;
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
  api.PJRT_Device_MemoryStats = nullptr;

  api.PJRT_Memory_Id = TT_Memory_Id;
  api.PJRT_Memory_Kind = TT_Memory_Kind;
  api.PJRT_Memory_DebugString = TT_Memory_DebugString;
  api.PJRT_Memory_ToString = TT_Memory_ToString;
  api.PJRT_Memory_AddressableByDevices = TT_Memory_AddressableByDevices;

  api.PJRT_Buffer_Destroy = TT_Buffer_Destroy;
  api.PJRT_Buffer_ElementType = TT_Buffer_ElementType;
  api.PJRT_Buffer_Dimensions = TT_Buffer_Dimensions;
  api.PJRT_Buffer_UnpaddedDimensions = TT_Buffer_UnpaddedDimensions;
  api.PJRT_Buffer_DynamicDimensionIndices = TT_Buffer_DynamicDimensionIndices;
  api.PJRT_Buffer_GetMemoryLayout = nullptr;
  api.PJRT_Buffer_OnDeviceSizeInBytes = TT_Buffer_OnDeviceSizeInBytes;
  api.PJRT_Buffer_Device = TT_Buffer_Device;
  api.PJRT_Buffer_Memory = TT_Buffer_Memory;
  api.PJRT_Buffer_Delete = TT_Buffer_Delete;
  api.PJRT_Buffer_IsDeleted = TT_Buffer_IsDeleted;
  api.PJRT_Buffer_CopyToDevice = nullptr;
  api.PJRT_Buffer_ToHostBuffer = TT_Buffer_ToHostBuffer;
  api.PJRT_Buffer_IsOnCpu = TT_Buffer_IsOnCpu;
  api.PJRT_Buffer_ReadyEvent = TT_Buffer_ReadyEvent;
  api.PJRT_Buffer_UnsafePointer = nullptr;
  api.PJRT_Buffer_IncreaseExternalReferenceCount = TT_Buffer_IncreaseExternalReferenceCount;
  api.PJRT_Buffer_DecreaseExternalReferenceCount = TT_Buffer_DecreaseExternalReferenceCount;
  api.PJRT_Buffer_OpaqueDeviceMemoryDataPointer = nullptr;

  api.PJRT_Client_TopologyDescription = TT_Client_TopologyDescription;
  api.PJRT_TopologyDescription_PlatformName = TT_TopologyDescription_PlatformName;
  api.PJRT_TopologyDescription_PlatformVersion = TT_TopologyDescription_PlatformVersion;
  api.PJRT_TopologyDescription_GetDeviceDescriptions =
      TT_TopologyDescription_GetDeviceDescriptions;
  api.PJRT_TopologyDescription_Attributes = TT_TopologyDescription_Attributes;
  return api;
}

const PJRT_Api kPjrtApi = MakePjrtApi();

}  // namespace

extern "C" __attribute__((visibility("default"))) const PJRT_Api* GetPjrtApi() {
  return &kPjrtApi;
}
