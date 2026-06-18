#include "api/event_instance.h"
#include "api_bindings.h"
#include "xla/pjrt/c/pjrt_c_api.h"

#include <cstring>
#include <string_view>
#include <type_traits>

#if defined(_WIN32) || defined(__CYGWIN__)
#define LIBTT_PJRT_EXPORTED __declspec(dllexport)
#else
#define LIBTT_PJRT_EXPORTED __attribute__((visibility("default")))
#endif

namespace {

PJRT_Api* g_api = nullptr;

void DestroyError(PJRT_Error* error) {
  if (error == nullptr || g_api == nullptr || g_api->PJRT_Error_Destroy == nullptr) {
    return;
  }
  PJRT_Error_Destroy_Args args;
  args.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.error = error;
  g_api->PJRT_Error_Destroy(&args);
}

int64_t DeviceDramSizeBytes(PJRT_Device* device) {
  if (g_api == nullptr || g_api->PJRT_Device_GetDescription == nullptr ||
      g_api->PJRT_DeviceDescription_Attributes == nullptr) {
    return 0;
  }

  PJRT_Device_GetDescription_Args description_args;
  description_args.struct_size = PJRT_Device_GetDescription_Args_STRUCT_SIZE;
  description_args.extension_start = nullptr;
  description_args.device = device;
  description_args.device_description = nullptr;
  PJRT_Error* error = g_api->PJRT_Device_GetDescription(&description_args);
  if (error != nullptr) {
    DestroyError(error);
    return 0;
  }

  PJRT_DeviceDescription_Attributes_Args attributes_args;
  attributes_args.struct_size = PJRT_DeviceDescription_Attributes_Args_STRUCT_SIZE;
  attributes_args.extension_start = nullptr;
  attributes_args.device_description = description_args.device_description;
  attributes_args.num_attributes = 0;
  attributes_args.attributes = nullptr;
  error = g_api->PJRT_DeviceDescription_Attributes(&attributes_args);
  if (error != nullptr) {
    DestroyError(error);
    return 0;
  }

  for (size_t i = 0; i < attributes_args.num_attributes; ++i) {
    const PJRT_NamedValue& attribute = attributes_args.attributes[i];
    const std::string_view name(attribute.name, attribute.name_size);
    if (name == "dram_size_bytes" && attribute.type == PJRT_NamedValue_kInt64 &&
        attribute.int64_value > 0) {
      return attribute.int64_value;
    }
  }
  return 0;
}

PJRT_Error* LibttDeviceMemoryStats(PJRT_Device_MemoryStats_Args* args) {
  const int64_t bytes_limit = DeviceDramSizeBytes(args->device);

  args->bytes_in_use = 0;
  args->peak_bytes_in_use = 0;
  args->peak_bytes_in_use_is_set = false;
  args->num_allocs = 0;
  args->num_allocs_is_set = false;
  args->largest_alloc_size = 0;
  args->largest_alloc_size_is_set = false;
  args->bytes_limit = bytes_limit;
  args->bytes_limit_is_set = bytes_limit > 0;
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

PJRT_Error* LibttDeviceClearMemoryStats(PJRT_Device_ClearMemoryStats_Args* args) {
  (void)args;
  return nullptr;
}

void InstallLibttApiOverrides(PJRT_Api* api) {
  g_api = api;
  api->PJRT_Device_MemoryStats = LibttDeviceMemoryStats;
  api->PJRT_Device_ClearMemoryStats = LibttDeviceClearMemoryStats;
}

PJRT_Api* GetRawAPIStruct() {
  static PJRT_Api static_api;
  return &static_api;
}

PJRT_Api* GetInitializedAPIStruct() {
  static PJRT_Api* initialized_api = [] {
    PJRT_Api* api = GetRawAPIStruct();
    std::memset(api, 0, sizeof(PJRT_Api));
    tt::pjrt::bindApi(api);
    InstallLibttApiOverrides(api);
    return api;
  }();
  return initialized_api;
}

}  // namespace

typedef unsigned PJRT_Plugin_ApiVersion_FN();
typedef PJRT_Api* PJRT_Plugin_Create_FN();

extern "C" LIBTT_PJRT_EXPORTED unsigned GetPjrtApiVersion();
static_assert(std::is_same_v<decltype(GetPjrtApiVersion), PJRT_Plugin_ApiVersion_FN>);

extern "C" LIBTT_PJRT_EXPORTED PJRT_Api* GetPjrtApi();
static_assert(std::is_same_v<decltype(GetPjrtApi), PJRT_Plugin_Create_FN>);

unsigned GetPjrtApiVersion() { return 1; }

PJRT_Api* GetPjrtApi() { return GetInitializedAPIStruct(); }

extern "C" LIBTT_PJRT_EXPORTED void tt_pjrt_shutdown() {
  tt::pjrt::EventInstance::shutdownCallbackWorker();
}
