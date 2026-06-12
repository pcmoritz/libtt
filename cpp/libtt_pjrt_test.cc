#include "cpp/libtt_pjrt.h"

#include <cassert>
#include <cstddef>
#include <cstring>
#include <string_view>

int main() {
  const PJRT_Api* api = GetPjrtApi();
  assert(api != nullptr);
  assert(api->PJRT_Client_Create != nullptr);
  assert(api->PJRT_Client_Destroy != nullptr);

  PJRT_Client_Create_Args create_args{};
  create_args.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
  PJRT_Error* error = api->PJRT_Client_Create(&create_args);
  assert(error == nullptr);
  assert(create_args.client != nullptr);

  PJRT_Client_PlatformName_Args platform_args{};
  platform_args.struct_size = PJRT_Client_PlatformName_Args_STRUCT_SIZE;
  platform_args.client = create_args.client;
  error = api->PJRT_Client_PlatformName(&platform_args);
  assert(error == nullptr);
  assert(std::string_view(platform_args.platform_name, platform_args.platform_name_size) == "tt");

  PJRT_Client_Devices_Args devices_args{};
  devices_args.struct_size = PJRT_Client_Devices_Args_STRUCT_SIZE;
  devices_args.client = create_args.client;
  error = api->PJRT_Client_Devices(&devices_args);
  assert(error == nullptr);

  PJRT_Client_Destroy_Args destroy_args{};
  destroy_args.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
  destroy_args.client = create_args.client;
  error = api->PJRT_Client_Destroy(&destroy_args);
  assert(error == nullptr);
  assert(destroy_args.client == nullptr);
  return 0;
}
