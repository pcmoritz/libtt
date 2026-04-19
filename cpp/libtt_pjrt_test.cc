#include <cstdlib>
#include <string>

#include "cpp/libtt_pjrt.h"

namespace {

std::string ErrorMessage(const PJRT_Api* api, PJRT_Error* error) {
  if (error == nullptr) {
    return {};
  }
  PJRT_Error_Message_Args message_args{
      .struct_size = PJRT_Error_Message_Args_STRUCT_SIZE,
      .extension_start = nullptr,
      .error = error,
      .message = nullptr,
      .message_size = 0,
  };
  api->PJRT_Error_Message(&message_args);
  std::string message(message_args.message, message_args.message_size);
  PJRT_Error_Destroy_Args destroy_args{
      .struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE,
      .extension_start = nullptr,
      .error = error,
  };
  api->PJRT_Error_Destroy(&destroy_args);
  return message;
}

int CheckExposesExpectedApiVersion() {
  const PJRT_Api* api = GetPjrtApi();
  if (api == nullptr) {
    return 1;
  }
  if (api->pjrt_api_version.major_version != PJRT_API_MAJOR) {
    return 2;
  }
  if (api->pjrt_api_version.minor_version != PJRT_API_MINOR) {
    return 3;
  }
  if (api->PJRT_Client_Create == nullptr || api->PJRT_Client_Destroy == nullptr) {
    return 4;
  }
  return 0;
}

int CheckClientCreateAndStrings() {
  const PJRT_Api* api = GetPjrtApi();
  if (api == nullptr) {
    return 10;
  }

  PJRT_Client_Create_Args create_args{
      .struct_size = PJRT_Client_Create_Args_STRUCT_SIZE,
      .extension_start = nullptr,
      .create_options = nullptr,
      .num_options = 0,
      .kv_get_callback = nullptr,
      .kv_get_user_arg = nullptr,
      .kv_put_callback = nullptr,
      .kv_put_user_arg = nullptr,
      .client = nullptr,
      .kv_try_get_callback = nullptr,
      .kv_try_get_user_arg = nullptr,
  };
  if (api->PJRT_Client_Create(&create_args) != nullptr || create_args.client == nullptr) {
    return 11;
  }

  PJRT_Client_PlatformName_Args name_args{
      .struct_size = PJRT_Client_PlatformName_Args_STRUCT_SIZE,
      .extension_start = nullptr,
      .client = create_args.client,
      .platform_name = nullptr,
      .platform_name_size = 0,
  };
  if (api->PJRT_Client_PlatformName(&name_args) != nullptr) {
    return 12;
  }
  if (std::string(name_args.platform_name, name_args.platform_name_size) != "tt") {
    return 13;
  }

  PJRT_Client_PlatformVersion_Args version_args{
      .struct_size = PJRT_Client_PlatformVersion_Args_STRUCT_SIZE,
      .extension_start = nullptr,
      .client = create_args.client,
      .platform_version = nullptr,
      .platform_version_size = 0,
  };
  if (api->PJRT_Client_PlatformVersion(&version_args) != nullptr) {
    return 14;
  }
  if (std::string(version_args.platform_version, version_args.platform_version_size)
          .find("libtt") == std::string::npos) {
    return 15;
  }

  PJRT_Client_Devices_Args devices_args{
      .struct_size = PJRT_Client_Devices_Args_STRUCT_SIZE,
      .extension_start = nullptr,
      .client = create_args.client,
      .devices = nullptr,
      .num_devices = 0,
  };
  if (api->PJRT_Client_Devices(&devices_args) != nullptr) {
    return 16;
  }

  PJRT_Client_Destroy_Args destroy_args{
      .struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE,
      .extension_start = nullptr,
      .client = create_args.client,
  };
  if (!ErrorMessage(api, api->PJRT_Client_Destroy(&destroy_args)).empty()) {
    return 17;
  }
  if (destroy_args.client != nullptr) {
    return 18;
  }
  return 0;
}

}  // namespace

int main() {
  if (int code = CheckExposesExpectedApiVersion(); code != 0) {
    return code;
  }
  if (int code = CheckClientCreateAndStrings(); code != 0) {
    return code;
  }
  return EXIT_SUCCESS;
}
