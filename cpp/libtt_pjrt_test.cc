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
  assert(devices_args.num_devices >= 1);
  assert(devices_args.devices != nullptr);

  PJRT_Client_AddressableDevices_Args addressable_devices_args{};
  addressable_devices_args.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
  addressable_devices_args.client = create_args.client;
  error = api->PJRT_Client_AddressableDevices(&addressable_devices_args);
  assert(error == nullptr);
  assert(addressable_devices_args.num_addressable_devices >= 1);
  assert(addressable_devices_args.addressable_devices != nullptr);

  PJRT_Client_ProcessIndex_Args process_index_args{};
  process_index_args.struct_size = PJRT_Client_ProcessIndex_Args_STRUCT_SIZE;
  process_index_args.client = create_args.client;
  error = api->PJRT_Client_ProcessIndex(&process_index_args);
  assert(error == nullptr);
  assert(process_index_args.process_index == 0);

  assert(api->PJRT_Client_Compile != nullptr);
  assert(api->PJRT_LoadedExecutable_GetExecutable != nullptr);
  assert(api->PJRT_Executable_OutputElementTypes != nullptr);
  assert(api->PJRT_Executable_OutputDimensions != nullptr);

  const char kFormat[] = "mlir";
  const char kProgram[] = R"mlir(
module {
  func.func public @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
    return %arg0 : tensor<2x2xf32>
  }
}
)mlir";

  PJRT_Program program{};
  program.struct_size = PJRT_Program_STRUCT_SIZE;
  program.code = const_cast<char*>(kProgram);
  program.code_size = std::strlen(kProgram);
  program.format = kFormat;
  program.format_size = std::strlen(kFormat);

  PJRT_Client_Compile_Args compile_args{};
  compile_args.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
  compile_args.client = create_args.client;
  compile_args.program = &program;
  error = api->PJRT_Client_Compile(&compile_args);
  assert(error == nullptr);
  assert(compile_args.executable != nullptr);

  PJRT_LoadedExecutable_GetExecutable_Args get_exec_args{};
  get_exec_args.struct_size = PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE;
  get_exec_args.loaded_executable = compile_args.executable;
  error = api->PJRT_LoadedExecutable_GetExecutable(&get_exec_args);
  assert(error == nullptr);
  assert(get_exec_args.executable != nullptr);

  PJRT_Executable_OutputElementTypes_Args output_types_args{};
  output_types_args.struct_size = PJRT_Executable_OutputElementTypes_Args_STRUCT_SIZE;
  output_types_args.executable = get_exec_args.executable;
  error = api->PJRT_Executable_OutputElementTypes(&output_types_args);
  assert(error == nullptr);
  assert(output_types_args.num_output_types == 1);
  assert(output_types_args.output_types[0] == PJRT_Buffer_Type_F32);

  PJRT_Executable_OutputDimensions_Args output_dims_args{};
  output_dims_args.struct_size = PJRT_Executable_OutputDimensions_Args_STRUCT_SIZE;
  output_dims_args.executable = get_exec_args.executable;
  error = api->PJRT_Executable_OutputDimensions(&output_dims_args);
  assert(error == nullptr);
  assert(output_dims_args.num_outputs == 1);
  assert(output_dims_args.dim_sizes[0] == 2);
  assert(output_dims_args.dims[0] == 2);
  assert(output_dims_args.dims[1] == 2);

  PJRT_Executable_Destroy_Args exec_destroy_args{};
  exec_destroy_args.struct_size = PJRT_Executable_Destroy_Args_STRUCT_SIZE;
  exec_destroy_args.executable = get_exec_args.executable;
  error = api->PJRT_Executable_Destroy(&exec_destroy_args);
  assert(error == nullptr);

  PJRT_LoadedExecutable_Destroy_Args loaded_destroy_args{};
  loaded_destroy_args.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
  loaded_destroy_args.executable = compile_args.executable;
  error = api->PJRT_LoadedExecutable_Destroy(&loaded_destroy_args);
  assert(error == nullptr);

  PJRT_Client_Destroy_Args destroy_args{};
  destroy_args.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
  destroy_args.client = create_args.client;
  error = api->PJRT_Client_Destroy(&destroy_args);
  assert(error == nullptr);
  assert(destroy_args.client == nullptr);
  return 0;
}
