#include "cpp/libtt_pjrt.h"

#include <array>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>

namespace {

[[noreturn]] void FailWithError(const PJRT_Api* api, PJRT_Error* error) {
  PJRT_Error_Message_Args message_args{};
  message_args.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
  message_args.error = error;
  api->PJRT_Error_Message(&message_args);
  std::fprintf(stderr, "PJRT error: %.*s\n", static_cast<int>(message_args.message_size),
               message_args.message == nullptr ? "" : message_args.message);

  PJRT_Error_Destroy_Args destroy_args{};
  destroy_args.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
  destroy_args.error = error;
  api->PJRT_Error_Destroy(&destroy_args);
  std::abort();
}

void Check(const PJRT_Api* api, PJRT_Error* error) {
  if (error != nullptr) {
    FailWithError(api, error);
  }
}

void DestroyEvent(const PJRT_Api* api, PJRT_Event* event) {
  if (event == nullptr) {
    return;
  }
  PJRT_Event_Destroy_Args args{};
  args.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
  args.event = event;
  Check(api, api->PJRT_Event_Destroy(&args));
}

void DestroyBuffer(const PJRT_Api* api, PJRT_Buffer* buffer) {
  if (buffer == nullptr) {
    return;
  }
  PJRT_Buffer_Destroy_Args args{};
  args.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
  args.buffer = buffer;
  Check(api, api->PJRT_Buffer_Destroy(&args));
}

}  // namespace

int main() {
  setenv("LIBTT_PJRT_HOST_FALLBACK", "1", 1);

  const PJRT_Api* api = GetPjrtApi();
  assert(api != nullptr);

  PJRT_Client_Create_Args create_args{};
  create_args.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
  Check(api, api->PJRT_Client_Create(&create_args));

  PJRT_Client_AddressableDevices_Args devices_args{};
  devices_args.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
  devices_args.client = create_args.client;
  Check(api, api->PJRT_Client_AddressableDevices(&devices_args));
  assert(devices_args.num_addressable_devices >= 1);
  PJRT_Device* device = devices_args.addressable_devices[0];

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
  Check(api, api->PJRT_Client_Compile(&compile_args));

  const std::array<int64_t, 2> dims = {2, 2};
  const std::array<float, 4> input = {1.0f, 2.5f, -3.0f, 4.25f};

  PJRT_Client_BufferFromHostBuffer_Args buffer_args{};
  buffer_args.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
  buffer_args.client = create_args.client;
  buffer_args.data = input.data();
  buffer_args.type = PJRT_Buffer_Type_F32;
  buffer_args.dims = dims.data();
  buffer_args.num_dims = dims.size();
  buffer_args.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
  buffer_args.device = device;
  Check(api, api->PJRT_Client_BufferFromHostBuffer(&buffer_args));
  DestroyEvent(api, buffer_args.done_with_host_buffer);

  PJRT_Buffer* argument_buffers[] = {buffer_args.buffer};
  PJRT_Buffer** argument_lists[] = {argument_buffers};
  PJRT_Buffer* output_buffers[] = {nullptr};
  PJRT_Buffer** output_lists[] = {output_buffers};
  PJRT_Event* complete_events[] = {nullptr};

  PJRT_LoadedExecutable_Execute_Args execute_args{};
  execute_args.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
  execute_args.executable = compile_args.executable;
  execute_args.argument_lists = argument_lists;
  execute_args.num_devices = 1;
  execute_args.num_args = 1;
  execute_args.output_lists = output_lists;
  execute_args.device_complete_events = complete_events;
  execute_args.execute_device = device;
  Check(api, api->PJRT_LoadedExecutable_Execute(&execute_args));
  DestroyEvent(api, complete_events[0]);
  assert(output_buffers[0] != nullptr);

  std::array<float, 4> output = {};
  PJRT_Buffer_ToHostBuffer_Args to_host_args{};
  to_host_args.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
  to_host_args.src = output_buffers[0];
  to_host_args.dst = output.data();
  to_host_args.dst_size = sizeof(output);
  Check(api, api->PJRT_Buffer_ToHostBuffer(&to_host_args));
  DestroyEvent(api, to_host_args.event);

  assert(output == input);
  std::printf("pjrt_identity: ok\n");

  DestroyBuffer(api, output_buffers[0]);
  DestroyBuffer(api, buffer_args.buffer);

  PJRT_LoadedExecutable_Destroy_Args loaded_destroy_args{};
  loaded_destroy_args.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
  loaded_destroy_args.executable = compile_args.executable;
  Check(api, api->PJRT_LoadedExecutable_Destroy(&loaded_destroy_args));

  PJRT_Client_Destroy_Args destroy_args{};
  destroy_args.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
  destroy_args.client = create_args.client;
  Check(api, api->PJRT_Client_Destroy(&destroy_args));
  return 0;
}
