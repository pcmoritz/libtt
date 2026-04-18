#ifndef LIBTT_CPP_LIBTT_RUST_FFI_H_
#define LIBTT_CPP_LIBTT_RUST_FFI_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

extern "C" {

typedef struct TTRustDiscovery TTRustDiscovery;
typedef struct TTRustBufferHandle TTRustBufferHandle;

typedef struct TTRustError {
  int32_t code;
  char* message;
} TTRustError;

typedef struct TTDeviceInfo {
  int32_t id;
  int32_t local_hardware_id;
  const char* arch;
  const char* device_kind;
  const char* device_debug_string;
  const char* device_to_string;
  const char* memory_debug_string;
  const char* memory_to_string;
} TTDeviceInfo;

TTRustDiscovery* tt_rust_discovery_create();
void tt_rust_discovery_destroy(TTRustDiscovery* discovery);
size_t tt_rust_discovery_len(const TTRustDiscovery* discovery);
const TTDeviceInfo* tt_rust_discovery_devices(const TTRustDiscovery* discovery);

TTRustError* tt_rust_buffer_from_host(size_t local_hardware_id, int32_t buffer_type,
                                      const int64_t* dims, size_t num_dims, const void* data,
                                      size_t data_len, TTRustBufferHandle** out_buffer);
void tt_rust_buffer_destroy(TTRustBufferHandle* buffer);
void tt_rust_buffer_delete(TTRustBufferHandle* buffer);
bool tt_rust_buffer_is_deleted(const TTRustBufferHandle* buffer);
size_t tt_rust_buffer_size(const TTRustBufferHandle* buffer);
TTRustError* tt_rust_buffer_read(const TTRustBufferHandle* buffer, void* dst, size_t dst_len,
                                 size_t* out_len);

void tt_rust_error_destroy(TTRustError* error);

}  // extern "C"

#endif  // LIBTT_CPP_LIBTT_RUST_FFI_H_
