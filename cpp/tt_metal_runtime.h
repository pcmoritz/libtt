#ifndef LIBTT_CPP_TT_METAL_RUNTIME_H_
#define LIBTT_CPP_TT_METAL_RUNTIME_H_

#include <memory>

#include <tt-metalium/mesh_device.hpp>

void EnsureTtMetalRuntimeReady();

std::shared_ptr<tt::tt_metal::distributed::MeshDevice>
GetTtMetalMeshDevice(int local_hardware_id);

#endif  // LIBTT_CPP_TT_METAL_RUNTIME_H_
