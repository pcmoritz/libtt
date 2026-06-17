#include "cpp/tt_metal_runtime.h"

#include <map>
#include <memory>
#include <mutex>

namespace {

using tt::tt_metal::distributed::MeshDevice;

class MeshDeviceCache {
 public:
  std::shared_ptr<MeshDevice> Get(int local_hardware_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& cached = devices_[local_hardware_id];
    if (!cached) {
      cached = MeshDevice::create_unit_mesh(local_hardware_id);
      cached->enable_program_cache();
    }
    return cached;
  }

 private:
  std::mutex mutex_;
  std::map<int, std::shared_ptr<MeshDevice>> devices_;
};

MeshDeviceCache& RuntimeDevices() {
  static MeshDeviceCache* cache = new MeshDeviceCache;
  return *cache;
}

}  // namespace

std::shared_ptr<MeshDevice> GetTtMetalMeshDevice(int local_hardware_id) {
  EnsureTtMetalRuntimeReady();
  return RuntimeDevices().Get(local_hardware_id);
}
