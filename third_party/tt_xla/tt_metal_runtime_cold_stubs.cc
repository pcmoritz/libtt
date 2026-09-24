#include <tt-metalium/experimental/inspector_config.hpp>
#include "tt_metal/impl/debug/noc_logging.hpp"
#include "tt_metal/impl/debug/dprint_server.hpp"
#include "tt_metal/impl/debug/watcher_server.hpp"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace tt::tt_metal {

class DPrintServer::Impl {};

DPrintServer::DPrintServer(MetalContext *, MetalEnv &, uint8_t,
                           const DispatchCoreConfig &) {}
DPrintServer::~DPrintServer() = default;
void DPrintServer::set_mute(bool) {}
void DPrintServer::await() {}
void DPrintServer::attach_devices() {}
void DPrintServer::detach_devices() {}
void DPrintServer::clear_log_file() {}
bool DPrintServer::reads_dispatch_cores(ChipId) { return false; }
std::vector<umd::CoreDescriptor> DPrintServer::get_print_cores(ChipId) const {
  return {};
}
std::vector<DPrintBufferInfo>
DPrintServer::get_core_buffers(ChipId, const umd::CoreDescriptor &) const {
  return {};
}
bool DPrintServer::hang_detected() { return false; }

class WatcherServer::Impl {};

WatcherServer::WatcherServer(MetalEnv &) {}
WatcherServer::~WatcherServer() = default;
void WatcherServer::init_devices() {}
void WatcherServer::attach_devices() {}
void WatcherServer::detach_devices() {}
void WatcherServer::clear_log() {}
std::string WatcherServer::log_file_name() { return {}; }
int WatcherServer::register_kernel(const std::string &) { return -1; }
void WatcherServer::register_kernel_elf_paths(int, std::vector<std::string> &) {}
bool WatcherServer::killed_due_to_error() { return false; }
void WatcherServer::set_killed_due_to_error_flag(bool) {}
std::string WatcherServer::exception_message() { return {}; }
void WatcherServer::set_exception_message(const std::string &) {}
int WatcherServer::dump_count() { return 0; }
std::unique_lock<std::mutex> WatcherServer::get_lock() {
  static std::mutex mutex;
  return std::unique_lock<std::mutex>(mutex);
}
void WatcherServer::isolated_dump(std::vector<ChipId> &) {}

namespace inspector {

void add_config_callback(ConfigCallback) {}

} // namespace inspector

} // namespace tt::tt_metal

namespace tt {

void ClearNocData(tt_metal::MetalEnvImpl &, ChipId) {}

} // namespace tt

namespace tracy {

void SetThreadName(const char *) {}

} // namespace tracy
