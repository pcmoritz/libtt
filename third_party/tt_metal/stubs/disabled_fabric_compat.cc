// SPDX-License-Identifier: Apache-2.0
//
// Compatibility definitions for the minimal libtt tt-metal Bazel overlay.
//
// The overlay builds tt-metal/TTNN tensor paths without the optional fabric
// control-plane, inspector RPC, DPRINT, watcher, or Tracy client runtimes.
// Some upstream object files still reference their public ABI even when those
// features are disabled. Keep those references local and explicit here instead
// of linking the full optional runtime graph into libtt.so.

#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/experimental/fabric/physical_system_descriptor.hpp>
#include <tt-metalium/experimental/fabric/topology_mapper.hpp>
#include <tt-metalium/experimental/inspector_config.hpp>
#include <tt-metalium/program.hpp>
#include <tt_metal/fabric/fabric_builder_context.hpp>
#include <tt_metal/fabric/fabric_context.hpp>
#include <tt_metal/fabric/fabric_init.hpp>
#include <tt_metal/fabric/fabric_tensix_builder.hpp>
#include <tt_metal/impl/debug/dprint_server.hpp>
#include <tt_metal/impl/debug/noc_logging.hpp>
#include <tt_metal/impl/debug/watcher_server.hpp>
#include <tt_metal/impl/debug/inspector/inspector.hpp>

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tracy {

#ifndef TRACY_ENABLE
void SetThreadName(const char*) {}
#endif

}  // namespace tracy

namespace tt::tt_metal::inspector {

void add_config_callback(ConfigCallback) {}

}  // namespace tt::tt_metal::inspector

namespace tt::tt_metal {

PhysicalSystemDescriptor::~PhysicalSystemDescriptor() = default;

class DPrintServer::Impl {};
class WatcherServer::Impl {};

DPrintServer::DPrintServer(MetalContext*, MetalEnv&, uint8_t, const DispatchCoreConfig&) {}
DPrintServer::~DPrintServer() = default;
void DPrintServer::set_mute(bool) {}
void DPrintServer::await() {}
void DPrintServer::attach_devices() {}
void DPrintServer::detach_devices() {}
void DPrintServer::clear_log_file() {}
bool DPrintServer::reads_dispatch_cores(ChipId) { return false; }
std::vector<umd::CoreDescriptor> DPrintServer::get_print_cores(ChipId) const { return {}; }
std::vector<DPrintBufferInfo> DPrintServer::get_core_buffers(ChipId, const umd::CoreDescriptor&) const { return {}; }
bool DPrintServer::hang_detected() { return false; }

WatcherServer::WatcherServer(MetalEnv&) {}
WatcherServer::~WatcherServer() = default;
void WatcherServer::init_devices() {}
void WatcherServer::attach_devices() {}
void WatcherServer::detach_devices() {}
void WatcherServer::clear_log() {}
std::string WatcherServer::log_file_name() { return {}; }
int WatcherServer::register_kernel(const std::string&) { return 0; }
void WatcherServer::register_kernel_elf_paths(int, std::vector<std::string>&) {}
bool WatcherServer::killed_due_to_error() { return false; }
void WatcherServer::set_killed_due_to_error_flag(bool) {}
std::string WatcherServer::exception_message() { return {}; }
void WatcherServer::set_exception_message(const std::string&) {}
int WatcherServer::dump_count() { return 0; }
std::unique_lock<std::mutex> WatcherServer::get_lock() {
    static std::mutex mutex;
    return std::unique_lock<std::mutex>(mutex);
}
void WatcherServer::isolated_dump(std::vector<ChipId>&) {}

}  // namespace tt::tt_metal

namespace tt {

void ClearNocData(tt_metal::MetalEnvImpl&, ChipId) {}

}  // namespace tt

namespace tt::tt_fabric {
namespace {

[[noreturn]] void ThrowFabricDisabled() {
    throw std::logic_error("tt-metal fabric is disabled in the libtt Bazel overlay");
}

const std::shared_ptr<tt_metal::distributed::multihost::DistributedContext>& WorldContext() {
    static std::shared_ptr<tt_metal::distributed::multihost::DistributedContext> context =
        tt_metal::distributed::multihost::DistributedContext::get_current_world();
    return context;
}

}  // namespace

MeshGraphDescriptor::~MeshGraphDescriptor() = default;
FabricContext::~FabricContext() = default;

ControlPlane::ControlPlane(
    const ::tt::Cluster& cluster,
    const ::tt::llrt::RunTimeOptions& rtoptions,
    const ::tt::tt_metal::Hal& hal,
    const tt_metal::distributed::multihost::DistributedContext& distributed_context,
    FabricConfig fabric_config,
    FabricReliabilityMode fabric_reliability_mode,
    FabricTensixConfig fabric_tensix_config,
    FabricUDMMode fabric_udm_mode,
    FabricRouterConfig fabric_router_config,
    FabricManagerMode fabric_manager) :
    cluster_(cluster),
    rtoptions_(rtoptions),
    hal_(hal),
    distributed_context_(distributed_context),
    fabric_config_(fabric_config),
    fabric_reliability_mode_(fabric_reliability_mode),
    fabric_tensix_config_(fabric_tensix_config),
    fabric_udm_mode_(fabric_udm_mode),
    fabric_router_config_(fabric_router_config),
    fabric_manager_(fabric_manager) {}

ControlPlane::ControlPlane(
    const ::tt::Cluster& cluster,
    const ::tt::llrt::RunTimeOptions& rtoptions,
    const ::tt::tt_metal::Hal& hal,
    const tt_metal::distributed::multihost::DistributedContext& distributed_context,
    const std::string&,
    FabricConfig fabric_config,
    FabricReliabilityMode fabric_reliability_mode,
    FabricTensixConfig fabric_tensix_config,
    FabricUDMMode fabric_udm_mode,
    FabricRouterConfig fabric_router_config,
    FabricManagerMode fabric_manager) :
    ControlPlane(
        cluster,
        rtoptions,
        hal,
        distributed_context,
        fabric_config,
        fabric_reliability_mode,
        fabric_tensix_config,
        fabric_udm_mode,
        fabric_router_config,
        fabric_manager) {}

ControlPlane::ControlPlane(
    const ::tt::Cluster& cluster,
    const ::tt::llrt::RunTimeOptions& rtoptions,
    const ::tt::tt_metal::Hal& hal,
    const tt_metal::distributed::multihost::DistributedContext& distributed_context,
    const std::string&,
    const std::map<FabricNodeId, ChipId>&,
    FabricConfig fabric_config,
    FabricReliabilityMode fabric_reliability_mode,
    FabricTensixConfig fabric_tensix_config,
    FabricUDMMode fabric_udm_mode,
    FabricRouterConfig fabric_router_config,
    FabricManagerMode fabric_manager) :
    ControlPlane(
        cluster,
        rtoptions,
        hal,
        distributed_context,
        fabric_config,
        fabric_reliability_mode,
        fabric_tensix_config,
        fabric_udm_mode,
        fabric_router_config,
        fabric_manager) {}

ControlPlane::~ControlPlane() = default;
void ControlPlane::configure_routing_tables_for_fabric_ethernet_channels() {}
void ControlPlane::write_routing_tables_to_all_chips() const {}
FabricNodeId ControlPlane::get_fabric_node_id_from_physical_chip_id(ChipId physical_chip_id) const {
    return FabricNodeId(MeshId{0}, static_cast<std::uint32_t>(physical_chip_id));
}
tt_metal::distributed::MeshShape ControlPlane::get_physical_mesh_shape(MeshId, MeshScope) const {
    return tt_metal::distributed::MeshShape(1, 1);
}
const std::shared_ptr<tt_metal::distributed::multihost::DistributedContext>& ControlPlane::get_distributed_context(
    MeshId) const {
    return WorldContext();
}
const std::shared_ptr<tt_metal::distributed::multihost::DistributedContext>& ControlPlane::get_host_local_context()
    const {
    return WorldContext();
}
std::vector<std::pair<FabricNodeId, chan_id_t>> ControlPlane::get_fabric_route(
    FabricNodeId, FabricNodeId, chan_id_t) const {
    return {};
}
std::optional<RoutingDirection> ControlPlane::get_forwarding_direction(FabricNodeId, FabricNodeId) const {
    return std::nullopt;
}
std::vector<chan_id_t> ControlPlane::get_active_fabric_eth_channels_in_direction(
    FabricNodeId, RoutingDirection) const {
    return {};
}
std::set<std::pair<chan_id_t, eth_chan_directions>> ControlPlane::get_active_fabric_eth_channels(
    FabricNodeId) const {
    return {};
}
eth_chan_directions ControlPlane::routing_direction_to_eth_direction(RoutingDirection) const {
    return eth_chan_directions::COUNT;
}
std::unordered_set<CoreCoord> ControlPlane::get_active_ethernet_cores(ChipId, bool) const { return {}; }
std::unordered_set<CoreCoord> ControlPlane::get_inactive_ethernet_cores(ChipId) const { return {}; }
std::map<std::string, std::string> ControlPlane::get_fabric_kernel_defines() const { return {}; }
void ControlPlane::clear_fabric_context() { fabric_context_.reset(); }
void ControlPlane::initialize_fabric_tensix_datamover_config() {}
const MeshGraph& ControlPlane::get_mesh_graph() const { ThrowFabricDisabled(); }
const std::unordered_map<tt_metal::distributed::multihost::Rank, std::pair<MeshId, MeshHostRankId>>&
ControlPlane::get_global_logical_bindings() const {
    static const std::unordered_map<tt_metal::distributed::multihost::Rank, std::pair<MeshId, MeshHostRankId>> bindings;
    return bindings;
}
FabricContext& ControlPlane::get_fabric_context() const { ThrowFabricDisabled(); }

std::filesystem::path MeshGraph::get_mesh_graph_descriptor_path_for_cluster_type(
    tt_metal::ClusterType, const std::string&, FabricType) {
    return {};
}
std::vector<SwitchId> MeshGraph::get_switch_ids() const { return {}; }
std::vector<MeshId> MeshGraph::get_mesh_ids() const { return {MeshId{0}}; }

std::unique_ptr<tt_metal::Program> create_and_compile_fabric_program(tt_metal::IDevice*) { return nullptr; }
void configure_fabric_cores(tt_metal::IDevice*) {}
void export_channel_trimming_capture(tt_metal::MetalEnvImpl&) {}
bool is_tt_fabric_config(FabricConfig) { return false; }
bool is_2d_fabric_config(FabricConfig) { return false; }
FabricType get_fabric_type(FabricConfig, bool) { return FabricType::MESH; }
size_t get_tt_fabric_max_payload_size_bytes() { return 0; }
FabricNodeId get_fabric_node_id_from_physical_chip_id(ChipId physical_chip_id) {
    return FabricNodeId(MeshId{0}, static_cast<std::uint32_t>(physical_chip_id));
}
std::vector<std::uint32_t> get_forwarding_link_indices(const FabricNodeId&, const FabricNodeId&) { return {}; }

FabricBuilderContext& FabricContext::get_builder_context() { ThrowFabricDisabled(); }
const FabricBuilderContext& FabricContext::get_builder_context() const { ThrowFabricDisabled(); }

std::uint32_t FabricBuilderContext::get_num_fabric_initialized_routers(ChipId) const { return 0; }
chan_id_t FabricBuilderContext::get_fabric_master_router_chan(ChipId) const { return 0; }
std::pair<std::uint32_t, std::uint32_t> FabricBuilderContext::get_fabric_router_sync_address_and_status() const {
    return {0, 0};
}
std::optional<std::pair<std::uint32_t, EDMStatus>> FabricBuilderContext::get_fabric_router_ready_address_and_signal()
    const {
    return std::nullopt;
}
std::pair<std::uint32_t, std::uint32_t> FabricBuilderContext::get_fabric_router_termination_address_and_signal() const {
    return {0, 0};
}
FabricTensixDatamoverConfig& FabricBuilderContext::get_tensix_config() const { ThrowFabricDisabled(); }

FabricTensixCoreType FabricTensixDatamoverConfig::get_core_id_for_channel(ChipId, std::uint32_t) const {
    return FabricTensixCoreType::MUX;
}
CoreCoord FabricTensixDatamoverConfig::get_core_for_channel(ChipId, std::uint32_t) const { return CoreCoord{0, 0}; }
std::pair<std::uint32_t, std::uint32_t> FabricTensixDatamoverConfig::get_termination_address_and_signal(
    FabricTensixCoreType) const {
    return {0, 0};
}

FabricMuxConfig::FabricMuxConfig(
    std::uint8_t, std::uint8_t, std::uint8_t, std::uint8_t, size_t, size_t base_l1_address, CoreType) :
    memory_map_end_address_(base_l1_address) {}
std::vector<std::uint32_t> FabricMuxConfig::get_fabric_mux_compile_time_args_for_relay_mux() const { return {}; }
size_t FabricMuxConfig::get_memory_map_end_address() const { return memory_map_end_address_; }
std::uint8_t FabricMuxConfig::get_num_buffers(FabricMuxChannelType) const { return 0; }
size_t FabricMuxConfig::get_buffer_size_bytes(FabricMuxChannelType) const { return 0; }
size_t FabricMuxConfig::get_channel_base_address(FabricMuxChannelType, std::uint8_t) const { return 0; }
size_t FabricMuxConfig::get_connection_info_address(FabricMuxChannelType, std::uint8_t) const { return 0; }
size_t FabricMuxConfig::get_connection_handshake_address(FabricMuxChannelType, std::uint8_t) const { return 0; }
size_t FabricMuxConfig::get_flow_control_address(FabricMuxChannelType, std::uint8_t) const { return 0; }
size_t FabricMuxConfig::get_buffer_index_address(FabricMuxChannelType, std::uint8_t) const { return 0; }
size_t FabricMuxConfig::get_status_address() const { return 0; }
size_t FabricMuxConfig::get_termination_signal_address() const { return 0; }
size_t FabricMuxConfig::get_channel_credits_stream_id(FabricMuxChannelType, std::uint8_t) const { return 0; }

template <>
std::vector<std::uint32_t> FabricMuxConfig::get_fabric_mux_run_time_args<tt_metal::Program>(
    const FabricNodeId&, const FabricNodeId&, std::uint32_t, tt_metal::Program&, const CoreCoord&) const {
    return {};
}

}  // namespace tt::tt_fabric
