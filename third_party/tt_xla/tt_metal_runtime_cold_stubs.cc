#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph_descriptor.hpp>
#include <tt-metalium/experimental/fabric/physical_system_descriptor.hpp>
#include <tt-metalium/experimental/fabric/routing_table_generator.hpp>
#include <tt-metalium/experimental/fabric/topology_mapper.hpp>
#include <tt-metalium/experimental/inspector_config.hpp>
#include <tt-metalium/internal/cluster.hpp>

#include "tt_metal/fabric/channel_trimming_export.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "tt_metal/fabric/fabric_host_utils.hpp"
#include "tt_metal/fabric/fabric_init.hpp"
#include "tt_metal/fabric/fabric_tensix_builder.hpp"
#include "tt_metal/impl/debug/noc_logging.hpp"
#include "tt_metal/impl/debug/dprint_server.hpp"
#include "tt_metal/impl/debug/watcher_server.hpp"

#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

[[noreturn]] void unsupported(const char *name) {
  throw std::runtime_error(std::string(name) +
                           " is not linked in this libtt build");
}

} // namespace

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

PhysicalSystemDescriptor::~PhysicalSystemDescriptor() = default;

} // namespace tt::tt_metal

namespace tt::tt_metal::internal {

AsicID get_chip_unique_id_from_fabric_node_id(uint32_t mesh_id,
                                              uint32_t chip_id) {
  return AsicID{(static_cast<uint64_t>(mesh_id) << 32) | chip_id};
}

} // namespace tt::tt_metal::internal

namespace tt {

void ClearNocData(tt_metal::MetalEnvImpl &, ChipId) {}

} // namespace tt

namespace tt::tt_fabric {

namespace {

tt::tt_metal::distributed::multihost::ContextPtr current_world_context() {
  return tt::tt_metal::distributed::multihost::DistributedContext::
      get_current_world();
}

} // namespace

ControlPlane::ControlPlane(
    const ::tt::Cluster &cluster, const ::tt::llrt::RunTimeOptions &rtoptions,
    const ::tt::tt_metal::Hal &hal,
    const tt_metal::distributed::multihost::DistributedContext
        &distributed_context,
    FabricConfig fabric_config, FabricReliabilityMode fabric_reliability_mode,
    FabricTensixConfig fabric_tensix_config, FabricUDMMode fabric_udm_mode,
    FabricRouterConfig fabric_router_config, FabricManagerMode fabric_manager)
    : cluster_(cluster), rtoptions_(rtoptions), hal_(hal),
      distributed_context_(distributed_context), fabric_config_(fabric_config),
      fabric_reliability_mode_(fabric_reliability_mode),
      fabric_tensix_config_(fabric_tensix_config),
      fabric_udm_mode_(fabric_udm_mode),
      fabric_router_config_(fabric_router_config),
      fabric_manager_(fabric_manager),
      host_local_context_(current_world_context()),
      local_mesh_binding_({std::vector<MeshId>{MeshId{0}}, MeshHostRankId{0}}) {}

ControlPlane::ControlPlane(
    const ::tt::Cluster &cluster, const ::tt::llrt::RunTimeOptions &rtoptions,
    const ::tt::tt_metal::Hal &hal,
    const tt_metal::distributed::multihost::DistributedContext
        &distributed_context,
    const std::string &, FabricConfig fabric_config,
    FabricReliabilityMode fabric_reliability_mode,
    FabricTensixConfig fabric_tensix_config, FabricUDMMode fabric_udm_mode,
    FabricRouterConfig fabric_router_config, FabricManagerMode fabric_manager)
    : ControlPlane(cluster, rtoptions, hal, distributed_context, fabric_config,
                   fabric_reliability_mode, fabric_tensix_config,
                   fabric_udm_mode, fabric_router_config, fabric_manager) {}

ControlPlane::ControlPlane(
    const ::tt::Cluster &cluster, const ::tt::llrt::RunTimeOptions &rtoptions,
    const ::tt::tt_metal::Hal &hal,
    const tt_metal::distributed::multihost::DistributedContext
        &distributed_context,
    const std::string &, const std::map<FabricNodeId, ChipId> &,
    FabricConfig fabric_config, FabricReliabilityMode fabric_reliability_mode,
    FabricTensixConfig fabric_tensix_config, FabricUDMMode fabric_udm_mode,
    FabricRouterConfig fabric_router_config, FabricManagerMode fabric_manager)
    : ControlPlane(cluster, rtoptions, hal, distributed_context, fabric_config,
                   fabric_reliability_mode, fabric_tensix_config,
                   fabric_udm_mode, fabric_router_config, fabric_manager) {}

ControlPlane::~ControlPlane() = default;

void ControlPlane::configure_routing_tables_for_fabric_ethernet_channels() {}
void ControlPlane::write_routing_tables_to_all_chips() const {}
void ControlPlane::clear_fabric_context() { fabric_context_.reset(); }
void ControlPlane::initialize_fabric_tensix_datamover_config() {}

FabricNodeId ControlPlane::get_fabric_node_id_from_physical_chip_id(
    ChipId physical_chip_id) const {
  return FabricNodeId(MeshId{0}, static_cast<std::uint32_t>(physical_chip_id));
}

std::vector<std::pair<FabricNodeId, chan_id_t>>
ControlPlane::get_fabric_route(FabricNodeId, FabricNodeId, chan_id_t) const {
  return {};
}

std::optional<RoutingDirection>
ControlPlane::get_forwarding_direction(FabricNodeId, FabricNodeId) const {
  return std::nullopt;
}

std::unordered_set<CoreCoord>
ControlPlane::get_active_ethernet_cores(ChipId, bool) const {
  return {};
}

std::unordered_set<CoreCoord>
ControlPlane::get_inactive_ethernet_cores(ChipId) const {
  return {};
}

std::set<std::pair<chan_id_t, eth_chan_directions>>
ControlPlane::get_active_fabric_eth_channels(FabricNodeId) const {
  return {};
}

std::vector<chan_id_t>
ControlPlane::get_active_fabric_eth_channels_in_direction(
    FabricNodeId, RoutingDirection) const {
  return {};
}

eth_chan_directions
ControlPlane::routing_direction_to_eth_direction(RoutingDirection) const {
  return eth_chan_directions::EAST;
}

std::map<std::string, std::string>
ControlPlane::get_fabric_kernel_defines() const {
  return {};
}

const std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext>
    &ControlPlane::get_host_local_context() const {
  return host_local_context_;
}

const std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext>
    &ControlPlane::get_distributed_context(MeshId) const {
  return host_local_context_;
}

const std::unordered_map<
    tt_metal::distributed::multihost::Rank, std::pair<MeshId, MeshHostRankId>>
    &ControlPlane::get_global_logical_bindings() const {
  return global_logical_bindings_;
}

const MeshGraph &ControlPlane::get_mesh_graph() const {
  unsupported("tt::tt_fabric::ControlPlane::get_mesh_graph");
}

tt::tt_metal::distributed::MeshShape
ControlPlane::get_physical_mesh_shape(MeshId, MeshScope) const {
  return tt::tt_metal::distributed::MeshShape(1, 1);
}

FabricContext &ControlPlane::get_fabric_context() const {
  unsupported("tt::tt_fabric::ControlPlane::get_fabric_context");
}

FabricContext::~FabricContext() = default;

FabricBuilderContext &FabricContext::get_builder_context() {
  unsupported("tt::tt_fabric::FabricContext::get_builder_context");
}

const FabricBuilderContext &FabricContext::get_builder_context() const {
  unsupported("tt::tt_fabric::FabricContext::get_builder_context");
}

FabricTensixDatamoverConfig &FabricBuilderContext::get_tensix_config() const {
  unsupported("tt::tt_fabric::FabricBuilderContext::get_tensix_config");
}

chan_id_t FabricBuilderContext::get_fabric_master_router_chan(ChipId) const {
  unsupported(
      "tt::tt_fabric::FabricBuilderContext::get_fabric_master_router_chan");
}

uint32_t
FabricBuilderContext::get_num_fabric_initialized_routers(ChipId) const {
  return 0;
}

std::pair<uint32_t, uint32_t>
FabricBuilderContext::get_fabric_router_sync_address_and_status() const {
  return {0, 0};
}

std::optional<std::pair<uint32_t, EDMStatus>>
FabricBuilderContext::get_fabric_router_ready_address_and_signal() const {
  return std::nullopt;
}

std::pair<uint32_t, uint32_t>
FabricBuilderContext::get_fabric_router_termination_address_and_signal() const {
  return {0, 0};
}

CoreCoord FabricTensixDatamoverConfig::get_core_for_channel(ChipId,
                                                            uint32_t) const {
  unsupported("tt::tt_fabric::FabricTensixDatamoverConfig::get_core_for_channel");
}

FabricTensixCoreType
FabricTensixDatamoverConfig::get_core_id_for_channel(ChipId,
                                                     uint32_t) const {
  unsupported(
      "tt::tt_fabric::FabricTensixDatamoverConfig::get_core_id_for_channel");
}

std::pair<uint32_t, uint32_t>
FabricTensixDatamoverConfig::get_termination_address_and_signal(
    FabricTensixCoreType) const {
  return {0, 0};
}

FabricMuxConfig::MemoryRegion::MemoryRegion(size_t base, size_t unit_size,
                                             size_t count)
    : base_address(base), unit_size(unit_size), num_units(count) {}

size_t FabricMuxConfig::MemoryRegion::get_address(size_t offset) const {
  return base_address + offset;
}
size_t FabricMuxConfig::MemoryRegion::get_end_address() const {
  return base_address + (unit_size * num_units);
}
size_t FabricMuxConfig::MemoryRegion::get_total_size() const {
  return unit_size * num_units;
}

FabricMuxConfig::FabricMuxConfig(uint8_t num_full_size_channels,
                                 uint8_t num_header_only_channels,
                                 uint8_t num_buffers_full_size_channel,
                                 uint8_t num_buffers_header_only_channel,
                                 size_t buffer_size_bytes_full_size_channel,
                                 size_t base_l1_address, CoreType core_type)
    : core_type_(core_type), num_full_size_channels_(num_full_size_channels),
      num_header_only_channels_(num_header_only_channels),
      num_buffers_full_size_channel_(num_buffers_full_size_channel),
      num_buffers_header_only_channel_(num_buffers_header_only_channel),
      buffer_size_bytes_full_size_channel_(buffer_size_bytes_full_size_channel),
      buffer_size_bytes_header_only_channel_(0),
      memory_map_end_address_(base_l1_address) {}

std::vector<uint32_t> FabricMuxConfig::get_fabric_mux_compile_time_args()
    const {
  return {};
}
std::vector<uint32_t>
FabricMuxConfig::get_fabric_mux_compile_time_args_for_relay_mux() const {
  return {};
}
uint8_t FabricMuxConfig::get_num_channels(FabricMuxChannelType type) const {
  return type == FabricMuxChannelType::FULL_SIZE_CHANNEL
             ? num_full_size_channels_
             : num_header_only_channels_;
}
uint8_t FabricMuxConfig::get_num_buffers(FabricMuxChannelType type) const {
  return type == FabricMuxChannelType::FULL_SIZE_CHANNEL
             ? num_buffers_full_size_channel_
             : num_buffers_header_only_channel_;
}
size_t FabricMuxConfig::get_buffer_size_bytes(FabricMuxChannelType type) const {
  return type == FabricMuxChannelType::FULL_SIZE_CHANNEL
             ? buffer_size_bytes_full_size_channel_
             : buffer_size_bytes_header_only_channel_;
}
size_t FabricMuxConfig::get_status_address() const { return 0; }
size_t FabricMuxConfig::get_termination_signal_address() const { return 0; }
size_t FabricMuxConfig::get_channel_credits_stream_id(FabricMuxChannelType,
                                                      uint8_t) const {
  return 0;
}
size_t FabricMuxConfig::get_channel_base_address(FabricMuxChannelType,
                                                 uint8_t) const {
  return 0;
}
size_t FabricMuxConfig::get_connection_info_address(FabricMuxChannelType,
                                                    uint8_t) const {
  return 0;
}
size_t FabricMuxConfig::get_connection_handshake_address(FabricMuxChannelType,
                                                         uint8_t) const {
  return 0;
}
size_t FabricMuxConfig::get_flow_control_address(FabricMuxChannelType,
                                                 uint8_t) const {
  return 0;
}
size_t FabricMuxConfig::get_buffer_index_address(FabricMuxChannelType,
                                                 uint8_t) const {
  return 0;
}
void FabricMuxConfig::set_num_full_size_channel_iters(size_t value) {
  num_full_size_channel_iters_ = value;
}
void FabricMuxConfig::set_num_iters_between_teardown_checks(size_t value) {
  num_iters_between_teardown_checks_ = value;
}
void FabricMuxConfig::set_wait_for_fabric_endpoint_ready(bool value) {
  wait_for_fabric_endpoint_ready_ = value;
}
void FabricMuxConfig::set_fabric_endpoint_channel_num_buffers(size_t value) {
  fabric_endpoint_channel_num_buffers_ = value;
}
void FabricMuxConfig::set_fabric_endpoint_status_address(size_t value) {
  fabric_endpoint_status_address_ = value;
}
size_t FabricMuxConfig::get_memory_map_end_address() const {
  return memory_map_end_address_;
}
std::vector<std::pair<size_t, size_t>>
FabricMuxConfig::get_memory_regions_to_clear() const {
  return {};
}

template <typename ProgramOrDescriptor>
std::vector<uint32_t> FabricMuxConfig::get_fabric_mux_run_time_args(
    const FabricNodeId &, const FabricNodeId &, uint32_t, ProgramOrDescriptor &,
    const CoreCoord &) const {
  unsupported("tt::tt_fabric::FabricMuxConfig::get_fabric_mux_run_time_args");
}

template std::vector<uint32_t>
FabricMuxConfig::get_fabric_mux_run_time_args<tt::tt_metal::Program>(
    const FabricNodeId &, const FabricNodeId &, uint32_t,
    tt::tt_metal::Program &, const CoreCoord &) const;
template std::vector<uint32_t>
FabricMuxConfig::get_fabric_mux_run_time_args<tt::tt_metal::ProgramDescriptor>(
    const FabricNodeId &, const FabricNodeId &, uint32_t,
    tt::tt_metal::ProgramDescriptor &, const CoreCoord &) const;

void SetFabricConfig(FabricConfig fabric_config,
                     FabricReliabilityMode reliability_mode,
                     std::optional<uint8_t> num_routing_planes,
                     FabricTensixConfig fabric_tensix_config,
                     FabricUDMMode fabric_udm_mode,
                     FabricManagerMode fabric_manager,
                     FabricRouterConfig router_config) {
  if (fabric_config != FabricConfig::DISABLED) {
    unsupported("tt::tt_fabric::SetFabricConfig");
  }
}

bool is_1d_fabric_config(FabricConfig fabric_config) {
  return fabric_config == FabricConfig::FABRIC_1D_NEIGHBOR_EXCHANGE ||
         fabric_config == FabricConfig::FABRIC_1D ||
         fabric_config == FabricConfig::FABRIC_1D_RING;
}

bool is_2d_fabric_config(FabricConfig fabric_config) {
  return fabric_config == FabricConfig::FABRIC_2D ||
         fabric_config == FabricConfig::FABRIC_2D_TORUS_X ||
         fabric_config == FabricConfig::FABRIC_2D_TORUS_Y ||
         fabric_config == FabricConfig::FABRIC_2D_TORUS_XY;
}

bool is_tt_fabric_config(FabricConfig fabric_config) {
  return is_1d_fabric_config(fabric_config) || is_2d_fabric_config(fabric_config);
}

FabricType get_fabric_type(FabricConfig fabric_config, bool) {
  switch (fabric_config) {
  case FabricConfig::FABRIC_2D_TORUS_X:
    return FabricType::TORUS_X;
  case FabricConfig::FABRIC_2D_TORUS_Y:
    return FabricType::TORUS_Y;
  case FabricConfig::FABRIC_2D_TORUS_XY:
    return FabricType::TORUS_XY;
  default:
    return FabricType::MESH;
  }
}

size_t get_tt_fabric_channel_buffer_size_bytes() { return 0; }
size_t get_tt_fabric_max_payload_size_bytes() { return 0; }

std::vector<eth_chan_directions>
get_neighbor_eth_directions(const FabricNodeId &, const FabricNodeId &) {
  return {};
}

std::vector<uint32_t> get_forwarding_link_indices(const FabricNodeId &,
                                                  const FabricNodeId &) {
  return {};
}

FabricNodeId get_fabric_node_id_from_physical_chip_id(ChipId physical_chip_id) {
  return FabricNodeId(MeshId{0}, static_cast<std::uint32_t>(physical_chip_id));
}

void configure_fabric_cores(tt::tt_metal::IDevice *) {}

std::unique_ptr<tt::tt_metal::Program>
create_and_compile_fabric_program(tt::tt_metal::IDevice *) {
  return nullptr;
}

void export_channel_trimming_capture(tt::tt_metal::MetalEnvImpl &) {}

template <typename ProgramOrDescriptor>
uint32_t append_routing_plane_connection_manager_rt_args(
    const FabricNodeId &, const std::vector<eth_chan_directions> &,
    const std::vector<uint32_t> &, ProgramOrDescriptor &, tt::tt_metal::KernelHandle &,
    const CoreCoord &, std::vector<uint32_t> &, FabricApiType, CoreType) {
  unsupported(
      "tt::tt_fabric::append_routing_plane_connection_manager_rt_args");
}

template uint32_t
append_routing_plane_connection_manager_rt_args<tt::tt_metal::ProgramDescriptor>(
    const FabricNodeId &, const std::vector<eth_chan_directions> &,
    const std::vector<uint32_t> &, tt::tt_metal::ProgramDescriptor &,
    tt::tt_metal::KernelHandle &, const CoreCoord &, std::vector<uint32_t> &,
    FabricApiType, CoreType);
template uint32_t
append_routing_plane_connection_manager_rt_args<tt::tt_metal::Program>(
    const FabricNodeId &, const std::vector<eth_chan_directions> &,
    const std::vector<uint32_t> &, tt::tt_metal::Program &,
    tt::tt_metal::KernelHandle &, const CoreCoord &, std::vector<uint32_t> &,
    FabricApiType, CoreType);

std::filesystem::path MeshGraph::get_mesh_graph_descriptor_path_for_cluster_type(
    tt::tt_metal::ClusterType, const std::string &, FabricType) {
  return {};
}

std::vector<MeshId> MeshGraph::get_mesh_ids() const { return {MeshId{0}}; }

std::vector<SwitchId> MeshGraph::get_switch_ids() const { return {}; }

MeshGraphDescriptor::~MeshGraphDescriptor() = default;

} // namespace tt::tt_fabric

namespace tracy {

void SetThreadName(const char *) {}

} // namespace tracy
