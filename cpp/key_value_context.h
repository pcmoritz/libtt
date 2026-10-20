#pragma once

#include <tt-metalium/distributed_context.hpp>

#include <functional>
#include <optional>
#include <string>
#include <string_view>

namespace libtt {

// Host coordination only. Tensor collectives remain on the device fabric.
struct KeyValueStore {
  std::function<void(std::string_view, std::string_view)> put;
  std::function<std::string(std::string_view)> get;
};

tt::tt_metal::distributed::multihost::ContextPtr
MakeKeyValueContext(KeyValueStore store, int rank, int size);

// A store for processes without a JAX coordinator: the process with `serve`
// listens on the port of `address` ("host:port"), and every process connects
// to it there.
KeyValueStore MakeTcpKeyValueStore(const std::string &address, bool serve);

// torch_tpu starts one process per chip and describes them through libtpu's
// environment rather than coordinator callbacks.
struct LibtpuProcesses {
  int index;
  int count;
  KeyValueStore store;
};

// Reads TPU_PROCESS_ADDRESSES (one host:port per process) and
// CLOUD_TPU_TASK_ID, and points TT-Metal at this process's chip, in a mesh of
// all processes' chips unless TT_MESH_GRAPH_DESC_PATH already describes one.
// Returns nullopt for a single process.
std::optional<LibtpuProcesses> ConfigureLibtpuProcesses();

} // namespace libtt
