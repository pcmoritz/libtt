#include "cpp/key_value_context.h"

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace libtt {
namespace {

void SetDefault(const char *name, const std::string &value) {
  setenv(name, value.c_str(), /*overwrite=*/0);
}

// One chip per process, in a line of `count` chips.
std::string WriteMeshDescriptor(int count) {
  const std::string n = std::to_string(count);
  const auto path = std::filesystem::temp_directory_path() /
                    ("libtt-libtpu-mesh-1x" + n + ".textproto");
  const auto temp = path.string() + "." + std::to_string(getpid());
  {
    std::ofstream out(temp);
    out << "mesh_descriptors {\n"
           "  name: \"M0\"\n"
           "  arch: BLACKHOLE\n"
           "  device_topology { dims: [1, " << n << "] dim_types: [LINE, LINE] }\n"
           "  host_topology { dims: [1, " << n << "] }\n"
           "  channels { count: 2 policy: RELAXED }\n"
           "}\n"
           "top_level_instance { mesh { mesh_descriptor: \"M0\" mesh_id: 0 } }\n";
    if (!out)
      throw std::runtime_error("Failed to write " + temp);
  }
  std::filesystem::rename(temp, path);
  return path.string();
}

} // namespace

std::optional<LibtpuProcesses> ConfigureLibtpuProcesses() {
  const char *addresses = std::getenv("TPU_PROCESS_ADDRESSES");
  const char *task = std::getenv("CLOUD_TPU_TASK_ID");
  if (addresses == nullptr || task == nullptr)
    return std::nullopt;
  std::vector<std::string> processes;
  std::stringstream list(addresses);
  for (std::string address; std::getline(list, address, ',');)
    processes.push_back(address);
  const int count = static_cast<int>(processes.size());
  const int index = std::stoi(task);
  if (index < 0 || index >= count)
    throw std::invalid_argument("CLOUD_TPU_TASK_ID is out of range");

  // torch_tpu's process groups expect one chip per process. TT-Metal needs a
  // mesh descriptor for a subset of a board's chips, even for one process.
  if (const char *chip = std::getenv("TPU_VISIBLE_CHIPS"))
    SetDefault("TT_VISIBLE_DEVICES", chip);
  if (std::getenv("TT_MESH_GRAPH_DESC_PATH") == nullptr) {
    setenv("TT_MESH_GRAPH_DESC_PATH", WriteMeshDescriptor(count).c_str(), 1);
    SetDefault("TT_MESH_ID", "0");
    SetDefault("TT_MESH_HOST_RANK", std::to_string(index));
  }
  if (count == 1)
    return std::nullopt;
  return LibtpuProcesses{index, count,
                         MakeTcpKeyValueStore(processes[0], index == 0)};
}

} // namespace libtt
