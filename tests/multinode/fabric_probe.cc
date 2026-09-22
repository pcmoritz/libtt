#include "cpp/key_value_context.h"
#include "cpp/tt_metal_runtime_root.h"
#include "tests/tt_metal/multihost/fabric_tests/mesh_socket_test_context.hpp"
#include "tt_metal/fabric/physical_system_discovery.hpp"
#include "tt_metal/llrt/tt_target_device.hpp"
#include <umd/device/cluster.hpp>
#include <yaml-cpp/yaml.h>

#include <fstream>
#include <iostream>

// Test-only bridge: Python supplies JAX's existing coordination service.
using Put = int (*)(const char *, const char *, std::size_t);
using Get = const char *(*)(const char *, std::size_t *);

extern "C" int libtt_discover(int rank, int size, Put put, Get get,
                              const char *output_path) {
  try {
    auto context = libtt::MakeKeyValueContext(
        {.put =
             [=](std::string_view key, std::string_view value) {
               if (put(std::string(key).c_str(), value.data(), value.size()))
                 throw std::runtime_error("JAX coordination put failed");
             },
         .get =
             [=](std::string_view key) {
               std::size_t length = 0;
               const char *value = get(std::string(key).c_str(), &length);
               if (!value)
                 throw std::runtime_error("JAX coordination get failed");
               return std::string(value, length);
             }},
        rank, size);
    auto cluster = tt::umd::Cluster::create_cluster_descriptor();
    auto descriptor = tt::tt_metal::run_physical_system_discovery(
        *cluster, context, tt::TargetDevice::Silicon);
    std::ofstream output(output_path);
    output << descriptor.generate_yaml_node();
    if (!output)
      throw std::runtime_error("Could not write physical system descriptor");
    std::cout << "Discovered " << descriptor.get_all_hostnames().size()
              << " hosts and " << descriptor.get_asic_descriptors().size()
              << " chips" << std::endl;
    if (const char *test_path = std::getenv("LIBTT_FABRIC_TEST_CONFIG")) {
      EnsureTtMetalRuntimeReady();
      tt::tt_metal::distributed::multihost::DistributedContext::
          set_current_world(context);
      auto config =
          tt::tt_fabric::mesh_socket_tests::MeshSocketYamlParser::parse_file(
              test_path);
      tt::tt_fabric::mesh_socket_tests::MeshSocketTestContext test(config);
      test.initialize();
      test.run_all_tests();
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}
