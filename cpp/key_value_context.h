#pragma once

#include <tt-metalium/distributed_context.hpp>

#include <functional>
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

} // namespace libtt
