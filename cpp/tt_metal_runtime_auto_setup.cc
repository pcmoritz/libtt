#include "cpp/tt_metal_runtime_root.h"

namespace {

struct TtMetalRuntimeAutoSetup {
  TtMetalRuntimeAutoSetup() { EnsureTtMetalRuntimeReady(); }
};

TtMetalRuntimeAutoSetup runtime_auto_setup;

}  // namespace
