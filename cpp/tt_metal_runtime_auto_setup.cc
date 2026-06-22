#include "cpp/tt_metal_runtime_root.h"

#include <cstdlib>

#include "tracy/TracyC.h"

namespace {

bool IsEnabledEnv(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool ShouldStartTracyProfiler() {
  return IsEnabledEnv("TTNN_OP_PROFILER") ||
         IsEnabledEnv("TT_METAL_DEVICE_PROFILER") ||
         IsEnabledEnv("TT_METAL_PROFILER");
}

struct TtMetalRuntimeAutoSetup {
  TtMetalRuntimeAutoSetup() {
    EnsureTtMetalRuntimeReady();
#if defined(TRACY_ENABLE) && defined(TRACY_MANUAL_LIFETIME)
    if (ShouldStartTracyProfiler() && !TracyCIsStarted) {
      ___tracy_startup_profiler();
    }
#endif
  }

  ~TtMetalRuntimeAutoSetup() {
#if defined(TRACY_ENABLE) && defined(TRACY_MANUAL_LIFETIME)
    if (TracyCIsStarted) {
      ___tracy_shutdown_profiler();
    }
#endif
  }
};

TtMetalRuntimeAutoSetup runtime_auto_setup;

}  // namespace
