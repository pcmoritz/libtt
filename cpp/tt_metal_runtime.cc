#include "cpp/tt_metal_runtime.h"

#include "libtt_embedded_runtime_assets.h"

#include <archive.h>
#include <archive_entry.h>

#include <tt_metal/llrt/rtoptions.hpp>

#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <map>
#include <mutex>
#include <optional>
#include <string>

namespace {

using tt::tt_metal::distributed::MeshDevice;

bool IsTtMetalRuntimeRoot(const std::filesystem::path& path) {
  return std::filesystem::is_directory(path / "tt_metal");
}

bool IsTtMetalRuntimeAssetRoot(const std::filesystem::path& path) {
  return std::filesystem::is_regular_file(
             path / "runtime/hw/toolchain/blackhole/firmware_brisc.ld") &&
         std::filesystem::is_regular_file(
             path / "runtime/hw/lib/blackhole/tmu-crt0.o");
}

bool IsSafeRelativePath(const std::filesystem::path& relative_path) {
  if (relative_path.empty() || relative_path.is_absolute()) {
    return false;
  }
  for (const std::filesystem::path& component : relative_path) {
    if (component == "..") {
      return false;
    }
  }
  return true;
}

struct ArchiveReadDeleter {
  void operator()(archive* value) const {
    if (value != nullptr) {
      archive_read_free(value);
    }
  }
};

bool ExtractEmbeddedRuntimeArchive(const std::filesystem::path& root) {
  namespace embedded = libtt::tt_metal_runtime_assets;
  if (embedded::kBlackholeRuntimeHwArchiveZstdSize == 0) {
    return false;
  }

  std::error_code error;
  std::filesystem::create_directories(root, error);
  if (error) {
    return false;
  }

  std::unique_ptr<archive, ArchiveReadDeleter> reader(archive_read_new());
  if (!reader) {
    return false;
  }
  if (archive_read_support_filter_zstd(reader.get()) != ARCHIVE_OK ||
      archive_read_support_format_tar(reader.get()) != ARCHIVE_OK ||
      archive_read_open_memory(reader.get(),
                               embedded::kBlackholeRuntimeHwArchiveZstd,
                               embedded::kBlackholeRuntimeHwArchiveZstdSize) !=
          ARCHIVE_OK) {
    return false;
  }

  archive_entry* entry = nullptr;
  while (true) {
    const int status = archive_read_next_header(reader.get(), &entry);
    if (status == ARCHIVE_EOF) {
      return true;
    }
    if (status != ARCHIVE_OK || entry == nullptr) {
      return false;
    }

    const char* pathname = archive_entry_pathname(entry);
    if (pathname == nullptr || pathname[0] == '\0') {
      return false;
    }
    const std::filesystem::path relative_path(pathname);
    if (!IsSafeRelativePath(relative_path)) {
      return false;
    }

    const mode_t file_type = archive_entry_filetype(entry);
    if (file_type != AE_IFDIR && file_type != AE_IFREG) {
      return false;
    }

    const std::filesystem::path output_path = root / relative_path;
    const std::string output_path_string = output_path.string();
    archive_entry_set_pathname(entry, output_path_string.c_str());
    if (archive_read_extract(reader.get(), entry,
                             ARCHIVE_EXTRACT_TIME | ARCHIVE_EXTRACT_PERM |
                                 ARCHIVE_EXTRACT_SECURE_SYMLINKS) != ARCHIVE_OK) {
      return false;
    }
  }
}

std::optional<std::filesystem::path> MaterializeEmbeddedRuntimeAssets() {
  namespace embedded = libtt::tt_metal_runtime_assets;
  if (embedded::kBlackholeRuntimeHwArchiveZstdSize == 0) {
    return std::nullopt;
  }

  std::error_code error;
  const std::filesystem::path temp_root =
      std::filesystem::temp_directory_path(error);
  if (error) {
    return std::nullopt;
  }
  const std::filesystem::path root =
      temp_root / ("libtt-tt-metal-runtime-assets-" +
                   std::string(embedded::kBlackholeRuntimeHwArchiveFingerprint));

  if (!ExtractEmbeddedRuntimeArchive(root) || !IsTtMetalRuntimeAssetRoot(root)) {
    return std::nullopt;
  }
  return root;
}

bool IsSfpiRoot(const std::filesystem::path& path) {
  return std::filesystem::is_regular_file(
             path / "compiler/bin/riscv-tt-elf-g++") &&
         std::filesystem::is_regular_file(path / "include/sfpi.h");
}

std::optional<std::filesystem::path> FindSfpiRootInExternal(
    const std::filesystem::path& external_root) {
  std::error_code error;
  if (!std::filesystem::is_directory(external_root, error)) {
    return std::nullopt;
  }

  for (const char* repo_name : {"+http_archive+sfpi", "sfpi"}) {
    const std::filesystem::path candidate = external_root / repo_name;
    if (IsSfpiRoot(candidate)) {
      return candidate;
    }
  }

  for (std::filesystem::directory_iterator it(external_root, error), end;
       !error && it != end; it.increment(error)) {
    if (!std::filesystem::is_directory(it->path(), error)) {
      continue;
    }
    if (it->path().filename().string().find("sfpi") != std::string::npos &&
        IsSfpiRoot(it->path())) {
      return it->path();
    }
  }
  return std::nullopt;
}

std::optional<std::filesystem::path> FindTtMetalRuntimeRootFrom(
    std::filesystem::path path) {
  std::error_code error;
  path = std::filesystem::weakly_canonical(path, error);
  if (error) {
    path.clear();
  }
  if (path.empty()) {
    return std::nullopt;
  }
  if (!std::filesystem::is_directory(path)) {
    path = path.parent_path();
  }

  const std::filesystem::path workspace_bazel_link =
      path / ("bazel-" + path.filename().string()) / "external" /
      "+http_archive+tt_metal";
  if (IsTtMetalRuntimeRoot(workspace_bazel_link)) {
    return workspace_bazel_link;
  }

  for (std::filesystem::path current = path; !current.empty();
       current = current.parent_path()) {
    if (IsTtMetalRuntimeRoot(current)) {
      return current;
    }
    const std::filesystem::path external_root =
        current / "external" / "+http_archive+tt_metal";
    if (IsTtMetalRuntimeRoot(external_root)) {
      return external_root;
    }
    const std::filesystem::path bazel_link =
        current / ("bazel-" + current.filename().string()) / "external" /
        "+http_archive+tt_metal";
    if (IsTtMetalRuntimeRoot(bazel_link)) {
      return bazel_link;
    }
    if (current == current.root_path()) {
      break;
    }
  }
  return std::nullopt;
}

std::optional<std::filesystem::path> FindSfpiRootFrom(std::filesystem::path path) {
  std::error_code error;
  path = std::filesystem::weakly_canonical(path, error);
  if (error) {
    path.clear();
  }
  if (path.empty()) {
    return std::nullopt;
  }
  if (!std::filesystem::is_directory(path)) {
    path = path.parent_path();
  }

  const std::filesystem::path workspace_external_link =
      path / ("bazel-" + path.filename().string()) / "external";
  if (std::optional<std::filesystem::path> root =
          FindSfpiRootInExternal(workspace_external_link)) {
    return root;
  }

  for (std::filesystem::path current = path; !current.empty();
       current = current.parent_path()) {
    if (IsSfpiRoot(current)) {
      return current;
    }
    if (std::optional<std::filesystem::path> root =
            FindSfpiRootInExternal(current / "external")) {
      return root;
    }
    const std::filesystem::path bazel_external_link =
        current / ("bazel-" + current.filename().string()) / "external";
    if (std::optional<std::filesystem::path> root =
            FindSfpiRootInExternal(bazel_external_link)) {
      return root;
    }
    if (current == current.root_path()) {
      break;
    }
  }
  return std::nullopt;
}

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

void EnsureTtMetalRuntimeReady() {
  static std::once_flag once;
  std::call_once(once, [] {
    const bool has_runtime_root_override =
        std::getenv("TT_METAL_RUNTIME_ROOT") != nullptr;
    const bool has_runtime_asset_root_override =
        std::getenv("TT_METAL_RUNTIME_ASSET_ROOT") != nullptr;
    const bool has_sfpi_root_override =
        std::getenv("TT_METAL_SFPI_ROOT") != nullptr;
    bool configured_runtime_root = has_runtime_root_override;

    Dl_info info;
    if (dladdr(reinterpret_cast<void*>(&EnsureTtMetalRuntimeReady), &info) != 0 &&
        info.dli_fname != nullptr) {
      if (!has_runtime_root_override) {
        if (std::optional<std::filesystem::path> root =
                FindTtMetalRuntimeRootFrom(info.dli_fname)) {
          tt::llrt::RunTimeOptions::set_root_dir(root->string());
          configured_runtime_root = true;
        }
      }
      if (!has_sfpi_root_override) {
        if (std::optional<std::filesystem::path> sfpi_root =
                FindSfpiRootFrom(info.dli_fname)) {
          setenv("TT_METAL_SFPI_ROOT", sfpi_root->string().c_str(), 0);
        }
      }
    }

    if (!configured_runtime_root) {
      if (std::optional<std::filesystem::path> root =
              FindTtMetalRuntimeRootFrom(std::filesystem::current_path())) {
        tt::llrt::RunTimeOptions::set_root_dir(root->string());
      }
    }

    if (!has_runtime_asset_root_override) {
      if (std::optional<std::filesystem::path> asset_root =
              MaterializeEmbeddedRuntimeAssets()) {
        setenv("TT_METAL_RUNTIME_ASSET_ROOT", asset_root->string().c_str(), 0);
      }
    }

    if (!has_sfpi_root_override && std::getenv("TT_METAL_SFPI_ROOT") == nullptr) {
      if (std::optional<std::filesystem::path> sfpi_root =
              FindSfpiRootFrom(std::filesystem::current_path())) {
        setenv("TT_METAL_SFPI_ROOT", sfpi_root->string().c_str(), 0);
      }
    }
  });
}

std::shared_ptr<MeshDevice> GetTtMetalMeshDevice(int local_hardware_id) {
  EnsureTtMetalRuntimeReady();
  return RuntimeDevices().Get(local_hardware_id);
}
