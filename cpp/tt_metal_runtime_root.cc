#include "cpp/tt_metal_runtime_root.h"

#include "libtt_embedded_runtime_assets.h"

#include <archive.h>
#include <archive_entry.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/file.h>
#include <unistd.h>

namespace {

struct ArchiveReadDeleter {
  void operator()(archive* value) const {
    if (value != nullptr) {
      archive_read_free(value);
    }
  }
};

class ScopedCurrentDirectory {
 public:
  explicit ScopedCurrentDirectory(const std::filesystem::path& path)
      : original_fd_(open(".", O_RDONLY | O_CLOEXEC)) {
    if (original_fd_ == -1) {
      throw std::runtime_error("failed to save current directory: " +
                               std::string(std::strerror(errno)));
    }
    if (chdir(path.c_str()) != 0) {
      const std::string message =
          "failed to enter tt-metal runtime directory: " +
          std::string(std::strerror(errno));
      close(original_fd_);
      original_fd_ = -1;
      throw std::runtime_error(message);
    }
  }

  ScopedCurrentDirectory(const ScopedCurrentDirectory&) = delete;
  ScopedCurrentDirectory& operator=(const ScopedCurrentDirectory&) = delete;

  ~ScopedCurrentDirectory() {
    if (original_fd_ != -1) {
      fchdir(original_fd_);
      close(original_fd_);
    }
  }

 private:
  int original_fd_;
};

class ScopedFileLock {
 public:
  explicit ScopedFileLock(const std::string& path)
      : fd_(open(path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0666)) {
    if (fd_ == -1 || flock(fd_, LOCK_EX) != 0) {
      const std::string message = "failed to lock tt-metal runtime directory: " +
                                  std::string(std::strerror(errno));
      if (fd_ != -1) {
        close(fd_);
      }
      throw std::runtime_error(message);
    }
  }

  ScopedFileLock(const ScopedFileLock&) = delete;
  ScopedFileLock& operator=(const ScopedFileLock&) = delete;

  ~ScopedFileLock() { close(fd_); }

 private:
  int fd_;
};

void ExtractEmbeddedRuntimeArchive(const std::filesystem::path& root) {
  namespace embedded = libtt::tt_metal_runtime_assets;
  if (embedded::kRuntimeArchiveZstdSize == 0) {
    throw std::runtime_error("embedded tt-metal runtime archive is empty");
  }

  std::error_code error;
  std::filesystem::create_directories(root, error);
  if (error) {
    throw std::runtime_error("failed to create tt-metal runtime directory: " +
                             error.message());
  }

  std::unique_ptr<archive, ArchiveReadDeleter> reader(archive_read_new());
  if (!reader) {
    throw std::runtime_error("failed to allocate tt-metal runtime archive reader");
  }
  ScopedCurrentDirectory scoped_runtime_root(root);
  if (archive_read_support_filter_zstd(reader.get()) != ARCHIVE_OK ||
      archive_read_support_format_tar(reader.get()) != ARCHIVE_OK ||
      archive_read_open_memory(reader.get(), embedded::kRuntimeArchiveZstd,
                               embedded::kRuntimeArchiveZstdSize) != ARCHIVE_OK) {
    throw std::runtime_error("failed to open embedded tt-metal runtime archive");
  }

  archive_entry* entry = nullptr;
  while (true) {
    const int status = archive_read_next_header(reader.get(), &entry);
    if (status == ARCHIVE_EOF) {
      return;
    }
    if (status != ARCHIVE_OK || entry == nullptr) {
      throw std::runtime_error("failed to read embedded tt-metal runtime archive");
    }

    const char* pathname = archive_entry_pathname(entry);
    if (pathname == nullptr || pathname[0] == '\0') {
      throw std::runtime_error("embedded tt-metal runtime archive has an empty path");
    }

    constexpr int kExtractFlags = ARCHIVE_EXTRACT_TIME | ARCHIVE_EXTRACT_PERM |
                                  ARCHIVE_EXTRACT_SECURE_SYMLINKS |
                                  ARCHIVE_EXTRACT_SECURE_NODOTDOT |
                                  ARCHIVE_EXTRACT_SECURE_NOABSOLUTEPATHS;
    if (archive_read_extract(reader.get(), entry, kExtractFlags) != ARCHIVE_OK) {
      throw std::runtime_error("failed to extract embedded tt-metal runtime archive");
    }
  }
}

std::filesystem::path MaterializeEmbeddedRuntimeRoot() {
  namespace embedded = libtt::tt_metal_runtime_assets;
  std::error_code error;
  const std::filesystem::path temp_root =
      std::filesystem::temp_directory_path(error);
  if (error) {
    throw std::runtime_error("failed to locate temporary directory: " +
                             error.message());
  }
  const std::filesystem::path root =
      temp_root / ("libtt-tt-metal-runtime-assets-" +
                   std::string(embedded::kRuntimeArchiveFingerprint));

  // Processes that start together, one per chip, extract the archive once:
  // the first under the lock, and the others reuse the finished directory
  // instead of rewriting files that a running process may be reading.
  ScopedFileLock lock(root.string() + ".lock");
  const std::filesystem::path marker = root / ".extracted";
  if (!std::filesystem::exists(marker)) {
    ExtractEmbeddedRuntimeArchive(root);
    const int fd = open(marker.c_str(), O_CREAT | O_WRONLY | O_CLOEXEC, 0644);
    if (fd == -1) {
      throw std::runtime_error("failed to mark tt-metal runtime directory: " +
                               std::string(std::strerror(errno)));
    }
    close(fd);
  }
  return root;
}

}  // namespace

void EnsureTtMetalRuntimeReady() {
  static std::once_flag once;
  std::call_once(once, [] {
    const std::filesystem::path runtime_root = MaterializeEmbeddedRuntimeRoot();
    const std::string runtime_root_string = runtime_root.string();

    setenv("TT_METAL_RUNTIME_ROOT", runtime_root_string.c_str(), 1);
  });
}
