#include "cpp/tt_metal_matmul_runtime.h"

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/alignment.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/page_config.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/experimental/tensor/spec/memory_config/memory_config.hpp>
#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <ttnn/operations/matmul/device/matmul_device_operation.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <tt_metal/llrt/rtoptions.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

constexpr uint32_t kTileRows = 32;
constexpr uint32_t kTileCols = 32;

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
      path / ("bazel-" + path.filename().string()) / "external" / "+http_archive+tt_metal";
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

std::optional<std::filesystem::path> FindTtMetalRuntimeAssetRootFrom(
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

  for (std::filesystem::path current = path; !current.empty();
       current = current.parent_path()) {
    const std::filesystem::path external_root =
        current / "external" / "+http_archive+tt_metal";
    if (IsTtMetalRuntimeAssetRoot(external_root)) {
      return external_root;
    }
    if (IsTtMetalRuntimeAssetRoot(current)) {
      return current;
    }
    if (current == current.root_path()) {
      break;
    }
  }
  return std::nullopt;
}

std::optional<std::filesystem::path> FindSfpiRootFrom(
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

void EnsureTtMetalRuntimeRoot() {
  static std::once_flag once;
  std::call_once(once, [] {
    const bool has_runtime_root_override =
        std::getenv("TT_METAL_RUNTIME_ROOT") != nullptr;
    const bool has_runtime_asset_root_override =
        std::getenv("TT_METAL_RUNTIME_ASSET_ROOT") != nullptr;
    const bool has_sfpi_root_override =
        std::getenv("TT_METAL_SFPI_ROOT") != nullptr;

    Dl_info info;
    if (dladdr(reinterpret_cast<void*>(&EnsureTtMetalRuntimeRoot), &info) != 0 &&
        info.dli_fname != nullptr) {
      bool found_runtime_root = false;
      if (!has_runtime_root_override) {
        if (std::optional<std::filesystem::path> root =
                FindTtMetalRuntimeRootFrom(info.dli_fname)) {
          tt::llrt::RunTimeOptions::set_root_dir(root->string());
          found_runtime_root = true;
        }
      }
      if (!has_runtime_asset_root_override) {
        if (std::optional<std::filesystem::path> asset_root =
                FindTtMetalRuntimeAssetRootFrom(info.dli_fname)) {
          setenv("TT_METAL_RUNTIME_ASSET_ROOT", asset_root->string().c_str(),
                 0);
        }
      }
      if (!has_sfpi_root_override) {
        if (std::optional<std::filesystem::path> sfpi_root =
                FindSfpiRootFrom(info.dli_fname)) {
          setenv("TT_METAL_SFPI_ROOT", sfpi_root->string().c_str(), 0);
        }
      }
      if (found_runtime_root) {
        return;
      }
    }

    if (!has_runtime_root_override) {
      if (std::optional<std::filesystem::path> root =
              FindTtMetalRuntimeRootFrom(std::filesystem::current_path())) {
        tt::llrt::RunTimeOptions::set_root_dir(root->string());
      }
    }

    if (!has_runtime_asset_root_override) {
      if (std::optional<std::filesystem::path> asset_root =
              FindTtMetalRuntimeAssetRootFrom(std::filesystem::current_path())) {
        setenv("TT_METAL_RUNTIME_ASSET_ROOT", asset_root->string().c_str(), 0);
      }
    }

    if (!has_sfpi_root_override) {
      if (std::optional<std::filesystem::path> sfpi_root =
              FindSfpiRootFrom(std::filesystem::current_path())) {
        setenv("TT_METAL_SFPI_ROOT", sfpi_root->string().c_str(), 0);
      }
    }
  });
}

PJRT_Error* CheckedElementCount(const std::vector<int64_t>& dims, size_t* out) {
  size_t count = 1;
  for (int64_t dim : dims) {
    if (dim < 0) {
      return InvalidArgument("matmul tensor dimensions must be >= 0");
    }
    const size_t value = static_cast<size_t>(dim);
    if (value != 0 && count > std::numeric_limits<size_t>::max() / value) {
      return ResourceExhausted("matmul tensor shape overflows size_t");
    }
    count *= value;
  }
  *out = count;
  return nullptr;
}

size_t ByteSize(PJRT_Buffer_Type type) {
  switch (type) {
    case PJRT_Buffer_Type_BF16:
      return sizeof(bfloat16);
    case PJRT_Buffer_Type_F32:
      return sizeof(float);
    default:
      return 0;
  }
}

std::optional<tt::tt_metal::DataType> MetalDataType(PJRT_Buffer_Type type) {
  switch (type) {
    case PJRT_Buffer_Type_BF16:
      return tt::tt_metal::DataType::BFLOAT16;
    case PJRT_Buffer_Type_F32:
      return tt::tt_metal::DataType::FLOAT32;
    default:
      return std::nullopt;
  }
}

PJRT_Error* ValidateBufferBytes(const TtMetalMatmulOperand& operand) {
  const size_t bytes_per_element = ByteSize(operand.type);
  if (bytes_per_element == 0) {
    return Unimplemented("tt-metal matmul supports bf16 and f32 inputs");
  }
  size_t element_count = 0;
  if (PJRT_Error* error = CheckedElementCount(operand.dims, &element_count)) {
    return error;
  }
  if (element_count > std::numeric_limits<size_t>::max() / bytes_per_element) {
    return ResourceExhausted("matmul tensor byte size overflows size_t");
  }
  if (operand.data.size() != element_count * bytes_per_element) {
    return InvalidArgument("matmul input byte size does not match tensor shape");
  }
  return nullptr;
}

PJRT_Error* ShapeFromDims(const std::vector<int64_t>& dims, tt::tt_metal::Shape* out) {
  tt::tt_metal::Shape::Container values;
  values.reserve(dims.size());
  for (int64_t dim : dims) {
    if (dim < 0 || dim > std::numeric_limits<uint32_t>::max()) {
      return InvalidArgument("matmul tensor dimensions must fit uint32_t for tt-metal tensors");
    }
    values.push_back(static_cast<uint32_t>(dim));
  }
  *out = tt::tt_metal::Shape(std::move(values));
  return nullptr;
}

PJRT_Error* TensorSpecFor(PJRT_Buffer_Type type,
                          const std::vector<int64_t>& dims,
                          std::optional<tt::tt_metal::TensorSpec>* out) {
  std::optional<tt::tt_metal::DataType> dtype = MetalDataType(type);
  if (!dtype.has_value()) {
    return Unimplemented("tt-metal matmul supports bf16 and f32 tensors");
  }
  tt::tt_metal::Shape shape;
  if (PJRT_Error* error = ShapeFromDims(dims, &shape)) {
    return error;
  }
  tt::tt_metal::TensorLayout layout(
      *dtype,
      tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
      tt::tt_metal::MemoryConfig{},
      tt::tt_metal::Alignment({kTileRows, kTileCols}));
  out->emplace(std::move(shape), std::move(layout));
  return nullptr;
}

template <typename T>
PJRT_Error* BytesToVector(const std::vector<std::byte>& bytes, std::vector<T>* out) {
  static_assert(std::is_trivially_copyable_v<T>);
  if (bytes.size() % sizeof(T) != 0) {
    return InvalidArgument("matmul byte buffer is not element aligned");
  }
  out->resize(bytes.size() / sizeof(T));
  if (!bytes.empty()) {
    std::memcpy(out->data(), bytes.data(), bytes.size());
  }
  return nullptr;
}

template <typename T>
void VectorToBytes(const std::vector<T>& values, std::vector<std::byte>* out) {
  static_assert(std::is_trivially_copyable_v<T>);
  out->resize(values.size() * sizeof(T));
  if (!values.empty()) {
    std::memcpy(out->data(), values.data(), out->size());
  }
}

PJRT_Error* TensorFromOperand(const TtMetalMatmulOperand& operand,
                              MeshDevice* mesh_device,
                              ttnn::Tensor* out) {
  std::optional<tt::tt_metal::TensorSpec> spec;
  if (PJRT_Error* error = TensorSpecFor(operand.type, operand.dims, &spec)) {
    return error;
  }
  switch (operand.type) {
    case PJRT_Buffer_Type_BF16: {
      std::vector<bfloat16> values;
      if (PJRT_Error* error = BytesToVector(operand.data, &values)) {
        return error;
      }
      *out = ttnn::Tensor::from_vector(std::move(values), *spec, mesh_device);
      return nullptr;
    }
    case PJRT_Buffer_Type_F32: {
      std::vector<float> values;
      if (PJRT_Error* error = BytesToVector(operand.data, &values)) {
        return error;
      }
      *out = ttnn::Tensor::from_vector(std::move(values), *spec, mesh_device);
      return nullptr;
    }
    default:
      return Unimplemented("tt-metal matmul supports bf16 and f32 inputs");
  }
}

PJRT_Error* TensorToBytes(const ttnn::Tensor& tensor,
                          PJRT_Buffer_Type output_type,
                          std::vector<std::byte>* output) {
  switch (output_type) {
    case PJRT_Buffer_Type_BF16: {
      std::vector<bfloat16> values = tensor.to_vector<bfloat16>();
      VectorToBytes(values, output);
      return nullptr;
    }
    case PJRT_Buffer_Type_F32: {
      std::vector<float> values = tensor.to_vector<float>();
      VectorToBytes(values, output);
      return nullptr;
    }
    default:
      return Unimplemented("tt-metal matmul supports bf16 and f32 outputs");
  }
}

bool IsPrefixDims(const std::vector<int64_t>& dims) {
  for (size_t i = 0; i < dims.size(); ++i) {
    if (dims[i] != static_cast<int64_t>(i)) {
      return false;
    }
  }
  return true;
}

PJRT_Error* ValidateTtnnCompatibleDotGeneral(const TtMetalMatmulRequest& request) {
  if (request.lhs.type != request.rhs.type) {
    return InvalidArgument("tt-metal matmul requires matching lhs/rhs input dtypes");
  }
  if (request.lhs.dims.size() < 2 || request.rhs.dims.size() < 2) {
    return Unimplemented("tt-metal matmul currently requires rank >= 2 inputs");
  }
  if (request.lhs_contracting_dimensions.size() != 1 ||
      request.rhs_contracting_dimensions.size() != 1) {
    return Unimplemented("tt-metal matmul currently supports a single contracting dimension");
  }
  const int64_t lhs_contract = request.lhs_contracting_dimensions[0];
  const int64_t rhs_contract = request.rhs_contracting_dimensions[0];
  if (lhs_contract != static_cast<int64_t>(request.lhs.dims.size() - 1) ||
      rhs_contract != static_cast<int64_t>(request.rhs.dims.size() - 2)) {
    return Unimplemented(
        "tt-metal matmul currently supports dot_general forms equivalent to [..., M, K] x "
        "[..., K, N]");
  }
  if (request.lhs_batching_dimensions.size() != request.rhs_batching_dimensions.size()) {
    return InvalidArgument("dot_general lhs/rhs batching dimension counts must match");
  }
  if (!IsPrefixDims(request.lhs_batching_dimensions) ||
      !IsPrefixDims(request.rhs_batching_dimensions)) {
    return Unimplemented("tt-metal matmul currently supports leading prefix batching dimensions");
  }
  if (request.lhs.dims[lhs_contract] != request.rhs.dims[rhs_contract]) {
    return InvalidArgument("dot_general contracting dimensions must have matching size");
  }
  return nullptr;
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
  // tt-metal owns process-global device teardown. Destroying cached MeshDevice
  // handles from a plugin static destructor can run after that teardown.
  static MeshDeviceCache* cache = new MeshDeviceCache;
  return *cache;
}

}  // namespace

PJRT_Error* ExecuteTtMetalMatmul(const TtMetalMatmulRequest& request,
                                 std::vector<std::byte>* output) {
  if (output == nullptr) {
    return InvalidArgument("output must not be null");
  }
  if (PJRT_Error* error = ValidateBufferBytes(request.lhs)) {
    return error;
  }
  if (PJRT_Error* error = ValidateBufferBytes(request.rhs)) {
    return error;
  }
  if (PJRT_Error* error = ValidateTtnnCompatibleDotGeneral(request)) {
    return error;
  }

  size_t output_elements = 0;
  if (PJRT_Error* error = CheckedElementCount(request.output_dims, &output_elements)) {
    return error;
  }
  const size_t output_element_bytes = ByteSize(request.output_type);
  if (output_element_bytes == 0) {
    return Unimplemented("tt-metal matmul supports bf16 and f32 outputs");
  }

  try {
    EnsureTtMetalRuntimeRoot();
    std::shared_ptr<MeshDevice> mesh_device = RuntimeDevices().Get(request.local_hardware_id);
    ttnn::Tensor lhs;
    ttnn::Tensor rhs;
    if (PJRT_Error* error = TensorFromOperand(request.lhs, mesh_device.get(), &lhs)) {
      return error;
    }
    if (PJRT_Error* error = TensorFromOperand(request.rhs, mesh_device.get(), &rhs)) {
      return error;
    }

    const tt::tt_metal::DataType output_dtype = *MetalDataType(request.output_type);
    ttnn::prim::MatmulParams parameters;
    parameters.output_mem_config = ttnn::DRAM_MEMORY_CONFIG;
    parameters.output_dtype = output_dtype;
    ttnn::prim::MatmulParams attributes =
        ttnn::prim::create_matmul_attributes(lhs, rhs, parameters, {std::nullopt});
    std::vector<ttnn::Tensor> results =
        ttnn::prim::matmul(lhs, rhs, std::nullopt, std::nullopt, attributes);
    if (results.empty()) {
      return Internal("tt-metal matmul did not produce an output tensor");
    }
    ttnn::Tensor result = std::move(results.front());
    if (PJRT_Error* error = TensorToBytes(result, request.output_type, output)) {
      return error;
    }
  } catch (const std::exception& ex) {
    return Internal(std::string("tt-metal matmul failed: ") + ex.what());
  } catch (...) {
    return Internal("tt-metal matmul failed with unknown exception");
  }

  if (output_elements > std::numeric_limits<size_t>::max() / output_element_bytes) {
    return ResourceExhausted("matmul output byte size overflows size_t");
  }
  if (output->size() != output_elements * output_element_bytes) {
    return Internal("tt-metal matmul output byte size does not match executable metadata");
  }
  return nullptr;
}
