#include "cpp/tt_metal_matmul_runtime.h"

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/alignment.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/page_config.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/experimental/tensor/spec/memory_config/memory_config.hpp>
#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/work_split.hpp>
#include <ttnn/operations/embedding/device/embedding_device_operation.hpp>
#include <ttnn/operations/matmul/device/matmul_device_operation.hpp>
#include <ttnn/operations/reduction/topk/device/topk_device_operation.hpp>
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
#include <sstream>
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
    case PJRT_Buffer_Type_U16:
      return sizeof(uint16_t);
    case PJRT_Buffer_Type_U32:
    case PJRT_Buffer_Type_S32:
      return sizeof(uint32_t);
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
    case PJRT_Buffer_Type_U32:
      return tt::tt_metal::DataType::UINT32;
    case PJRT_Buffer_Type_S32:
      return tt::tt_metal::DataType::INT32;
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

std::string DimsToString(const std::vector<int64_t>& dims) {
  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < dims.size(); ++i) {
    if (i != 0) {
      os << ",";
    }
    os << dims[i];
  }
  os << "]";
  return os.str();
}

PJRT_Error* TensorSpecFor(PJRT_Buffer_Type type,
                          const std::vector<int64_t>& dims,
                          tt::tt_metal::Layout metal_layout,
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
      tt::tt_metal::PageConfig(metal_layout),
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
                              tt::tt_metal::Layout layout,
                              ttnn::Tensor* out) {
  std::optional<tt::tt_metal::TensorSpec> spec;
  if (PJRT_Error* error = TensorSpecFor(operand.type, operand.dims, layout, &spec)) {
    return error;
  }
  switch (operand.type) {
    case PJRT_Buffer_Type_U32: {
      std::vector<uint32_t> values;
      if (PJRT_Error* error = BytesToVector(operand.data, &values)) {
        return error;
      }
      *out = ttnn::Tensor::from_vector(std::move(values), *spec, mesh_device);
      return nullptr;
    }
    case PJRT_Buffer_Type_S32: {
      std::vector<int32_t> values;
      if (PJRT_Error* error = BytesToVector(operand.data, &values)) {
        return error;
      }
      *out = ttnn::Tensor::from_vector(std::move(values), *spec, mesh_device);
      return nullptr;
    }
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
      return Unimplemented("tt-metal tensor import supports u32, s32, bf16, and f32 inputs");
  }
}

PJRT_Error* TensorToBytes(const ttnn::Tensor& tensor,
                          PJRT_Buffer_Type output_type,
                          std::vector<std::byte>* output) {
  switch (output_type) {
    case PJRT_Buffer_Type_U32: {
      std::vector<uint32_t> values = tensor.to_vector<uint32_t>();
      VectorToBytes(values, output);
      return nullptr;
    }
    case PJRT_Buffer_Type_S32: {
      std::vector<int32_t> values = tensor.to_vector<int32_t>();
      VectorToBytes(values, output);
      return nullptr;
    }
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
      return Unimplemented("tt-metal tensor export supports u32, s32, bf16, and f32 outputs");
  }
}

template <typename T>
PJRT_Error* CopyTopKRows(const std::vector<T>& source,
                         size_t rows,
                         size_t source_k,
                         size_t target_k,
                         std::vector<T>* out) {
  if (source.size() != rows * source_k) {
    return Internal("tt-metal top_k source tensor size does not match expected shape");
  }
  out->resize(rows * target_k);
  for (size_t row = 0; row < rows; ++row) {
    const T* source_row = source.data() + row * source_k;
    T* target_row = out->data() + row * target_k;
    std::copy(source_row, source_row + target_k, target_row);
  }
  return nullptr;
}

PJRT_Error* TopKValuesToBytes(const ttnn::Tensor& tensor,
                              PJRT_Buffer_Type output_type,
                              size_t rows,
                              size_t source_k,
                              size_t target_k,
                              std::vector<std::byte>* output) {
  if (output_type != PJRT_Buffer_Type_BF16) {
    return Unimplemented("tt-metal top_k currently supports bf16 value outputs");
  }
  std::vector<bfloat16> source = tensor.to_vector<bfloat16>();
  std::vector<bfloat16> trimmed;
  if (PJRT_Error* error = CopyTopKRows(source, rows, source_k, target_k, &trimmed)) {
    return error;
  }
  VectorToBytes(trimmed, output);
  return nullptr;
}

template <typename SourceT>
PJRT_Error* TopKIndicesToTypedBytes(const std::vector<SourceT>& source,
                                    PJRT_Buffer_Type output_type,
                                    size_t rows,
                                    size_t source_k,
                                    size_t target_k,
                                    std::vector<std::byte>* output) {
  if (source.size() != rows * source_k) {
    return Internal("tt-metal top_k index tensor size does not match expected shape");
  }
  switch (output_type) {
    case PJRT_Buffer_Type_U16: {
      std::vector<uint16_t> values(rows * target_k);
      for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < target_k; ++col) {
          values[row * target_k + col] =
              static_cast<uint16_t>(source[row * source_k + col]);
        }
      }
      VectorToBytes(values, output);
      return nullptr;
    }
    case PJRT_Buffer_Type_U32: {
      std::vector<uint32_t> values(rows * target_k);
      for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < target_k; ++col) {
          values[row * target_k + col] =
              static_cast<uint32_t>(source[row * source_k + col]);
        }
      }
      VectorToBytes(values, output);
      return nullptr;
    }
    case PJRT_Buffer_Type_S32: {
      std::vector<int32_t> values(rows * target_k);
      for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < target_k; ++col) {
          values[row * target_k + col] =
              static_cast<int32_t>(source[row * source_k + col]);
        }
      }
      VectorToBytes(values, output);
      return nullptr;
    }
    default:
      return Unimplemented("tt-metal top_k currently supports u16/u32/s32 indices");
  }
}

PJRT_Error* TopKIndicesToBytes(const ttnn::Tensor& tensor,
                               PJRT_Buffer_Type output_type,
                               size_t rows,
                               size_t source_k,
                               size_t target_k,
                               std::vector<std::byte>* output) {
  switch (tensor.dtype()) {
    case tt::tt_metal::DataType::UINT16: {
      std::vector<uint16_t> source = tensor.to_vector<uint16_t>();
      return TopKIndicesToTypedBytes(source, output_type, rows, source_k, target_k, output);
    }
    case tt::tt_metal::DataType::UINT32: {
      std::vector<uint32_t> source = tensor.to_vector<uint32_t>();
      return TopKIndicesToTypedBytes(source, output_type, rows, source_k, target_k, output);
    }
    default:
      return Internal("tt-metal top_k returned unsupported index tensor dtype");
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
        "[..., K, N]; got lhs_dims=" + DimsToString(request.lhs.dims) +
        " rhs_dims=" + DimsToString(request.rhs.dims) +
        " lhs_contracting=" + DimsToString(request.lhs_contracting_dimensions) +
        " rhs_contracting=" + DimsToString(request.rhs_contracting_dimensions));
  }
  if (request.lhs_batching_dimensions.size() != request.rhs_batching_dimensions.size()) {
    return InvalidArgument("dot_general lhs/rhs batching dimension counts must match");
  }
  if (!IsPrefixDims(request.lhs_batching_dimensions) ||
      !IsPrefixDims(request.rhs_batching_dimensions)) {
    return Unimplemented(
        "tt-metal matmul currently supports leading prefix batching dimensions; got lhs_dims=" +
        DimsToString(request.lhs.dims) + " rhs_dims=" + DimsToString(request.rhs.dims) +
        " lhs_batching=" + DimsToString(request.lhs_batching_dimensions) +
        " rhs_batching=" + DimsToString(request.rhs_batching_dimensions));
  }
  if (request.lhs.dims[lhs_contract] != request.rhs.dims[rhs_contract]) {
    return InvalidArgument("dot_general contracting dimensions must have matching size");
  }
  return nullptr;
}

bool HasTtnnMatmulContractingDims(const TtMetalMatmulRequest& request) {
  if (request.lhs.dims.size() < 2 || request.rhs.dims.size() < 2 ||
      request.lhs_contracting_dimensions.size() != 1 ||
      request.rhs_contracting_dimensions.size() != 1) {
    return false;
  }
  return request.lhs_contracting_dimensions[0] ==
             static_cast<int64_t>(request.lhs.dims.size() - 1) &&
         request.rhs_contracting_dimensions[0] ==
             static_cast<int64_t>(request.rhs.dims.size() - 2);
}

bool IsBatchedMatrixVectorDot(const TtMetalMatmulRequest& request) {
  if (request.lhs.dims.size() < 2 || request.rhs.dims.empty() ||
      request.lhs_contracting_dimensions.size() != 1 ||
      request.rhs_contracting_dimensions.size() != 1) {
    return false;
  }
  return request.lhs_contracting_dimensions[0] ==
             static_cast<int64_t>(request.lhs.dims.size() - 1) &&
         request.rhs_contracting_dimensions[0] ==
             static_cast<int64_t>(request.rhs.dims.size() - 1);
}

bool ContainsDim(const std::vector<int64_t>& dims, size_t dim) {
  return std::find(dims.begin(), dims.end(), static_cast<int64_t>(dim)) != dims.end();
}

PJRT_Error* ProductOfDims(const std::vector<int64_t>& dims,
                          const std::vector<int64_t>& dim_indices,
                          size_t* out) {
  size_t product = 1;
  for (int64_t dim_index : dim_indices) {
    const int64_t value = dims[static_cast<size_t>(dim_index)];
    if (value < 0) {
      return InvalidArgument("matmul tensor dimensions must be >= 0");
    }
    const size_t dim = static_cast<size_t>(value);
    if (dim != 0 && product > std::numeric_limits<size_t>::max() / dim) {
      return ResourceExhausted("matmul canonicalized dimension overflows size_t");
    }
    product *= dim;
  }
  *out = product;
  return nullptr;
}

std::vector<size_t> RowMajorStrides(const std::vector<int64_t>& dims) {
  std::vector<size_t> strides(dims.size(), 1);
  size_t stride = 1;
  for (size_t i = dims.size(); i > 0; --i) {
    strides[i - 1] = stride;
    stride *= static_cast<size_t>(dims[i - 1]);
  }
  return strides;
}

void SetCoordsFromFlatIndex(size_t flat_index,
                            const std::vector<int64_t>& dims,
                            const std::vector<int64_t>& dim_indices,
                            std::vector<size_t>* coords) {
  for (size_t i = dim_indices.size(); i > 0; --i) {
    const size_t dim = static_cast<size_t>(dims[static_cast<size_t>(dim_indices[i - 1])]);
    const size_t coord = dim == 0 ? 0 : flat_index % dim;
    flat_index = dim == 0 ? 0 : flat_index / dim;
    (*coords)[static_cast<size_t>(dim_indices[i - 1])] = coord;
  }
}

size_t LinearIndexFromCoords(const std::vector<size_t>& coords,
                             const std::vector<size_t>& strides) {
  size_t index = 0;
  for (size_t dim = 0; dim < coords.size(); ++dim) {
    index += coords[dim] * strides[dim];
  }
  return index;
}

PJRT_Error* ValidateDotDimensions(const TtMetalMatmulRequest& request) {
  if (request.lhs.type != request.rhs.type) {
    return InvalidArgument("tt-metal matmul requires matching lhs/rhs input dtypes");
  }
  if (request.lhs.dims.empty() || request.rhs.dims.empty()) {
    return Unimplemented("tt-metal matmul currently requires rank >= 1 inputs");
  }
  if (request.lhs_contracting_dimensions.size() != 1 ||
      request.rhs_contracting_dimensions.size() != 1) {
    return Unimplemented("tt-metal matmul currently supports a single contracting dimension");
  }
  const int64_t lhs_contract = request.lhs_contracting_dimensions[0];
  const int64_t rhs_contract = request.rhs_contracting_dimensions[0];
  if (lhs_contract < 0 ||
      lhs_contract >= static_cast<int64_t>(request.lhs.dims.size()) ||
      rhs_contract < 0 ||
      rhs_contract >= static_cast<int64_t>(request.rhs.dims.size())) {
    return InvalidArgument("dot_general contracting dimensions are out of bounds");
  }
  if (request.lhs.dims[static_cast<size_t>(lhs_contract)] !=
      request.rhs.dims[static_cast<size_t>(rhs_contract)]) {
    return InvalidArgument("dot_general contracting dimensions must have matching size");
  }
  if (request.lhs_batching_dimensions.size() != request.rhs_batching_dimensions.size()) {
    return InvalidArgument("dot_general lhs/rhs batching dimension counts must match");
  }
  std::vector<bool> lhs_seen(request.lhs.dims.size(), false);
  std::vector<bool> rhs_seen(request.rhs.dims.size(), false);
  lhs_seen[static_cast<size_t>(lhs_contract)] = true;
  rhs_seen[static_cast<size_t>(rhs_contract)] = true;
  for (size_t i = 0; i < request.lhs_batching_dimensions.size(); ++i) {
    const int64_t lhs_dim = request.lhs_batching_dimensions[i];
    const int64_t rhs_dim = request.rhs_batching_dimensions[i];
    if (lhs_dim < 0 || lhs_dim >= static_cast<int64_t>(request.lhs.dims.size()) ||
        rhs_dim < 0 || rhs_dim >= static_cast<int64_t>(request.rhs.dims.size())) {
      return InvalidArgument("dot_general batching dimensions are out of bounds");
    }
    if (lhs_seen[static_cast<size_t>(lhs_dim)] ||
        rhs_seen[static_cast<size_t>(rhs_dim)]) {
      return InvalidArgument("dot_general batching dimensions must be unique and not contracted");
    }
    if (request.lhs.dims[static_cast<size_t>(lhs_dim)] !=
        request.rhs.dims[static_cast<size_t>(rhs_dim)]) {
      return InvalidArgument("dot_general batching dimensions must have matching sizes");
    }
    lhs_seen[static_cast<size_t>(lhs_dim)] = true;
    rhs_seen[static_cast<size_t>(rhs_dim)] = true;
  }
  return nullptr;
}

PJRT_Error* CanonicalizeDotGeneralForTtnn(const TtMetalMatmulRequest& request,
                                          TtMetalMatmulRequest* out) {
  if (out == nullptr) {
    return InvalidArgument("canonicalized matmul request output must not be null");
  }

  TtMetalMatmulRequest expanded = request;
  if (!HasTtnnMatmulContractingDims(expanded) && IsBatchedMatrixVectorDot(expanded)) {
    expanded.rhs.dims.push_back(1);
    expanded.rhs_contracting_dimensions[0] =
        static_cast<int64_t>(expanded.rhs.dims.size() - 2);
  }
  if (PJRT_Error* error = ValidateDotDimensions(expanded)) {
    return error;
  }

  const size_t lhs_rank = expanded.lhs.dims.size();
  const size_t rhs_rank = expanded.rhs.dims.size();
  const size_t lhs_contract =
      static_cast<size_t>(expanded.lhs_contracting_dimensions[0]);
  const size_t rhs_contract =
      static_cast<size_t>(expanded.rhs_contracting_dimensions[0]);

  std::vector<int64_t> lhs_free_dims;
  std::vector<int64_t> rhs_free_dims;
  for (size_t dim = 0; dim < lhs_rank; ++dim) {
    if (dim != lhs_contract && !ContainsDim(expanded.lhs_batching_dimensions, dim)) {
      lhs_free_dims.push_back(static_cast<int64_t>(dim));
    }
  }
  for (size_t dim = 0; dim < rhs_rank; ++dim) {
    if (dim != rhs_contract && !ContainsDim(expanded.rhs_batching_dimensions, dim)) {
      rhs_free_dims.push_back(static_cast<int64_t>(dim));
    }
  }

  size_t batch_size = 0;
  size_t m_size = 0;
  size_t n_size = 0;
  if (PJRT_Error* error =
          ProductOfDims(expanded.lhs.dims, expanded.lhs_batching_dimensions, &batch_size)) {
    return error;
  }
  if (PJRT_Error* error = ProductOfDims(expanded.lhs.dims, lhs_free_dims, &m_size)) {
    return error;
  }
  if (PJRT_Error* error = ProductOfDims(expanded.rhs.dims, rhs_free_dims, &n_size)) {
    return error;
  }
  const size_t k_size = static_cast<size_t>(expanded.lhs.dims[lhs_contract]);
  const bool has_batch = !expanded.lhs_batching_dimensions.empty();

  const size_t element_size = ByteSize(expanded.lhs.type);
  if (element_size == 0) {
    return Unimplemented("tt-metal matmul supports bf16 and f32 inputs");
  }

  TtMetalMatmulRequest normalized = expanded;
  normalized.lhs.dims = has_batch ? std::vector<int64_t>{
                                        static_cast<int64_t>(batch_size),
                                        static_cast<int64_t>(m_size),
                                        static_cast<int64_t>(k_size)}
                                  : std::vector<int64_t>{
                                        static_cast<int64_t>(m_size),
                                        static_cast<int64_t>(k_size)};
  normalized.rhs.dims = has_batch ? std::vector<int64_t>{
                                        static_cast<int64_t>(batch_size),
                                        static_cast<int64_t>(k_size),
                                        static_cast<int64_t>(n_size)}
                                  : std::vector<int64_t>{
                                        static_cast<int64_t>(k_size),
                                        static_cast<int64_t>(n_size)};
  normalized.lhs_batching_dimensions = has_batch ? std::vector<int64_t>{0}
                                                 : std::vector<int64_t>{};
  normalized.rhs_batching_dimensions = has_batch ? std::vector<int64_t>{0}
                                                 : std::vector<int64_t>{};
  normalized.lhs_contracting_dimensions = {
      static_cast<int64_t>(normalized.lhs.dims.size() - 1)};
  normalized.rhs_contracting_dimensions = {
      static_cast<int64_t>(normalized.rhs.dims.size() - 2)};

  const std::vector<size_t> lhs_strides = RowMajorStrides(expanded.lhs.dims);
  const std::vector<size_t> rhs_strides = RowMajorStrides(expanded.rhs.dims);
  normalized.lhs.data.assign(expanded.lhs.data.size(), std::byte{0});
  normalized.rhs.data.assign(expanded.rhs.data.size(), std::byte{0});

  std::vector<size_t> lhs_coords(lhs_rank, 0);
  for (size_t b = 0; b < batch_size; ++b) {
    SetCoordsFromFlatIndex(b, expanded.lhs.dims, expanded.lhs_batching_dimensions,
                           &lhs_coords);
    for (size_t m = 0; m < m_size; ++m) {
      SetCoordsFromFlatIndex(m, expanded.lhs.dims, lhs_free_dims, &lhs_coords);
      for (size_t k = 0; k < k_size; ++k) {
        lhs_coords[lhs_contract] = k;
        const size_t src_index = LinearIndexFromCoords(lhs_coords, lhs_strides);
        const size_t dst_index = has_batch ? (b * m_size + m) * k_size + k
                                           : m * k_size + k;
        std::memcpy(normalized.lhs.data.data() + dst_index * element_size,
                    expanded.lhs.data.data() + src_index * element_size,
                    element_size);
      }
    }
  }

  std::vector<size_t> rhs_coords(rhs_rank, 0);
  for (size_t b = 0; b < batch_size; ++b) {
    SetCoordsFromFlatIndex(b, expanded.rhs.dims, expanded.rhs_batching_dimensions,
                           &rhs_coords);
    for (size_t k = 0; k < k_size; ++k) {
      rhs_coords[rhs_contract] = k;
      for (size_t n = 0; n < n_size; ++n) {
        SetCoordsFromFlatIndex(n, expanded.rhs.dims, rhs_free_dims, &rhs_coords);
        const size_t src_index = LinearIndexFromCoords(rhs_coords, rhs_strides);
        const size_t dst_index = has_batch ? (b * k_size + k) * n_size + n
                                           : k * n_size + n;
        std::memcpy(normalized.rhs.data.data() + dst_index * element_size,
                    expanded.rhs.data.data() + src_index * element_size,
                    element_size);
      }
    }
  }

  *out = std::move(normalized);
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
  TtMetalMatmulRequest normalized_request;
  if (PJRT_Error* error = CanonicalizeDotGeneralForTtnn(request, &normalized_request)) {
    return error;
  }
  if (PJRT_Error* error = ValidateTtnnCompatibleDotGeneral(normalized_request)) {
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
    if (PJRT_Error* error =
            TensorFromOperand(normalized_request.lhs, mesh_device.get(),
                              tt::tt_metal::Layout::TILE, &lhs)) {
      return error;
    }
    if (PJRT_Error* error =
            TensorFromOperand(normalized_request.rhs, mesh_device.get(),
                              tt::tt_metal::Layout::TILE, &rhs)) {
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

PJRT_Error* ExecuteTtMetalEmbedding(const TtMetalEmbeddingRequest& request,
                                    std::vector<std::byte>* output) {
  if (output == nullptr) {
    return InvalidArgument("output must not be null");
  }
  if (request.output_type != PJRT_Buffer_Type_BF16) {
    return Unimplemented("tt-metal embedding currently supports bf16 outputs");
  }
  if (request.table.type != PJRT_Buffer_Type_BF16) {
    return Unimplemented("tt-metal embedding currently requires a bf16 table");
  }
  if (request.indices.type != PJRT_Buffer_Type_S32 &&
      request.indices.type != PJRT_Buffer_Type_U32) {
    return Unimplemented("tt-metal embedding currently requires s32/u32 indices");
  }
  if (request.table.dims.size() != 4 || request.indices.dims.size() != 4) {
    return Unimplemented("tt-metal embedding expects 4D internal table and indices tensors");
  }
  if (PJRT_Error* error = ValidateBufferBytes(request.table)) {
    return error;
  }
  if (PJRT_Error* error = ValidateBufferBytes(request.indices)) {
    return error;
  }

  size_t output_elements = 0;
  if (PJRT_Error* error = CheckedElementCount(request.output_dims, &output_elements)) {
    return error;
  }
  try {
    EnsureTtMetalRuntimeRoot();
    std::shared_ptr<MeshDevice> mesh_device = RuntimeDevices().Get(request.local_hardware_id);
    ttnn::Tensor indices;
    ttnn::Tensor table;

    TtMetalMatmulOperand indices_as_u32 = request.indices;
    indices_as_u32.type = PJRT_Buffer_Type_U32;
    if (PJRT_Error* error = TensorFromOperand(
            indices_as_u32, mesh_device.get(), tt::tt_metal::Layout::ROW_MAJOR, &indices)) {
      return error;
    }
    if (PJRT_Error* error =
            TensorFromOperand(request.table, mesh_device.get(), tt::tt_metal::Layout::ROW_MAJOR, &table)) {
      return error;
    }

    ttnn::Tensor result = ttnn::prim::embedding(
        indices, table, false, ttnn::prim::EmbeddingsType::GENERIC, ttnn::DRAM_MEMORY_CONFIG);
    if (PJRT_Error* error = TensorToBytes(result, request.output_type, output)) {
      return error;
    }
  } catch (const std::exception& ex) {
    return Internal(std::string("tt-metal embedding failed: ") + ex.what());
  } catch (...) {
    return Internal("tt-metal embedding failed with unknown exception");
  }

  const size_t output_element_bytes = ByteSize(request.output_type);
  if (output_elements > std::numeric_limits<size_t>::max() / output_element_bytes) {
    return ResourceExhausted("embedding output byte size overflows size_t");
  }
  if (output->size() != output_elements * output_element_bytes) {
    return Internal("tt-metal embedding output byte size does not match executable metadata");
  }
  return nullptr;
}

PJRT_Error* ExecuteTtMetalTopK(const TtMetalTopKRequest& request,
                               std::vector<std::byte>* values_output,
                               std::vector<std::byte>* indices_output) {
  if (values_output == nullptr || indices_output == nullptr) {
    return InvalidArgument("top_k output buffers must not be null");
  }
  if (request.k == 0) {
    return Unimplemented("tt-metal top_k currently requires k > 0");
  }
  if (request.input.type != PJRT_Buffer_Type_BF16) {
    return Unimplemented("tt-metal top_k currently supports bf16 inputs");
  }
  if (request.input.dims.empty()) {
    return Unimplemented("tt-metal top_k currently requires rank >= 1 inputs");
  }
  if (PJRT_Error* error = ValidateBufferBytes(request.input)) {
    return error;
  }

  const int64_t width_i64 = request.input.dims.back();
  if (width_i64 < 0) {
    return InvalidArgument("top_k input dimensions must be >= 0");
  }
  const size_t width = static_cast<size_t>(width_i64);
  if (request.k > width) {
    return InvalidArgument("top_k k cannot exceed input last dimension");
  }
  const size_t adjusted_k =
      static_cast<size_t>(kTileCols) *
      ((static_cast<size_t>(request.k) + static_cast<size_t>(kTileCols) - 1) /
       static_cast<size_t>(kTileCols));
  if (adjusted_k > width) {
    return Unimplemented("tt-metal top_k requires input width >= tile-aligned k");
  }

  size_t input_elements = 0;
  if (PJRT_Error* error = CheckedElementCount(request.input.dims, &input_elements)) {
    return error;
  }
  if (width == 0 || input_elements % width != 0) {
    return InvalidArgument("top_k input shape is invalid");
  }
  const size_t rows = input_elements / width;

  size_t values_elements = 0;
  size_t indices_elements = 0;
  if (PJRT_Error* error = CheckedElementCount(request.values_output_dims, &values_elements)) {
    return error;
  }
  if (PJRT_Error* error = CheckedElementCount(request.indices_output_dims, &indices_elements)) {
    return error;
  }
  const size_t expected_output_elements = rows * static_cast<size_t>(request.k);
  if (values_elements != expected_output_elements ||
      indices_elements != expected_output_elements) {
    return InvalidArgument("top_k output shapes do not match input rows and k");
  }

  TtMetalMatmulOperand input_4d = request.input;
  input_4d.dims = {1, 1, static_cast<int64_t>(rows), static_cast<int64_t>(width)};

  try {
    EnsureTtMetalRuntimeRoot();
    std::shared_ptr<MeshDevice> mesh_device = RuntimeDevices().Get(request.local_hardware_id);
    ttnn::Tensor input_tensor;
    if (PJRT_Error* error =
            TensorFromOperand(input_4d, mesh_device.get(), tt::tt_metal::Layout::TILE, &input_tensor)) {
      return error;
    }

    const auto compute_with_storage_grid_size =
        input_tensor.device()->compute_with_storage_grid_size();
    const uint32_t max_number_of_cores =
        compute_with_storage_grid_size.y * compute_with_storage_grid_size.x;
    const auto full_core_grids = tt::tt_metal::num_cores_to_corerangeset(
        max_number_of_cores, compute_with_storage_grid_size, true);
    auto [values_tensor, indices_tensor] = ttnn::prim::topk(
        input_tensor,
        static_cast<uint32_t>(adjusted_k),
        -1,
        true,
        true,
        ttnn::DRAM_MEMORY_CONFIG,
        full_core_grids,
        std::nullopt,
        std::nullopt);

    if (PJRT_Error* error = TopKValuesToBytes(
            values_tensor, request.values_output_type, rows, adjusted_k,
            static_cast<size_t>(request.k), values_output)) {
      return error;
    }
    if (PJRT_Error* error = TopKIndicesToBytes(
            indices_tensor, request.indices_output_type, rows, adjusted_k,
            static_cast<size_t>(request.k), indices_output)) {
      return error;
    }
  } catch (const std::exception& ex) {
    return Internal(std::string("tt-metal top_k failed: ") + ex.what());
  } catch (...) {
    return Internal("tt-metal top_k failed with unknown exception");
  }

  const size_t values_element_bytes = ByteSize(request.values_output_type);
  const size_t indices_element_bytes = ByteSize(request.indices_output_type);
  if (values_element_bytes == 0 || indices_element_bytes == 0) {
    return Unimplemented("tt-metal top_k output type is unsupported");
  }
  if (values_output->size() != values_elements * values_element_bytes ||
      indices_output->size() != indices_elements * indices_element_bytes) {
    return Internal("tt-metal top_k output byte size does not match executable metadata");
  }
  return nullptr;
}
