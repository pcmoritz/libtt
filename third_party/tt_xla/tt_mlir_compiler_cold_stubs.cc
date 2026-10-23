#include "ttmlir/Conversion/TTIRToEmitPy/TTIRToEmitPy.h"
#include "ttmlir/Conversion/TTNNToEmitPy/TTNNToEmitPy.h"
#include "ttmlir/Target/LLVM/LLVMToDynamicLib.h"

#include "llvm/Support/ErrorHandling.h"

namespace {

[[noreturn]] void unsupported(const char *name) {
  llvm::report_fatal_error(name);
}

} // namespace

namespace mlir::tt {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTTIRCPUToEmitPyPass() {
  unsupported("TTIR CPU to EmitPy is not linked in this libtt build");
}

std::unique_ptr<OperationPass<ModuleOp>> createConvertTTNNToEmitPyPass() {
  unsupported("TTNN to EmitPy is not linked in this libtt build");
}

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTTNNToEmitPyPass(const ConvertTTNNToEmitPyOptions &) {
  unsupported("TTNN to EmitPy is not linked in this libtt build");
}

std::unique_ptr<OperationPass<ModuleOp>> createEmitPyConstEvalCachingPass() {
  unsupported("EmitPy const-eval caching is not linked in this libtt build");
}

std::unique_ptr<OperationPass<ModuleOp>> createEmitPyFormExpressionsPass() {
  unsupported("EmitPy form-expressions is not linked in this libtt build");
}

std::unique_ptr<OperationPass<ModuleOp>> createEmitPyNameVarsPass() {
  unsupported("EmitPy name-vars is not linked in this libtt build");
}

std::unique_ptr<OperationPass<ModuleOp>> createEmitPyLinkModulesPass() {
  unsupported("EmitPy link-modules is not linked in this libtt build");
}

std::unique_ptr<OperationPass<ModuleOp>> createEmitPyAddImportsPass() {
  unsupported("EmitPy add-imports is not linked in this libtt build");
}

} // namespace mlir::tt

namespace mlir::tt::llvm_to_cpu {

LogicalResult translateLLVMToDyLib(Operation *, llvm::raw_ostream &) {
  return failure();
}

} // namespace mlir::tt::llvm_to_cpu
