// ttmlir-opt for libtt's tt-mlir build: tt-mlir's passes and dialects plus
// MLIR's generic transforms (canonicalize, cse, ...), without the upstream
// all-passes registry, which libtt does not build.

#include "mlir/IR/MLIRContext.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

#include "ttmlir/RegisterAll.h"

int main(int argc, char **argv) {
  mlir::registerTransformsPasses();
  mlir::tt::registerAllPasses();

  mlir::DialectRegistry registry;
  mlir::tt::registerAllDialects(registry);
  mlir::tt::registerAllExtensions(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "ttmlir optimizer driver\n", registry));
}
