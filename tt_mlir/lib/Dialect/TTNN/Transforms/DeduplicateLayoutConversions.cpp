// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/DeduplicateLayoutConversions.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#include <utility>

namespace mlir::tt::ttnn {

void deduplicateLayoutConversions(ModuleOp module) {
  module.walk([](Block *block) {
    llvm::DenseMap<std::pair<Value, Type>, ToTensorSpecOp> first;
    for (Operation &op : llvm::make_early_inc_range(*block)) {
      auto conversion = dyn_cast<ToTensorSpecOp>(op);
      if (!conversion) {
        continue;
      }
      auto type = conversion.getResult().getType();
      auto layout = mlir::dyn_cast_or_null<TTNNLayoutAttr>(
          mlir::cast<RankedTensorType>(type).getEncoding());
      if (!layout || layout.hasL1BufferType()) {
        continue;
      }
      auto [it, inserted] =
          first.try_emplace({conversion.getInput(), type}, conversion);
      if (!inserted) {
        conversion.getResult().replaceAllUsesWith(it->second.getResult());
        conversion.erase();
      }
    }
  });
}

} // namespace mlir::tt::ttnn
