// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

// libtt's TTIR rewrites of D2M subgraphs before TTIRToD2M (simplifyForD2M,
// see ttmlir/D2MFusion.h): an L2 clip's sign/min/abs form as one per-row
// factor, per-row factors multiplied together first, constants on the right
// of commutative ops (D2M scalarizes them there), and cheap values copied per
// user where GridSelection spreads rows over cores.

#include "ttmlir/D2MFusion.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "ttmlir/AffineMapUtils.h"
#include "ttmlir/Asserts.h"
#include "ttmlir/Conversion/TTIRToD2M/TTIRToD2M.h"
#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/Utils/GridSelectionUtils.h"
#include "ttmlir/Dialect/D2M/Utils/TopKUtils.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Dialect/D2M/Utils/VirtualGrid.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <type_traits>

namespace mlir::tt {
namespace {

// (A * f) * g == A * (f * g) when f and g are both constant along
// A's last dim (per-row factors, e.g. an L2 clip's factor and a per-token
// scale): the row factors combine on the reduced shape once instead of
// scaling every element of A twice, and the epilogue reads one row operand.
static void reassociateRowFactors(ModuleOp module) {
  SmallVector<ttir::MultiplyOp> candidates;
  module.walk([&](ttir::MultiplyOp op) { candidates.push_back(op); });
  auto isRowFactor = [](Value v, RankedTensorType full) {
    auto t = mlir::cast<RankedTensorType>(v.getType());
    return t.getRank() == full.getRank() && t.getShape().back() == 1 &&
           full.getShape().back() > 1;
  };
  for (ttir::MultiplyOp outer : candidates) {
    auto outType = mlir::cast<RankedTensorType>(outer.getType());
    if (!outType.getElementType().isF32()) {
      continue;
    }
    for (auto [inner, g] : {std::pair(outer.getLhs(), outer.getRhs()),
                            std::pair(outer.getRhs(), outer.getLhs())}) {
      auto innerMul = inner.getDefiningOp<ttir::MultiplyOp>();
      auto innerType = mlir::cast<RankedTensorType>(inner.getType());
      if (!innerMul || !innerMul->hasOneUse() || !isRowFactor(g, outType) ||
          innerType.getShape() != outType.getShape() ||
          innerType.getElementType() != outType.getElementType()) {
        continue;
      }
      Value a, f;
      if (isRowFactor(innerMul.getRhs(), outType)) {
        a = innerMul.getLhs();
        f = innerMul.getRhs();
      } else if (isRowFactor(innerMul.getLhs(), outType)) {
        a = innerMul.getRhs();
        f = innerMul.getLhs();
      } else {
        continue;
      }
      auto aType = mlir::cast<RankedTensorType>(a.getType());
      if (aType.getShape() != outType.getShape() ||
          aType.getElementType() != outType.getElementType()) {
        continue;
      }
      auto fType = mlir::cast<RankedTensorType>(f.getType());
      auto gType = mlir::cast<RankedTensorType>(g.getType());
      if (!fType.getElementType().isF32() || !gType.getElementType().isF32()) {
        continue;
      }
      SmallVector<int64_t> shape;
      bool ok = true;
      for (auto [x, y] : llvm::zip(fType.getShape(), gType.getShape())) {
        if (x != y && x != 1 && y != 1) {
          ok = false;
        }
        shape.push_back(std::max(x, y));
      }
      if (!ok) {
        continue;
      }
      OpBuilder b(outer);
      // The factor product's own layout (a TTNN encoding describes the tile
      // grid of its shape).
      Attribute fgEncoding = fType.getShape() == ArrayRef<int64_t>(shape)
                                 ? fType.getEncoding()
                                 : gType.getEncoding();
      if (auto layout =
              mlir::dyn_cast_if_present<ttnn::TTNNLayoutAttr>(fgEncoding)) {
        fgEncoding = ttnn::TTNNLayoutAttr::Builder(layout, shape).build();
      }
      auto fgType =
          RankedTensorType::get(shape, outType.getElementType(), fgEncoding);
      Value fg = b.create<ttir::MultiplyOp>(outer.getLoc(), fgType, f, g);
      Value result = b.create<ttir::MultiplyOp>(outer.getLoc(), outType, a, fg);
      outer.getResult().replaceAllUsesWith(result);
      outer.erase();
      innerMul.erase();
      break;
    }
  }
}

// Constants (ttir.full) go to the right of commutative binary ops,
// where D2M's ScalarizeConstTensors turns them into scalar operands.
static void constantsToRhs(ModuleOp module) {
  module.walk([](Operation *op) {
    if (!mlir::isa<ttir::MultiplyOp, ttir::AddOp, ttir::MaximumOp,
                   ttir::MinimumOp>(op) ||
        op->getNumOperands() != 2) {
      return;
    }
    Value lhs = op->getOperand(0), rhs = op->getOperand(1);
    if (lhs.getDefiningOp<ttir::FullOp>() &&
        !rhs.getDefiningOp<ttir::FullOp>()) {
      op->setOperand(0, rhs);
      op->setOperand(1, lhs);
    }
  });
}

// Give each user of a cheap elementwise value its own copy (TTIR,
// before D2M), in functions whose row reductions GridSelection spreads over
// the columns (few tile rows or a transcendental epilogue): a producer shared
// by a reduction and an elementwise epilogue becomes two generics, the
// reduction's copy on whole rows, the epilogue's spread.
static void rematerializeForSpreadRows(ModuleOp module) {
  auto cheap = [](Operation *op) {
    if (op->getNumResults() != 1 || op->getNumRegions() != 0) {
      return false;
    }
    if (!mlir::isa<ttir::TypecastOp, ttir::MultiplyOp, ttir::AddOp,
                   ttir::SubtractOp, ttir::AbsOp, ttir::NegOp>(op)) {
      return false;
    }
    auto t = mlir::dyn_cast<RankedTensorType>(op->getResult(0).getType());
    return t && llvm::all_of(op->getOperands(), [](Value v) {
             Operation *def = v.getDefiningOp();
             return !def || mlir::isa<BlockArgument>(v) ||
                    mlir::isa<ttir::TypecastOp, ttir::MultiplyOp, ttir::AddOp,
                              ttir::SubtractOp, ttir::AbsOp, ttir::NegOp,
                              ttir::FullOp, ttir::ReshapeOp>(def);
           });
  };
  // As GridSelection decides: more than one tile column and fewer than four
  // tile rows, or a transcendental epilogue.
  auto transcendental = [](func::FuncOp f) {
    bool found = false;
    f.walk([&](Operation *op) {
      found |= mlir::isa<ttir::SiluOp, ttir::SigmoidOp, ttir::ExpOp,
                         ttir::TanhOp, ttir::GeluOp, ttir::LogOp>(op);
    });
    return found;
  };
  auto spreads = [&](Operation *op) {
    auto t = mlir::dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    if (!t || t.getRank() < 2) {
      return false;
    }
    int64_t rank = t.getRank();
    int64_t rows = (t.getDimSize(rank - 2) + 31) / 32;
    for (int64_t d = 0; d < rank - 2; ++d) {
      rows *= t.getDimSize(d);
    }
    return (t.getDimSize(rank - 1) + 31) / 32 > 1 &&
           (rows < 4 || transcendental(op->getParentOfType<func::FuncOp>()));
  };
  llvm::SmallPtrSet<Operation *, 4> spreadFuncs;
  module.walk([&](Operation *op) {
    if (mlir::isa<ttir::SumOp, ttir::MeanOp, ttir::MaxOp>(op) && spreads(op)) {
      if (auto f = op->getParentOfType<func::FuncOp>()) {
        spreadFuncs.insert(f);
      }
    }
  });
  int budget = 32;
  bool changed = true;
  while (changed && budget > 0) {
    changed = false;
    SmallVector<Operation *> ops;
    module.walk([&](Operation *op) {
      auto f = op->getParentOfType<func::FuncOp>();
      if (f && spreadFuncs.contains(f) && cheap(op)) {
        ops.push_back(op);
      }
    });
    for (Operation *op : llvm::reverse(ops)) {
      SmallVector<Operation *> users;
      for (Operation *u : op->getResult(0).getUsers()) {
        if (!llvm::is_contained(users, u)) {
          users.push_back(u);
        }
      }
      if (users.size() < 2) {
        continue;
      }
      for (Operation *u : llvm::drop_begin(users)) {
        if (budget-- <= 0) {
          break;
        }
        OpBuilder b(u);
        Operation *clone = b.clone(*op);
        u->replaceUsesOfWith(op->getResult(0), clone->getResult(0));
        changed = true;
      }
    }
  }
}

// sign(v) * min(|v|, C * |v * R|) == v * min(1, C * |R|), since
// min(a, k a) = a min(1, k) for a = |v| >= 0 and sign(v) |v| = v. The left form
// is how models write an L2 clip (Inkling 2); the right one computes a per-row
// factor once (R, C broadcast along the reduced dims) instead of four
// elementwise ops over the whole row, and needs fewer DST inputs per tile.
static void simplifySignMinAbs(ModuleOp module) {
  SmallVector<ttir::MultiplyOp> candidates;
  module.walk([&](ttir::MultiplyOp op) { candidates.push_back(op); });
  auto otherOf = [](Value pair0, Value pair1, Value known) -> Value {
    return pair0 == known ? pair1 : (pair1 == known ? pair0 : Value());
  };
  for (ttir::MultiplyOp out : candidates) {
    Value v, minVal;
    for (auto [a, b] : {std::pair(out.getLhs(), out.getRhs()),
                        std::pair(out.getRhs(), out.getLhs())}) {
      if (auto sign = a.getDefiningOp<ttir::SignOp>()) {
        v = sign.getInput();
        minVal = b;
        break;
      }
    }
    auto minOp =
        minVal ? minVal.getDefiningOp<ttir::MinimumOp>() : ttir::MinimumOp();
    if (!minOp) {
      continue;
    }
    Value t;
    for (auto [a, b] : {std::pair(minOp.getLhs(), minOp.getRhs()),
                        std::pair(minOp.getRhs(), minOp.getLhs())}) {
      auto absV = a.getDefiningOp<ttir::AbsOp>();
      if (absV && absV.getInput() == v) {
        t = b;
        break;
      }
    }
    auto tMul = t ? t.getDefiningOp<ttir::MultiplyOp>() : ttir::MultiplyOp();
    if (!tMul) {
      continue;
    }
    Value c, n;
    for (auto [a, b] : {std::pair(tMul.getLhs(), tMul.getRhs()),
                        std::pair(tMul.getRhs(), tMul.getLhs())}) {
      if (auto absN = b.getDefiningOp<ttir::AbsOp>()) {
        c = a;
        n = absN.getInput();
        break;
      }
    }
    auto nMul = n ? n.getDefiningOp<ttir::MultiplyOp>() : ttir::MultiplyOp();
    if (!nMul) {
      continue;
    }
    Value r = otherOf(nMul.getLhs(), nMul.getRhs(), v);
    if (!r) {
      continue;
    }
    auto rType = mlir::cast<RankedTensorType>(r.getType());
    auto outType = mlir::cast<RankedTensorType>(out.getType());
    if (!rType.getElementType().isF32() || !outType.getElementType().isF32()) {
      continue;
    }
    OpBuilder b(out);
    Location loc = out.getLoc();
    Value absR = b.create<ttir::AbsOp>(loc, rType, r);
    Value scaled = b.create<ttir::MultiplyOp>(loc, rType, c, absR);
    // min(1, x) as a clamp: a splat constant into ttir.minimum is not
    // scalarized and has no layout inside a D2M subgraph.
    Value factor = b.create<ttir::ClampScalarOp>(
        loc, rType, scaled,
        b.getF32FloatAttr(-std::numeric_limits<float>::max()),
        b.getF32FloatAttr(1.0f));
    Value replacement = b.create<ttir::MultiplyOp>(loc, outType, v, factor);
    out.getResult().replaceAllUsesWith(replacement);
    out.erase();
  }
}

} // namespace
} // namespace mlir::tt

namespace mlir::tt::d2m_fusion {

void simplifyForD2M(ModuleOp module) {
  constantsToRhs(module);
  simplifySignMinAbs(module);
  reassociateRowFactors(module);
  constantsToRhs(module);
  rematerializeForSpreadRows(module);
}

} // namespace mlir::tt::d2m_fusion
