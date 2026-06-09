#include "mlir/rope_fusing_pattern.h"

#include <optional>

#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/stablehlo_utils.h"

namespace libtt::mlir_frontend {
namespace {

struct SliceHalf {
  mlir::Value input;
  int64_t start = 0;
  int64_t width = 0;
};

struct ScaledHalf {
  SliceHalf half;
  mlir::Value scale;
};

struct RopeComponents {
  mlir::Value input;
  mlir::Value cos;
  mlir::Value sin;
};

std::optional<mlir::RankedTensorType> staticRankedTensor(mlir::Value value) {
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
  if (!type || !type.hasStaticShape()) {
    return std::nullopt;
  }
  return type;
}

bool allOnes(llvm::ArrayRef<int64_t> values) {
  for (int64_t value : values) {
    if (value != 1) {
      return false;
    }
  }
  return true;
}

mlir::Value peelScaleBroadcasts(mlir::Value value) {
  while (true) {
    value = peelIdentityCustomCalls(value);
    auto broadcast = value.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
    if (!broadcast) {
      return value;
    }
    value = broadcast.getOperand();
  }
}

std::optional<SliceHalf> matchHalfSlice(mlir::Value value) {
  value = peelIdentityCustomCalls(value);
  auto slice = value.getDefiningOp<mlir::stablehlo::SliceOp>();
  if (!slice || !allOnes(slice.getStrides())) {
    return std::nullopt;
  }

  auto inputType = staticRankedTensor(slice.getOperand());
  auto outputType = staticRankedTensor(slice.getResult());
  if (!inputType || !outputType || inputType->getRank() < 2 ||
      inputType->getRank() != outputType->getRank() ||
      inputType->getElementType() != outputType->getElementType()) {
    return std::nullopt;
  }

  auto starts = slice.getStartIndices();
  auto limits = slice.getLimitIndices();
  auto inputShape = inputType->getShape();
  auto outputShape = outputType->getShape();
  int64_t rank = inputType->getRank();
  int64_t half = outputShape.back();
  if (half <= 0 || inputShape.back() != 2 * half) {
    return std::nullopt;
  }

  for (int64_t dim = 0; dim < rank - 1; ++dim) {
    if (starts[dim] != 0 || limits[dim] != inputShape[dim] ||
        outputShape[dim] != inputShape[dim]) {
      return std::nullopt;
    }
  }
  int64_t start = starts[rank - 1];
  if ((start != 0 && start != half) || limits[rank - 1] != start + half) {
    return std::nullopt;
  }

  return SliceHalf{slice.getOperand(), start, half};
}

std::optional<ScaledHalf> matchScaledHalf(mlir::Value value) {
  value = peelIdentityCustomCalls(value);
  auto mul = value.getDefiningOp<mlir::stablehlo::MulOp>();
  if (!mul) {
    return std::nullopt;
  }
  if (auto half = matchHalfSlice(mul.getLhs())) {
    return ScaledHalf{*half, peelScaleBroadcasts(mul.getRhs())};
  }
  if (auto half = matchHalfSlice(mul.getRhs())) {
    return ScaledHalf{*half, peelScaleBroadcasts(mul.getLhs())};
  }
  return std::nullopt;
}

bool sameHalf(const ScaledHalf &value, mlir::Value input, int64_t start,
              int64_t width) {
  return value.half.input == input && value.half.start == start &&
         value.half.width == width;
}

std::optional<RopeComponents> matchRope(mlir::stablehlo::ConcatenateOp concat) {
  if (concat->getNumOperands() != 2) {
    return std::nullopt;
  }
  auto outputType = staticRankedTensor(concat.getResult());
  if (!outputType || !outputType->getElementType().isBF16() ||
      outputType->getRank() != 3 || concat.getDimension() != 2) {
    return std::nullopt;
  }

  auto first = peelIdentityCustomCalls(concat.getOperand(0));
  auto second = peelIdentityCustomCalls(concat.getOperand(1));
  auto sub = first.getDefiningOp<mlir::stablehlo::SubtractOp>();
  auto add = second.getDefiningOp<mlir::stablehlo::AddOp>();
  if (!sub || !add) {
    return std::nullopt;
  }

  auto x1Cos = matchScaledHalf(sub.getLhs());
  auto x2Sin = matchScaledHalf(sub.getRhs());
  if (!x1Cos || !x2Sin || x1Cos->half.start != 0 ||
      x2Sin->half.start != x1Cos->half.width ||
      x1Cos->half.input != x2Sin->half.input ||
      x1Cos->half.width != x2Sin->half.width) {
    return std::nullopt;
  }

  auto lhs = matchScaledHalf(add.getLhs());
  auto rhs = matchScaledHalf(add.getRhs());
  if (!lhs || !rhs) {
    return std::nullopt;
  }

  mlir::Value input = x1Cos->half.input;
  int64_t half = x1Cos->half.width;
  mlir::Value cos = x1Cos->scale;
  mlir::Value sin = x2Sin->scale;
  bool direct =
      sameHalf(*lhs, input, half, half) && lhs->scale == cos &&
      sameHalf(*rhs, input, 0, half) && rhs->scale == sin;
  bool swapped =
      sameHalf(*rhs, input, half, half) && rhs->scale == cos &&
      sameHalf(*lhs, input, 0, half) && lhs->scale == sin;
  if (!direct && !swapped) {
    return std::nullopt;
  }

  auto inputType = staticRankedTensor(input);
  auto cosType = staticRankedTensor(cos);
  auto sinType = staticRankedTensor(sin);
  if (!inputType || !cosType || !sinType || *inputType != outputType ||
      cosType->getElementType() != inputType->getElementType() ||
      sinType->getElementType() != inputType->getElementType() ||
      cosType->getShape() != sinType->getShape() ||
      cosType->getRank() != 2 || cosType->getShape().back() != half ||
      cosType->getShape().front() != inputType->getShape().front()) {
    return std::nullopt;
  }

  return RopeComponents{input, cos, sin};
}

} // namespace

mlir::LogicalResult RopeFusing::matchAndRewrite(
    mlir::stablehlo::ConcatenateOp concatOp,
    mlir::PatternRewriter &rewriter) const {
  auto components = matchRope(concatOp);
  if (!components) {
    return mlir::failure();
  }

  rewriter.setInsertionPoint(concatOp);
  auto customCall = rewriter.create<mlir::stablehlo::CustomCallOp>(
      concatOp.getLoc(), concatOp->getResultTypes(),
      mlir::ValueRange{components->input, components->cos, components->sin},
      kRopeTarget,
      /*hasSideEffect=*/false,
      /*backendConfig=*/nullptr,
      mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI,
      rewriter.getArrayAttr({}),
      /*calledComputations=*/nullptr,
      /*operandLayouts=*/nullptr,
      /*resultLayouts=*/nullptr);
  rewriter.replaceOp(concatOp, customCall.getResults());
  return mlir::success();
}

} // namespace libtt::mlir_frontend
