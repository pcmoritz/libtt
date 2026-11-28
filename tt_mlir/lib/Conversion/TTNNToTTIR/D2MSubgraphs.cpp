// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

// libtt's TTNN->TTIR conversion of D2M subgraphs (see ttmlir/D2MFusion.h):
// TTNN binary ops' activation lists expanded, RMS norms decomposed, the head
// splits and merges and column slices D2M reads as views, per-head subgraphs
// rewritten head-major or group-major and wide rows spread over more tile
// rows, scalar constants inlined and identical subgraphs deduplicated.

#include "ttmlir/D2MFusion.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ttmlir/Conversion/TTNNToTTIR/TTNNToTTIR.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "llvm/ADT/STLExtras.h"

#include <functional>
#include <limits>
#include <type_traits>

namespace {

// TTNN binary ops carry unary activations on their inputs and result (fused
// by TTNN's glue patterns); TTIR has none, so each becomes its own op.
mlir::Value createActivation(mlir::OpBuilder &builder, mlir::Location loc,
                             mlir::tt::ttnn::UnaryOpType type,
                             mlir::Value input) {
  namespace ttir = mlir::tt::ttir;
  using mlir::tt::ttnn::UnaryOpType;
  auto t = mlir::cast<mlir::RankedTensorType>(input.getType());
  switch (type) {
  case UnaryOpType::Abs:
    return builder.create<ttir::AbsOp>(loc, t, input);
  case UnaryOpType::Exp:
    return builder.create<ttir::ExpOp>(loc, t, input);
  case UnaryOpType::Recip:
    return builder.create<ttir::ReciprocalOp>(loc, t, input);
  case UnaryOpType::Gelu:
    return builder.create<ttir::GeluOp>(loc, t, input);
  case UnaryOpType::Relu:
    return builder.create<ttir::ReluOp>(loc, t, input);
  case UnaryOpType::Sqrt:
    return builder.create<ttir::SqrtOp>(loc, t, input);
  case UnaryOpType::Sigmoid:
    return builder.create<ttir::SigmoidOp>(loc, t, input);
  case UnaryOpType::Log:
    return builder.create<ttir::LogOp>(loc, t, input);
  case UnaryOpType::Tanh:
    return builder.create<ttir::TanhOp>(loc, t, input);
  case UnaryOpType::Sign:
    return builder.create<ttir::SignOp>(loc, t, input);
  case UnaryOpType::Square:
    return builder.create<ttir::SquareOp>(loc, t, input);
  case UnaryOpType::Rsqrt:
    return builder.create<ttir::RsqrtOp>(loc, t, input);
  case UnaryOpType::Silu:
    return builder.create<ttir::SiluOp>(loc, t, input);
  case UnaryOpType::Neg:
    return builder.create<ttir::NegOp>(loc, t, input);
  default:
    return nullptr;
  }
}

bool isExpandableActivation(mlir::tt::ttnn::UnaryOpType type) {
  using mlir::tt::ttnn::UnaryOpType;
  switch (type) {
  case UnaryOpType::Abs:
  case UnaryOpType::Exp:
  case UnaryOpType::Recip:
  case UnaryOpType::Gelu:
  case UnaryOpType::Relu:
  case UnaryOpType::Sqrt:
  case UnaryOpType::Sigmoid:
  case UnaryOpType::Log:
  case UnaryOpType::Tanh:
  case UnaryOpType::Sign:
  case UnaryOpType::Square:
  case UnaryOpType::Rsqrt:
  case UnaryOpType::Silu:
  case UnaryOpType::Neg:
    return true;
  default:
    return false;
  }
}

// Applies an activation list in order (the first entry is applied first).
mlir::Value applyActivations(mlir::OpBuilder &builder, mlir::Location loc,
                             mlir::Value value, mlir::ArrayAttr activations) {
  if (!activations) {
    return value;
  }
  for (mlir::Attribute attr : activations) {
    auto activation = mlir::cast<mlir::tt::ttnn::UnaryWithParamAttr>(attr);
    value = createActivation(builder, loc, activation.getOpType(), value);
  }
  return value;
}

} // namespace

namespace mlir::tt::d2m_fusion {

static RankedTensorType withShape(RankedTensorType t, ArrayRef<int64_t> shape);
static mlir::tt::ttir::ReshapeOp reinterpret(mlir::OpBuilder &b,
                                             mlir::Location loc, mlir::Value v,
                                             RankedTensorType to);

mlir::Value applyTTNNActivations(mlir::OpBuilder &builder, mlir::Location loc,
                                 mlir::Value value,
                                 mlir::ArrayAttr activations) {
  return ::applyActivations(builder, loc, value, activations);
}

bool canExpandTTNNActivations(ArrayAttr activations) {
  if (!activations) {
    return true;
  }
  return llvm::all_of(activations, [](Attribute attr) {
    auto activation = mlir::dyn_cast<ttnn::UnaryWithParamAttr>(attr);
    return activation && activation.getParams().empty() &&
           isExpandableActivation(activation.getOpType());
  });
}

// t with a new shape and element type (its TTNN layout rebuilt for them).
static mlir::RankedTensorType retypeTensor(mlir::RankedTensorType t,
                                           llvm::ArrayRef<int64_t> shape,
                                           mlir::Type element) {
  mlir::Attribute encoding = t.getEncoding();
  if (auto layout =
          mlir::dyn_cast_if_present<mlir::tt::ttnn::TTNNLayoutAttr>(encoding)) {
    encoding = mlir::tt::ttnn::TTNNLayoutAttr::Builder(layout, shape)
                   .setElementType(element)
                   .setLayout(layout.getLayout())
                   .build();
  }
  return mlir::RankedTensorType::get(shape, element, encoding);
}

namespace {
// RMS norms inside D2M subgraphs become their FP32
// primitives: x * rsqrt(sum(x * x) / D + eps) * weight [* silu(gate)], the
// result narrowed by the last multiply (as TTNN's own kernels compute it).
// Rows wider than a core's share split over cores later (splitWideRows).
template <typename SrcOp>
class TTNNNormDecompositionPattern : public mlir::OpConversionPattern<SrcOp> {
  using mlir::OpConversionPattern<SrcOp>::OpConversionPattern;

  static mlir::RankedTensorType retype(mlir::RankedTensorType t,
                                       llvm::ArrayRef<int64_t> shape,
                                       mlir::Type element) {
    return retypeTensor(t, shape, element);
  }

public:
  mlir::LogicalResult
  matchAndRewrite(SrcOp op, typename SrcOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rw) const override {
    namespace ttir = mlir::tt::ttir;
    mlir::Location loc = op.getLoc();
    auto outType = mlir::cast<mlir::RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    mlir::Type f32 = rw.getF32Type();
    auto asF32 = [&](mlir::Value v) -> mlir::Value {
      auto t = mlir::cast<mlir::RankedTensorType>(v.getType());
      if (t.getElementType().isF32()) {
        return v;
      }
      return rw.create<ttir::TypecastOp>(loc, retype(t, t.getShape(), f32), v);
    };
    auto inT = mlir::cast<mlir::RankedTensorType>(adaptor.getInput().getType());
    int64_t rank = inT.getRank();
    int64_t width = inT.getShape().back();
    auto scalar = [&](float value, int64_t r) -> mlir::Value {
      return rw.create<ttir::FullOp>(
          loc,
          mlir::RankedTensorType::get(llvm::SmallVector<int64_t>(r, 1), f32),
          llvm::SmallVector<int32_t>(r, 1), rw.getF32FloatAttr(value));
    };
    const float eps = op.getEpsilon().convertToFloat();
    // The weight [D] as it came: through the [1, D] -> [D] reshape the
    // subgraph outlining adds for the TTNN op.
    auto weightArg = [&]() -> mlir::Value {
      mlir::Value w = adaptor.getWeight();
      if (auto flat = w.getDefiningOp<ttir::ReshapeOp>();
          flat && mlir::cast<mlir::RankedTensorType>(flat.getInput().getType())
                          .getNumElements() == width) {
        return flat.getInput();
      }
      return w;
    };
    mlir::Value x = asF32(adaptor.getInput());
    auto xt = mlir::cast<mlir::RankedTensorType>(x.getType());
    llvm::SmallVector<int64_t> rowShape(xt.getShape());
    rowShape.back() = 1;
    auto rowType = retype(xt, rowShape, f32);
    mlir::Value squares = rw.create<ttir::MultiplyOp>(loc, xt, x, x);
    mlir::Value sum = rw.create<ttir::SumOp>(
        loc, rowType, squares, /*keep_dim=*/true,
        rw.getI32ArrayAttr({static_cast<int32_t>(rank - 1)}));
    mlir::Value mean = rw.create<ttir::MultiplyOp>(
        loc, rowType, sum, scalar(1.0f / static_cast<float>(width), rank));
    mlir::Value var =
        rw.create<ttir::AddOp>(loc, rowType, mean, scalar(eps, rank));
    mlir::Value rstd = rw.create<ttir::RsqrtOp>(loc, rowType, var);
    // The last multiply narrows as TTNN->TTIR's FP32-then-narrow binaries do:
    // FP32 with the result's encoding, then a typecast D2M folds into the pack.
    auto finish = [&](mlir::Value a, mlir::Value b) {
      if (outType.getElementType().isF32()) {
        rw.replaceOpWithNewOp<ttir::MultiplyOp>(op, outType, a, b);
        return;
      }
      auto wide = mlir::RankedTensorType::get(outType.getShape(), f32,
                                              outType.getEncoding());
      mlir::Value product = rw.create<ttir::MultiplyOp>(loc, wide, a, b);
      rw.replaceOpWithNewOp<ttir::TypecastOp>(op, outType, product);
    };
    if (!adaptor.getWeight()) {
      finish(x, rstd);
      return mlir::success();
    }
    mlir::Value normed = rw.create<ttir::MultiplyOp>(loc, xt, x, rstd);
    // The weight [D] as one row [1, ..., 1, D] (the same tiles).
    mlir::Value w = asF32(weightArg());
    auto wt = mlir::cast<mlir::RankedTensorType>(w.getType());
    if (wt.getRank() != rank) {
      llvm::SmallVector<int64_t> wShape(rank, 1);
      wShape.back() = width;
      auto reshape = rw.create<ttir::ReshapeOp>(
          loc, retype(wt, wShape, f32), w,
          rw.getI32ArrayAttr(llvm::to_vector_of<int32_t>(wShape)));
      reshape->setAttr("libtt.tile_reinterpret", rw.getUnitAttr());
      w = reshape;
    }
    if constexpr (std::is_same_v<SrcOp, mlir::tt::ttnn::RMSNormGatedOp>) {
      mlir::Value weighted = rw.create<ttir::MultiplyOp>(loc, xt, normed, w);
      mlir::Value z = asF32(adaptor.getGate());
      finish(weighted, rw.create<ttir::SiluOp>(loc, xt, z));
    } else {
      finish(normed, w);
    }
    return mlir::success();
  }
};

// Inkling 2's attention residuals at decode (one token, one query row; see
// ttir.attn_res) over one tile row of more than 32 tiles, in k chunks of c
// columns, one per core (attnResChunks), as one row-merged kernel. X holds
// the core's chunk of source j in rows j and N + j; the query multiplies
// rows N.. (M), so one row sum gives every source's sum of squares (rows
// < N) and query product (rows N + j), summed over the chunks by one
// exchange (ttir.chunk_allreduce; each row's total in every column). Down
// the rows then: r = rsqrt(mean + eps), the logits (products moved up to
// rows < N, times r; other rows -inf), their softmax (column reductions)
// and the weights (times r when normalized; other rows 0); the mixture is
// the column sum of weights * X. With a norm weight, a second exchange sums
// the mixture's squares. The reductions down the rows read the 2N logical
// rows only (ttir.column_reduce): no padding mask splits the kernel.
class TTNNAttnResDecompositionPattern
    : public mlir::OpConversionPattern<mlir::tt::ttnn::AttnResOp> {
  using OpConversionPattern::OpConversionPattern;

public:
  mlir::LogicalResult
  matchAndRewrite(mlir::tt::ttnn::AttnResOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rw) const override {
    namespace ttir = mlir::tt::ttir;
    std::optional<int64_t> chunks = attnResChunks(op);
    if (!chunks) {
      return rw.notifyMatchFailure(op, "not an attention residual D2M takes");
    }
    mlir::Location loc = op.getLoc();
    mlir::Type f32 = rw.getF32Type();
    auto outType = mlir::cast<mlir::RankedTensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    mlir::ValueRange sources = adaptor.getSources();
    auto st = mlir::cast<mlir::RankedTensorType>(sources.front().getType());
    const int64_t n = static_cast<int64_t>(sources.size());
    const int64_t width = st.getShape().back(), k = *chunks, c = width / k;
    const int64_t kRows = 2 * n;
    auto type = [&](int64_t r, int64_t cols) {
      return retypeTensor(st, {k, r, cols}, f32);
    };
    auto asF32 = [&](mlir::Value v) -> mlir::Value {
      auto t = mlir::cast<mlir::RankedTensorType>(v.getType());
      if (t.getElementType().isF32()) {
        return v;
      }
      return rw.create<ttir::TypecastOp>(
          loc, retypeTensor(t, t.getShape(), f32), v);
    };
    // An argument's chunks [k, 1, c] (a view in DRAM, cast after).
    auto view = [&](mlir::Value v) {
      auto t = mlir::cast<mlir::RankedTensorType>(v.getType());
      return asF32(reinterpret(rw, loc, v, withShape(t, {k, 1, c})));
    };
    auto full = [&](float value, int64_t cols) -> mlir::Value {
      return rw.create<ttir::FullOp>(
          loc, type(kRows, cols),
          llvm::SmallVector<int32_t>{static_cast<int32_t>(k),
                                     static_cast<int32_t>(kRows),
                                     static_cast<int32_t>(cols)},
          rw.getF32FloatAttr(value));
    };
    auto scalar = [&](float value) -> mlir::Value {
      return rw.create<ttir::FullOp>(
          loc, mlir::RankedTensorType::get({1, 1, 1}, f32),
          llvm::SmallVector<int32_t>{1, 1, 1}, rw.getF32FloatAttr(value));
    };
    auto place = [&](mlir::Value dst, mlir::Value src, int64_t to,
                     int64_t from, int64_t count) -> mlir::Value {
      auto t = mlir::cast<mlir::RankedTensorType>(dst.getType());
      return rw.create<ttir::PlaceRowsOp>(
          loc, t, dst, src, rw.getI64IntegerAttr(to),
          rw.getI64IntegerAttr(from), rw.getI64IntegerAttr(count));
    };
    auto columns = [&](mlir::Value v, bool max) -> mlir::Value {
      return rw.create<ttir::ColumnReduceOp>(loc, v.getType(), v,
                                             rw.getBoolAttr(max),
                                             rw.getI64IntegerAttr(kRows));
    };
    auto sum = [&](mlir::Value v, int32_t dim, mlir::RankedTensorType t) {
      return rw.create<ttir::SumOp>(loc, t, v, /*keep_dim=*/true,
                                    rw.getI32ArrayAttr({dim}));
    };
    auto mul = [&](mlir::Value a, mlir::Value b, mlir::RankedTensorType t) {
      return rw.create<ttir::MultiplyOp>(loc, t, a, b).getResult();
    };
    auto meanRsqrt = [&](mlir::Value total, float eps) -> mlir::Value {
      auto t = mlir::cast<mlir::RankedTensorType>(total.getType());
      mlir::Value mean =
          mul(total, scalar(1.0f / static_cast<float>(width)), t);
      mlir::Value var = rw.create<ttir::AddOp>(loc, t, mean, scalar(eps));
      return rw.create<ttir::RsqrtOp>(loc, t, var);
    };

    llvm::SmallVector<mlir::Value> x;
    for (mlir::Value s : sources) {
      x.push_back(view(s));
    }
    if (adaptor.getAddend()) {
      x.back() = rw.create<ttir::AddOp>(loc, type(1, c), x.back(),
                                        view(adaptor.getAddend()));
    }
    // Source j in rows j and n + j (row 0 of x_j broadcast down the rows;
    // the row of the same number moves, within its band); M has the query
    // in rows n.., so one row sum of X2 * M gives the squares (rows < n) and
    // the logits' dot products (rows n..).
    mlir::Value X2 = full(0.0f, c);
    for (int64_t j = 0; j < n; ++j) {
      X2 = place(X2, x[j], j, j, 1);
      X2 = place(X2, x[j], n + j, n + j, 1);
    }
    mlir::Value M = place(X2, view(adaptor.getQueries()), n, n, n);
    mlir::Value partial = sum(mul(X2, M, type(kRows, c)), 2, type(kRows, 1));
    mlir::Value total =
        rw.create<ttir::ChunkAllReduceOp>(loc, type(kRows, 1), partial);
    mlir::Value r = place(full(1.0f, 1),
                          meanRsqrt(total, op.getEpsRes().convertToFloat()), 0,
                          0, n);
    mlir::Value logits =
        place(full(-std::numeric_limits<float>::infinity(), 1),
              mul(place(total, total, 0, n, n), r, type(kRows, 1)), 0, 0, n);
    mlir::Value e = rw.create<ttir::ExpOp>(
        loc, type(kRows, 1),
        rw.create<ttir::SubtractOp>(loc, type(kRows, 1), logits,
                                    columns(logits, /*max=*/true)));
    mlir::Value p = rw.create<ttir::DivOp>(loc, type(kRows, 1), e,
                                           columns(e, /*max=*/false));
    if (op.getNormalize()) {
      p = mul(p, r, type(kRows, 1));
    }
    // p is 0 in rows n.. (exp(-inf)): X2's second copy adds nothing.
    mlir::Value mix = sum(mul(p, X2, type(kRows, c)), 1, type(1, c));
    auto outSplit = withShape(outType, {k, 1, c});
    mlir::Value y = mix;
    if (mlir::Value weight = adaptor.getWeight()) {
      mlir::Value squares = sum(mul(mix, mix, type(1, c)), 2, type(1, 1));
      mlir::Value rstd = meanRsqrt(
          rw.create<ttir::ChunkAllReduceOp>(loc, type(1, 1), squares),
          op.getEpsNorm().convertToFloat());
      // The last multiply FP32 with the result's encoding: D2M folds the
      // narrowing into the pack.
      y = mul(mul(mix, rstd, type(1, c)), view(weight),
              mlir::RankedTensorType::get(outSplit.getShape(), f32,
                                          outSplit.getEncoding()));
    }
    if (!outType.getElementType().isF32()) {
      y = rw.create<ttir::TypecastOp>(loc, outSplit, y);
    }
    rw.replaceOp(op, reinterpret(rw, loc, y, outType).getResult());
    return mlir::success();
  }
};

// Reshapes inside D2M subgraphs (head splits and merges; see
// rewriteHeadMajorSubgraph).
class TTNNToTTIRReshapePattern
    : public mlir::OpConversionPattern<mlir::tt::ttnn::ReshapeOp> {
  using OpConversionPattern::OpConversionPattern;

public:
  mlir::LogicalResult
  matchAndRewrite(mlir::tt::ttnn::ReshapeOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto type = mlir::cast<mlir::RankedTensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    rewriter.replaceOpWithNewOp<mlir::tt::ttir::ReshapeOp>(
        op, type, adaptor.getInput(), op.getShapeAttr());
    return mlir::success();
  }
};
// Tile-aligned column slices of whole rows inside D2M subgraphs,
// lowered to tile-offset views (libtt.tile_slice).
class TTNNToTTIRSlicePattern
    : public mlir::OpConversionPattern<mlir::tt::ttnn::SliceStaticOp> {
  using OpConversionPattern::OpConversionPattern;

public:
  mlir::LogicalResult
  matchAndRewrite(mlir::tt::ttnn::SliceStaticOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto type = mlir::cast<mlir::RankedTensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    bool columns = isTileColumnSlice(op);
    auto slice = rewriter.replaceOpWithNewOp<mlir::tt::ttir::SliceStaticOp>(
        op, type, adaptor.getInput(), op.getBeginsAttr(), op.getEndsAttr(),
        op.getStepAttr());
    if (columns) {
      slice->setAttr("libtt.tile_slice", rewriter.getUnitAttr());
    }
    return mlir::success();
  }
};
// Last-dim concats and leading-dim repeats inside D2M subgraphs (split over
// cores: splitWideRows).
class TTNNToTTIRConcatPattern
    : public mlir::OpConversionPattern<mlir::tt::ttnn::ConcatOp> {
  using OpConversionPattern::OpConversionPattern;

public:
  mlir::LogicalResult
  matchAndRewrite(mlir::tt::ttnn::ConcatOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto type = mlir::cast<mlir::RankedTensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    rewriter.replaceOpWithNewOp<mlir::tt::ttir::ConcatOp>(
        op, type, adaptor.getInputs(), op.getDim());
    return mlir::success();
  }
};
class TTNNToTTIRRepeatPattern
    : public mlir::OpConversionPattern<mlir::tt::ttnn::RepeatOp> {
  using OpConversionPattern::OpConversionPattern;

public:
  mlir::LogicalResult
  matchAndRewrite(mlir::tt::ttnn::RepeatOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto type = mlir::cast<mlir::RankedTensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    rewriter.replaceOpWithNewOp<mlir::tt::ttir::RepeatOp>(
        op, type, adaptor.getInput(),
        rewriter.getDenseI64ArrayAttr(op.getRepeatDims().getShape()));
    return mlir::success();
  }
};
} // namespace

void populateD2MSubgraphPatterns(MLIRContext *ctx, RewritePatternSet &patterns,
                                 TypeConverter &typeConverter) {
  patterns.add<TTNNToTTIRReshapePattern, TTNNToTTIRSlicePattern,
               TTNNToTTIRConcatPattern, TTNNToTTIRRepeatPattern,
               TTNNNormDecompositionPattern<mlir::tt::ttnn::RMSNormOp>,
               TTNNNormDecompositionPattern<mlir::tt::ttnn::RMSNormGatedOp>,
               TTNNAttnResDecompositionPattern>(typeConverter, ctx);
}

// Head-major form for per-head row ops in decode.
//
// A subgraph that splits [T, H*D] into [T, H, D] heads, computes elementwise
// ops and last-dim reductions, and merges the heads back moves data twice in
// TTNN's tile layout (each leading index pads to whole tiles). Computed as
// [H, T, D] instead (every rank-3 value has dims 0 and 1 swapped), the split
// and the merge keep every tile and its order when T <= 32 and D is a multiple
// of 32: they become tile reinterpretations (libtt.tile_reinterpret reshapes,
// lowered to views below). A per-token [T] factor becomes [1, T, 1], a
// transpose of one tile.
static RankedTensorType withShape(RankedTensorType t, ArrayRef<int64_t> shape) {
  Attribute encoding = t.getEncoding();
  if (auto layout =
          mlir::dyn_cast_if_present<mlir::tt::ttnn::TTNNLayoutAttr>(encoding)) {
    encoding = mlir::tt::ttnn::TTNNLayoutAttr::Builder(layout, shape).build();
  }
  return RankedTensorType::get(shape, t.getElementType(), encoding);
}

// A reshape to `to` that keeps every tile in place (a view).
static mlir::tt::ttir::ReshapeOp reinterpret(mlir::OpBuilder &b,
                                             mlir::Location loc, mlir::Value v,
                                             RankedTensorType to) {
  auto r = b.create<mlir::tt::ttir::ReshapeOp>(
      loc, to, v,
      b.getI32ArrayAttr(llvm::to_vector_of<int32_t>(to.getShape())));
  r->setAttr("libtt.tile_reinterpret", b.getUnitAttr());
  return r;
}

// Makes `r` such a reshape, to `to`.
static void reinterpretInPlace(mlir::tt::ttir::ReshapeOp r,
                               RankedTensorType to) {
  mlir::Builder b(r.getContext());
  r.getResult().setType(to);
  r.setShapeAttr(b.getI32ArrayAttr(llvm::to_vector_of<int32_t>(to.getShape())));
  r->setAttr("libtt.tile_reinterpret", b.getUnitAttr());
}

// Gives `op`'s results their types in a rewritten subgraph's layout
// (a ttir.full also its shape).
static void
retypeResults(Operation &op,
              llvm::function_ref<RankedTensorType(RankedTensorType)> layout) {
  for (Value r : op.getResults()) {
    r.setType(layout(mlir::cast<RankedTensorType>(r.getType())));
  }
  if (auto full = mlir::dyn_cast<mlir::tt::ttir::FullOp>(op)) {
    auto t = mlir::cast<RankedTensorType>(full.getType());
    full.setShapeAttr(
        mlir::Builder(op.getContext())
            .getDenseI32ArrayAttr(llvm::to_vector_of<int32_t>(t.getShape())));
  }
}

// A per-token [T] or [1, T] row as the [T, 1] column: the same tile
// reinterpreted, then row 0 moved to the columns in DST (ttir.row_to_column),
// so the per-token factor needs no TTNN reshape before the subgraph.
static mlir::Value rowToColumn(mlir::OpBuilder &b, mlir::Location loc,
                               mlir::Value row, int64_t T) {
  auto colType = withShape(mlir::cast<RankedTensorType>(row.getType()), {T, 1});
  return b.create<mlir::tt::ttir::RowToColumnOp>(
      loc, colType, reinterpret(b, loc, row, colType));
}

std::optional<HeadDims> headMerge(ArrayRef<int64_t> heads,
                                  ArrayRef<int64_t> merged) {
  if (heads.size() != 3 || merged.size() != 2) {
    return std::nullopt;
  }
  int64_t T = heads[0], H = heads[1], D = heads[2];
  if (T > 32 || (D % 32 != 0 && D != 16) || merged[0] != T ||
      merged[1] != H * D) {
    return std::nullopt;
  }
  return HeadDims{T, H, D};
}

std::optional<HeadReshape>
classifyHeadReshape(ArrayRef<int64_t> in, ArrayRef<int64_t> out, HeadDims h) {
  if (out.size() != 3 || in.empty() || in.size() > 2) {
    return std::nullopt;
  }
  int64_t rows = in.size() == 2 ? in[0] : 1, cols = in.back();
  auto is = [&](ArrayRef<int64_t> shape) { return out == shape; };
  if (is({h.T, h.H, h.D}) && rows == h.T && cols == h.H * h.D) {
    return HeadReshape::Split;
  }
  if (is({1, h.H, h.D}) && rows == 1 && cols == h.H * h.D) {
    return HeadReshape::Scale;
  }
  if (is({h.T, 1, 1}) && in.size() == 2 && rows == h.T && cols == 1) {
    return HeadReshape::Token;
  }
  if (is({h.T, 1, 1}) && rows == 1 && cols == h.T) {
    return HeadReshape::TokenRow;
  }
  if (is({1, 1, h.D}) && rows == 1 && cols == h.D && h.D % 32 == 0) {
    return HeadReshape::Row;
  }
  return std::nullopt;
}

bool isHeadValue(ArrayRef<int64_t> shape, HeadDims h) {
  return shape.size() == 3 &&
         ((shape[1] == h.H && (shape[2] == h.D || shape[2] == 1)) ||
          (shape[1] == 1 && shape[2] == 1));
}

// Group-major form for heads narrower than a tile (D = 16). Values stay in [T,
// H*D] (rank-3 [a, H, D] and [a, H, 1] become [a, H*D], [a, 1, 1] becomes [a,
// 1]): a per-head value [a, H, 1] is kept replicated across its head's D
// columns, so per-head broadcasts are plain elementwise ops, and a last-dim sum
// is a group sum (the sum of the D columns of each head, replicated),
// tile-local because D divides 32. The split and merge reshapes disappear.
static bool rewriteGroupMajor(
    func::FuncOp func, mlir::tt::ttir::ReshapeOp outReshape,
    ArrayRef<std::pair<mlir::tt::ttir::ReshapeOp, HeadReshape>> inputs,
    HeadDims h) {
  using namespace mlir::tt;
  auto flat = [&](RankedTensorType t) -> std::optional<RankedTensorType> {
    if (t.getRank() != 3) {
      return t;
    }
    if (!isHeadValue(t.getShape(), h)) {
      return std::nullopt;
    }
    int64_t a = t.getDimSize(0);
    return t.getDimSize(1) == h.H ? withShape(t, {a, h.H * h.D})
                                  : withShape(t, {a, 1});
  };
  Block &body = func.getBody().front();
  // Check first: every rank-3 value must map; sums keep the last dim.
  for (Operation &op : body.without_terminator()) {
    if (&op == outReshape.getOperation()) {
      continue;
    }
    for (Value r : op.getResults()) {
      if (!flat(mlir::cast<RankedTensorType>(r.getType()))) {
        return false;
      }
    }
    if (auto sum = mlir::dyn_cast<ttir::SumOp>(op); sum && !sum.getKeepDim()) {
      return false;
    }
  }
  mlir::OpBuilder b(func.getContext());
  b.setInsertionPointToStart(&body);
  // Rank-3 arguments are single elements ([1, 1, 1] -> [1, 1], one tile).
  for (BlockArgument arg : body.getArguments()) {
    auto t = mlir::cast<RankedTensorType>(arg.getType());
    if (t.getRank() != 3 || arg.use_empty()) {
      continue;
    }
    auto scalar = reinterpret(b, arg.getLoc(), arg, withShape(t, {1, 1}));
    arg.replaceAllUsesExcept(scalar.getResult(), scalar);
  }
  for (auto [reshape, kind] : inputs) {
    auto in = mlir::cast<RankedTensorType>(reshape.getInput().getType());
    if (kind == HeadReshape::TokenRow) {
      b.setInsertionPoint(reshape);
      reshape.getResult().replaceAllUsesWith(
          rowToColumn(b, reshape.getLoc(), reshape.getInput(), h.T));
      reshape.erase();
      continue;
    }
    if (in.getRank() == 2) {
      reshape.getResult().replaceAllUsesWith(reshape.getInput());
      reshape.erase();
      continue;
    }
    // [H*D] (one tile row) -> [1, H*D]: the same tiles.
    reinterpretInPlace(
        reshape, withShape(mlir::cast<RankedTensorType>(reshape.getType()),
                           {1, in.getDimSize(0)}));
  }
  for (Operation &op : llvm::make_early_inc_range(body.without_terminator())) {
    if (&op == outReshape.getOperation() || mlir::isa<ttir::ReshapeOp>(op)) {
      continue;
    }
    if (auto sum = mlir::dyn_cast<ttir::SumOp>(op)) {
      b.setInsertionPoint(sum);
      auto type = *flat(mlir::cast<RankedTensorType>(sum.getType()));
      auto groupSum = b.create<ttir::GroupSumOp>(
          sum.getLoc(), type, sum.getInput(), b.getI64IntegerAttr(h.D));
      sum.getResult().replaceAllUsesWith(groupSum.getResult());
      sum.erase();
      continue;
    }
    retypeResults(op, [&](RankedTensorType t) { return *flat(t); });
  }
  // The merge: the last value already is [T, H*D]; give it the result type.
  Value last = outReshape.getInput();
  last.setType(outReshape.getType());
  outReshape.getResult().replaceAllUsesWith(last);
  outReshape.erase();
  func->setAttr("libtt.group_major", b.getUnitAttr());
  return true;
}

// Wide rows split over cores (intra-chip SPMD along the last dim).
//
// A subgraph of decode rows [B.., T, X] (T <= 32 rows, B <= 8 rows of rows)
// whose width W (the widest row) is more than a core's share computes as k
// chunks of c = W / k columns, chunk i on core i: each row of rows b is a
// value [k, T, c] (the same tiles, chunk-major; the batch unrolled), a row
// value [.., T, 1] or a narrow one [.., T, X] (X a few tiles) is computed on
// every core ([k, T, X]; arguments' and constants' broadcast: [1, T, X]).
// Elementwise ops apply per chunk; a sum or mean over W is the chunk's sum
// and ttir.chunk_allreduce (the cores sum their partials inside the kernel);
// a matmul with a weight argument contracts its chunk of K = W (or of each
// segment of a concat of width rows: [a | b] x D = a D_0 + b D_1) into an
// FP32 partial summed the same way, or (a narrow A, N = W) makes its chunk
// of the columns; leading-dim slices, repeats and reshapes pick parts.
// Without reductions (an elementwise row of more than 8 tiles: Inkling 2's
// attention output gate) k <= 8 spreads the row over one core row; with
// them (more than 32 tiles: Inkling 2's 2304-wide norms, its loop-message
// controller) k is wideNormChunks's, one chunk per core.
static bool splitWideRows(func::FuncOp func) {
  namespace ttir = mlir::tt::ttir;
  if (func.getBody().getBlocks().size() != 1) {
    return false;
  }
  Block &body = func.getBody().front();
  if (llvm::any_of(body, [](Operation &op) {
        return mlir::isa<ttir::ChunkAllReduceOp>(op);
      })) {
    return false;
  }
  // [B.., T, X] as (B, T, X): the leading dims' product, rows, columns.
  struct Form {
    int64_t batch = 1, rows = 1, cols = 1;
  };
  auto formOf = [](Type type) -> std::optional<Form> {
    auto t = mlir::dyn_cast<RankedTensorType>(type);
    if (!t) {
      return std::nullopt;
    }
    Form f;
    if (t.getRank() == 0) {
      return f;
    }
    f.cols = t.getShape().back();
    if (t.getRank() >= 2) {
      f.rows = t.getDimSize(t.getRank() - 2);
    }
    for (int64_t d = 0; d + 2 < t.getRank(); ++d) {
      f.batch *= t.getDimSize(d);
    }
    return f;
  };
  auto sliceSource = [](BlockArgument arg) {
    return !arg.use_empty() && llvm::all_of(arg.getUsers(), [](Operation *u) {
      auto s = mlir::dyn_cast<ttir::SliceStaticOp>(u);
      return s && s->hasAttr("libtt.tile_slice");
    });
  };
  // A matmul's weight, viewed per chunk at the matmul.
  auto weight = [](Value v) {
    return mlir::isa<BlockArgument>(v) && !v.use_empty() &&
           llvm::all_of(v.getUses(), [](OpOperand &use) {
             return mlir::isa<ttir::MatmulOp>(use.getOwner()) &&
                    use.getOperandNumber() == 1;
           });
  };
  int64_t T = 1, W = 1;
  auto visit = [&](Type type) {
    if (auto f = formOf(type)) {
      T = std::max(T, f->rows);
      W = std::max(W, f->cols);
    }
  };
  for (BlockArgument arg : body.getArguments()) {
    if (!sliceSource(arg) && !weight(arg)) {
      visit(arg.getType());
    }
  }
  bool exchanges = false;
  // Concats of rows (and their reshapes) are as wide as their segments.
  llvm::DenseSet<Value> segmented;
  for (Operation &op : body) {
    if (mlir::isa<ttir::ConcatOp>(op) ||
        (mlir::isa<ttir::ReshapeOp>(op) &&
         segmented.contains(op.getOperand(0)))) {
      segmented.insert(op.getResult(0));
    } else {
      for (Type t : op.getResultTypes()) {
        visit(t);
      }
    }
    if (mlir::isa<ttir::SumOp, ttir::MeanOp, ttir::MatmulOp>(op)) {
      exchanges = true;
    }
  }
  if (T > 32 || W % 32 != 0 || W / 32 <= 8) {
    return false;
  }
  const int64_t tiles = W / 32;
  int64_t k = 0;
  if (exchanges) {
    if (auto chunks = wideNormChunks(
            func.getOperation(),
            RankedTensorType::get({T, W}, Float32Type::get(func.getContext())))) {
      k = *chunks;
    }
  } else {
    for (int64_t d = 2; d <= 8 && !k; ++d) {
      if (tiles % d == 0 && tiles / d <= 8) {
        k = d;
      }
    }
  }
  if (!k) {
    return false;
  }
  const int64_t c = W / k;
  MLIRContext *ctx = func.getContext();
  mlir::OpBuilder b(ctx);
  SmallVector<Operation *> created;
  auto rollBack = [&]() {
    for (Operation *op : llvm::reverse(created)) {
      op->erase();
    }
    return false;
  };
  auto record = [&](Operation *op) {
    created.push_back(op);
    return op;
  };
  // Rows of rows (a batch B > 1, B T <= 32) pack into the tile rows: rows of
  // rows b in rows [b T, b T + T) of every value (R = B T rows). A value is
  // Packed (each b in its rows), Every (one value for all b: computed from
  // batchless ones; with T = 1 a single row that broadcasts) or Picked (rows
  // of rows `index` of a packed value: row-wise ops apply to the packed value
  // and the pick happens at the results, so per-b ops of one packed value
  // become one, CSE'd: Inkling 2's per-query norms, one exchange).
  int64_t B = 1;
  for (BlockArgument arg : body.getArguments()) {
    if (!sliceSource(arg) && !weight(arg)) {
      B = std::max(B, formOf(arg.getType())->batch);
    }
  }
  for (Operation &op : body) {
    for (Value r : op.getResults()) {
      if (!segmented.contains(r)) {
        B = std::max(B, formOf(r.getType())->batch);
      }
    }
  }
  const int64_t R = B * T;
  if (R > 32 || B > 8) {
    return false;
  }
  enum class Kind { Packed, Every, Picked };
  struct Val {
    Value v;
    Kind kind = Kind::Every;
    SmallVector<Value> segments;
    int64_t index = 0;
  };
  llvm::DenseMap<Value, Val> vals;
  auto rowsOf = [](Value v) {
    return mlir::cast<RankedTensorType>(v.getType()).getShape()[1];
  };
  auto place = [&](Location loc, Value dst, Value src, int64_t to,
                   int64_t from, int64_t count) -> Value {
    return record(b.create<ttir::PlaceRowsOp>(
                      loc, dst.getType(), dst, src, b.getI64IntegerAttr(to),
                      b.getI64IntegerAttr(from), b.getI64IntegerAttr(count)))
        ->getResult(0);
  };
  // R rows of zeros of v's columns.
  auto zeros = [&](Location loc, Value v) -> Value {
    auto t = mlir::cast<RankedTensorType>(v.getType());
    auto to = withShape(t, {t.getShape()[0], R, t.getShape()[2]});
    SmallVector<int32_t> shape(to.getShape().begin(), to.getShape().end());
    return record(b.create<ttir::FullOp>(loc, to, shape,
                                         b.getZeroAttr(t.getElementType())))
        ->getResult(0);
  };
  // Rows of rows b (from rows [from_b, from_b + T) of parts[b]) packed.
  auto pack = [&](Location loc, ArrayRef<std::pair<Value, int64_t>> parts) {
    Value acc = rowsOf(parts[0].first) == R ? parts[0].first
                                             : zeros(loc, parts[0].first);
    for (auto [i, part] : llvm::enumerate(parts)) {
      int64_t to = static_cast<int64_t>(i) * T;
      if (acc != part.first || part.second != to) {
        acc = place(loc, acc, part.first, to, part.second, T);
      }
    }
    return acc;
  };
  // Where rows of rows i's data sits in x (Every: rows [0, T)).
  auto from = [&](const Val &x, int64_t i) {
    return x.kind == Kind::Packed   ? i * T
           : x.kind == Kind::Picked ? x.index * T
                                    : 0;
  };
  // A Picked value in every rows of rows (Every); one row broadcasts.
  auto spread = [&](Location loc, const Val &x) -> Value {
    if (x.kind != Kind::Picked || rowsOf(x.v) == 1) {
      return x.v;
    }
    SmallVector<std::pair<Value, int64_t>> parts(B, {x.v, x.index * T});
    return pack(loc, parts);
  };
  // The tile view of an argument's rows of rows `index` (chunk-major for the
  // width) as R rows (or its own rows): output tile (j, r, x) <- input tile
  // (index.., r, j c/32 + x).
  auto argView = [&](Value arg, int64_t index, RankedTensorType to,
                     bool chunked) -> Value {
    auto t = mlir::cast<RankedTensorType>(arg.getType());
    SmallVector<AffineExpr> results;
    SmallVector<int64_t> lead(t.getShape().begin(),
                              t.getShape().end() - std::min<int64_t>(2, t.getRank()));
    SmallVector<int64_t> coords(lead.size(), 0);
    for (int64_t d = static_cast<int64_t>(lead.size()) - 1, rest = index;
         d >= 0; --d) {
      coords[d] = rest % lead[d];
      rest /= lead[d];
    }
    for (int64_t x : coords) {
      results.push_back(getAffineConstantExpr(x, ctx));
    }
    AffineExpr j = getAffineDimExpr(0, ctx), r = getAffineDimExpr(1, ctx),
               col = getAffineDimExpr(2, ctx);
    if (t.getRank() >= 2) {
      results.push_back(r);
    }
    results.push_back(chunked ? j * (c / 32) + col : col);
    return record(b.create<ttir::TileViewOp>(
                      arg.getLoc(), to, arg,
                      AffineMapAttr::get(AffineMap::get(3, 0, results, ctx))))
        ->getResult(0);
  };
  // Arguments (not slice sources or weights): views. One row (a norm
  // weight; with T = 1 any row) broadcasts; T rows of a batchless argument
  // repeat in every rows of rows; a batched one packs.
  b.setInsertionPointToStart(&body);
  for (BlockArgument arg : body.getArguments()) {
    if (arg.use_empty() || sliceSource(arg) || weight(arg)) {
      continue;
    }
    auto t = mlir::cast<RankedTensorType>(arg.getType());
    Form f = *formOf(t);
    if ((f.cols != W && f.cols != 1) || (f.rows != 1 && f.rows != T) ||
        (f.batch != 1 && f.batch != B) || (f.batch > 1 && f.rows != T)) {
      return rollBack();
    }
    const bool wide = f.cols == W;
    const int64_t rows = f.rows == 1 && f.batch == 1 ? 1 : R;
    auto to = withShape(t, {wide ? k : 1, rows, wide ? c : 1});
    Location loc = arg.getLoc();
    Val x;
    if (rows == f.rows) {
      x.v = record(reinterpret(b, loc, arg, to))->getResult(0);
    } else if (f.batch == 1) {
      x.v = spread(loc, {argView(arg, 0, to, wide), Kind::Picked, {}, 0});
    } else {
      SmallVector<std::pair<Value, int64_t>> parts;
      for (int64_t i = 0; i < f.batch; ++i) {
        parts.push_back({argView(arg, i, to, wide), 0});
      }
      x.v = pack(loc, parts);
      x.kind = Kind::Packed;
    }
    vals[arg] = x;
  }
  auto chunkedValue = [&](Value v) {
    return mlir::cast<RankedTensorType>(v.getType()).getShape()[0] == k &&
           k > 1;
  };
  SmallVector<Operation *> originals;
  llvm::SmallPtrSet<Operation *, 4> kept;
  for (Operation &op : body.without_terminator()) {
    if (llvm::is_contained(created, &op)) {
      continue;
    }
    originals.push_back(&op);
    b.setInsertionPoint(&op);
    Location loc = op.getLoc();
    auto lookup = [&](Value v) -> std::optional<Val> {
      auto it = vals.find(v);
      if (it == vals.end()) {
        return std::nullopt;
      }
      return it->second;
    };
    auto resultType = mlir::cast<RankedTensorType>(op.getResult(0).getType());
    std::optional<Form> rf = formOf(resultType);
    if (!rf) {
      return rollBack();
    }
    if (mlir::isa<ttir::FullOp>(op)) {
      if (resultType.getNumElements() != 1) {
        return rollBack();
      }
      Operation *n = record(b.clone(op));
      n->getResult(0).setType(withShape(resultType, {1, 1, 1}));
      retypeResults(*n, [](RankedTensorType t) { return t; });
      vals[op.getResult(0)] = {n->getResult(0), Kind::Every, {}};
      continue;
    }
    if (auto slice = mlir::dyn_cast<ttir::SliceStaticOp>(op)) {
      if (slice->hasAttr("libtt.tile_slice")) {
        // A column slice of an argument, read in place.
        if (!mlir::isa<BlockArgument>(slice.getInput()) || rf->cols != W ||
            B != 1) {
          return rollBack();
        }
        kept.insert(&op);
        b.setInsertionPointAfter(slice);
        vals[slice.getResult()] = {
            record(reinterpret(b, loc, slice.getResult(),
                               withShape(resultType, {k, rf->rows, c})))
                ->getResult(0),
            Kind::Every,
            {}};
        continue;
      }
      // Leading dims only: one rows of rows picked.
      std::optional<Val> in = lookup(slice.getInput());
      auto inType = slice.getInput().getType();
      auto begins = extractFromIntegerArrayAttr<int32_t>(slice.getBegins());
      auto steps = extractFromIntegerArrayAttr<int32_t>(slice.getStep());
      const int64_t rank = inType.getRank();
      if (!in || !in->segments.empty() || rank < 3 || rf->batch != 1 ||
          rf->rows != inType.getDimSize(rank - 2) ||
          rf->cols != inType.getDimSize(rank - 1) ||
          llvm::any_of(steps, [](int32_t x) { return x != 1; })) {
        return rollBack();
      }
      int64_t index = 0;
      for (int64_t d = 0; d < rank - 2; ++d) {
        index = index * inType.getDimSize(d) + begins[d];
      }
      Val x = *in;
      if (in->kind == Kind::Packed) {
        x.kind = Kind::Picked;
        x.index = index;
      } else if (in->kind == Kind::Picked) {
        return rollBack();
      }
      vals[slice.getResult()] = x;
      continue;
    }
    if (auto repeat = mlir::dyn_cast<ttir::RepeatOp>(op)) {
      std::optional<Val> in = lookup(repeat.getInput());
      ArrayRef<int64_t> dims = repeat.getRepeatDimensions();
      if (!in || !in->segments.empty() || in->kind == Kind::Packed ||
          dims.size() < 2 || dims.take_back(2) != ArrayRef<int64_t>{1, 1}) {
        return rollBack();
      }
      vals[repeat.getResult()] = {spread(loc, *in), Kind::Every, {}};
      continue;
    }
    if (auto reshape = mlir::dyn_cast<ttir::ReshapeOp>(op)) {
      std::optional<Val> in = lookup(reshape.getInput());
      std::optional<Form> f = formOf(reshape.getInput().getType());
      if (!in || f->rows != rf->rows || f->cols != rf->cols ||
          (in->kind == Kind::Packed && f->batch != rf->batch)) {
        return rollBack();
      }
      vals[reshape.getResult()] = *in;
      continue;
    }
    if (auto concat = mlir::dyn_cast<ttir::ConcatOp>(op)) {
      const int64_t rank = resultType.getRank();
      int64_t dim =
          concat.getDim() < 0 ? concat.getDim() + rank : concat.getDim();
      if (dim + 2 < rank &&
          llvm::all_of(resultType.getShape().take_front(dim),
                       [](int64_t x) { return x == 1; })) {
        // Rows of rows stacked (a repeat, decomposed).
        SmallVector<std::pair<Value, int64_t>> parts;
        bool same = true;
        std::optional<Val> first = lookup(concat.getInputs().front());
        for (Value v : concat.getInputs()) {
          std::optional<Val> in = lookup(v);
          if (!in || !in->segments.empty() || in->kind == Kind::Packed) {
            return rollBack();
          }
          same &= in->v == first->v;
          for (int64_t i = 0, n = formOf(v.getType())->batch; i < n; ++i) {
            parts.push_back({spread(loc, *in), 0});
          }
        }
        vals[concat.getResult()] =
            same ? Val{spread(loc, *first), Kind::Every, {}}
                 : Val{pack(loc, parts), Kind::Packed, {}};
        continue;
      }
      if (dim != rank - 1) {
        return rollBack();
      }
      Val x;
      x.kind = Kind::Every;
      for (Value v : concat.getInputs()) {
        std::optional<Val> in = lookup(v);
        if (!in || !in->segments.empty() || formOf(v.getType())->cols != W) {
          return rollBack();
        }
        if (in->kind == Kind::Packed) {
          x.kind = Kind::Packed;
        }
      }
      for (Value v : concat.getInputs()) {
        x.segments.push_back(spread(loc, *lookup(v)));
      }
      vals[concat.getResult()] = x;
      continue;
    }
    if (mlir::isa<ttir::SumOp, ttir::MeanOp>(op)) {
      std::optional<Val> in = lookup(op.getOperand(0));
      auto dims = op.getAttrOfType<ArrayAttr>("dim_arg");
      auto keep = op.getAttrOfType<BoolAttr>("keep_dim");
      auto inType = mlir::cast<RankedTensorType>(op.getOperand(0).getType());
      if (!in || !in->segments.empty() || !dims || dims.size() != 1 || !keep ||
          !keep.getValue()) {
        return rollBack();
      }
      int64_t d = mlir::cast<IntegerAttr>(dims[0]).getInt();
      if (d != -1 && d != inType.getRank() - 1) {
        return rollBack();
      }
      auto shape = mlir::cast<RankedTensorType>(in->v.getType()).getShape();
      auto rowType = withShape(resultType, {shape[0], shape[1], 1});
      Value r;
      if (inType.getShape().back() == W) {
        // The chunk's sum, summed over the chunks (a mean divides by W).
        Value s = record(b.create<ttir::SumOp>(loc, rowType, in->v,
                                               /*keep_dim=*/true,
                                               b.getI32ArrayAttr({2})))
                      ->getResult(0);
        r = record(b.create<ttir::ChunkAllReduceOp>(loc, rowType, s))
                ->getResult(0);
        if (mlir::isa<ttir::MeanOp>(op)) {
          auto f32 = RankedTensorType::get({1, 1, 1}, b.getF32Type());
          Value inv =
              record(b.create<ttir::FullOp>(
                         loc, f32, SmallVector<int32_t>{1, 1, 1},
                         b.getF32FloatAttr(1.0f / static_cast<float>(W))))
                  ->getResult(0);
          r = record(b.create<ttir::MultiplyOp>(loc, rowType, r, inv))
                  ->getResult(0);
        }
      } else {
        IRMapping m;
        m.map(op.getOperand(0), in->v);
        Operation *n = record(b.clone(op, m));
        n->getResult(0).setType(rowType);
        n->setAttr("dim_arg", b.getI32ArrayAttr({2}));
        r = n->getResult(0);
      }
      vals[op.getResult(0)] = {r, in->kind, {}, in->index};
      continue;
    }
    if (auto mm = mlir::dyn_cast<ttir::MatmulOp>(op)) {
      std::optional<Val> a = lookup(mm.getA());
      Value w = mm.getB();
      if (!a || !weight(w) || mm.getTransposeA() || mm.getTransposeB()) {
        return rollBack();
      }
      auto wt = mlir::cast<RankedTensorType>(w.getType());
      Form wf = *formOf(wt);
      const int64_t K = wt.getDimSize(wt.getRank() - 2), N = wf.cols;
      if ((wf.batch != 1 && wf.batch != B) || K % 32 != 0 || N % 32 != 0) {
        return rollBack();
      }
      SmallVector<Value> segments =
          !a->segments.empty() ? a->segments : SmallVector<Value>{a->v};
      auto aShape =
          mlir::cast<RankedTensorType>(segments.front().getType()).getShape();
      const bool contractsWidth =
          aShape[2] == c && K == static_cast<int64_t>(segments.size()) * W;
      const bool makesWidth = a->segments.empty() && aShape[2] == K &&
                              K != W && N == W && chunkedValue(a->v);
      if (!contractsWidth && !makesWidth) {
        return rollBack();
      }
      AffineExpr j = getAffineDimExpr(0, ctx), r = getAffineDimExpr(1, ctx),
                 x = getAffineDimExpr(2, ctx);
      // Rows of rows i's matmul (weight i, or the one weight).
      auto product = [&](int64_t i) -> Value {
        SmallVector<int64_t> lead(wt.getShape().drop_back(2));
        SmallVector<AffineExpr> coords;
        for (int64_t d = static_cast<int64_t>(lead.size()) - 1, rest = i;
             d >= 0; --d) {
          coords.insert(coords.begin(),
                        getAffineConstantExpr(rest % lead[d], ctx));
          rest /= lead[d];
        }
        auto view = [&](SmallVector<int64_t> shape, AffineExpr row,
                        AffineExpr col) {
          SmallVector<AffineExpr> results(coords);
          results.append({row, col});
          return record(b.create<ttir::TileViewOp>(
                            loc, withShape(wt, shape), w,
                            AffineMapAttr::get(
                                AffineMap::get(3, 0, results, ctx))))
              ->getResult(0);
        };
        if (makesWidth) {
          // Narrow rows times the width: this chunk's columns of D.
          Value d = view({k, K, c}, r, j * (c / 32) + x);
          Operation *m = record(b.create<ttir::MatmulOp>(
              loc, withShape(resultType, {k, rowsOf(a->v), c}), a->v, d,
              /*transpose_a=*/false, /*transpose_b=*/false));
          m->setAttr("libtt.chunk_matmul", b.getUnitAttr());
          return m->getResult(0);
        }
        // Contracting the width: per segment, this chunk's rows of D, FP32.
        // A one-row segment among packed ones broadcasts down the rows first
        // (the same tile matmul; the products then add tile for tile, so the
        // second matmul can accumulate into the first's tiles).
        int64_t most = 1;
        for (Value seg : segments) {
          most = std::max(most, rowsOf(seg));
        }
        Value sum;
        for (auto [s, segment] : llvm::enumerate(segments)) {
          Value seg = segment;
          if (rowsOf(seg) == 1 && most > 1) {
            auto st = mlir::cast<RankedTensorType>(seg.getType());
            seg = record(b.create<ttir::BroadcastOp>(
                             loc, withShape(st, {k, most, c}), seg,
                             b.getDenseI64ArrayAttr({1, most, 1})))
                      ->getResult(0);
          }
          Value d = view({k, c, N},
                         j * (c / 32) + r + static_cast<int64_t>(s) * (W / 32),
                         x);
          auto partial = retypeTensor(resultType, {k, rowsOf(seg), N},
                                      b.getF32Type());
          Operation *m = record(b.create<ttir::MatmulOp>(
              loc, partial, seg, d, /*transpose_a=*/false,
              /*transpose_b=*/false));
          m->setAttr("libtt.chunk_matmul", b.getUnitAttr());
          if (sum) {
            auto both = retypeTensor(
                resultType, {k, std::max(rowsOf(sum), rowsOf(seg)), N},
                b.getF32Type());
            sum = record(b.create<ttir::AddOp>(loc, both, sum,
                                               m->getResult(0)))
                      ->getResult(0);
          } else {
            sum = m->getResult(0);
          }
        }
        return sum;
      };
      Val out{Value(), a->kind, {}, a->index};
      if (wf.batch == 1) {
        out.v = product(0);
      } else {
        // Batch i's rows from its product: where A holds batch i's rows
        // (a packed A), or A's own rows.
        SmallVector<std::pair<Value, int64_t>> parts;
        for (int64_t i = 0; i < B; ++i) {
          parts.push_back({product(i), from(*a, i)});
        }
        out.v = pack(loc, parts);
        out.kind = Kind::Packed;
      }
      if (contractsWidth) {
        // The sum over the chunks, narrowed (if the result is) by its pack:
        // FP32 values in the result's layout, then a typecast D2M folds (an
        // SFPU narrowing typecast garbled the values).
        auto shape = mlir::cast<RankedTensorType>(out.v.getType()).getShape();
        auto narrow = withShape(resultType, shape);
        auto total = RankedTensorType::get(shape, b.getF32Type(),
                                           narrow.getEncoding());
        Operation *exchange =
            record(b.create<ttir::ChunkAllReduceOp>(loc, total, out.v));
        // Whole tiles (every column), not row sums.
        exchange->setAttr("libtt.full_tile", b.getUnitAttr());
        out.v = exchange->getResult(0);
        if (!resultType.getElementType().isF32()) {
          out.v = record(b.create<ttir::TypecastOp>(loc, narrow, out.v))
                      ->getResult(0);
        }
      }
      vals[mm.getResult()] = out;
      continue;
    }
    if (!mlir::isa<ttir::ElementwiseUnary, ttir::ElementwiseBinary,
                   ttir::TypecastOp, ttir::WhereOp>(op) ||
        op.getNumResults() != 1 || op.getNumRegions() != 0) {
      return rollBack();
    }
    // Elementwise: Packed if any operand is (Picked ones spread), else
    // Picked if all picked operands pick the same rows of rows (spread
    // otherwise), else Every; the broadcast of the operands' shapes.
    Kind kind = Kind::Every;
    std::optional<int64_t> picked;
    bool mixedPicks = false;
    for (Value v : op.getOperands()) {
      std::optional<Val> in = lookup(v);
      if (!in || !in->segments.empty()) {
        return rollBack();
      }
      if (in->kind == Kind::Packed) {
        kind = Kind::Packed;
      } else if (in->kind == Kind::Picked) {
        mixedPicks |= picked && *picked != in->index;
        picked = in->index;
      }
    }
    if (kind != Kind::Packed && picked && !mixedPicks) {
      kind = Kind::Picked;
    }
    if (formOf(resultType)->cols != W && formOf(resultType)->cols != 1 &&
        !llvm::any_of(op.getOperands(),
                      [&](Value v) { return chunkedValue(lookup(v)->v); })) {
      return rollBack();
    }
    IRMapping m;
    SmallVector<int64_t> shape{1, 1, 1};
    for (Value v : op.getOperands()) {
      Val in = *lookup(v);
      Value part = kind == Kind::Picked ? in.v : spread(loc, in);
      m.map(v, part);
      for (auto [d, n] : llvm::enumerate(
               mlir::cast<RankedTensorType>(part.getType()).getShape())) {
        shape[d] = std::max(shape[d], n);
      }
    }
    Operation *clone = record(b.clone(op, m));
    clone->getResult(0).setType(withShape(resultType, shape));
    vals[op.getResult(0)] = {clone->getResult(0), kind, {},
                             picked.value_or(0)};
  }
  // The results: width rows, one rows of rows (its rows [0, T)), back to
  // the result's layout.
  auto ret = mlir::cast<func::ReturnOp>(body.getTerminator());
  b.setInsertionPoint(ret);
  SmallVector<Value> results;
  for (auto [i, v] : llvm::enumerate(ret.getOperands())) {
    auto it = vals.find(v);
    auto to = mlir::cast<RankedTensorType>(func.getResultTypes()[i]);
    auto shape = it == vals.end()
                     ? ArrayRef<int64_t>{}
                     : mlir::cast<RankedTensorType>(it->second.v.getType())
                           .getShape();
    if (it == vals.end() || it->second.kind == Kind::Packed ||
        !it->second.segments.empty() || shape[2] != c || shape[0] != k ||
        formOf(to)->batch != 1) {
      return rollBack();
    }
    Value out = it->second.v;
    if (it->second.kind == Kind::Picked && it->second.index != 0) {
      out = place(ret.getLoc(), out, out, 0, it->second.index * T, T);
    }
    if (shape[1] == formOf(to)->rows) {
      out = record(reinterpret(b, ret.getLoc(), out, to))->getResult(0);
    } else {
      // Rows [0, T) of R: the chunks' tiles back in row order.
      const int64_t rank = to.getRank();
      AffineExpr col = getAffineDimExpr(rank - 1, ctx);
      SmallVector<AffineExpr> map{col.floorDiv(c / 32),
                                  getAffineConstantExpr(0, ctx),
                                  col % (c / 32)};
      out = record(b.create<ttir::TileViewOp>(
                       ret.getLoc(), to, out,
                       AffineMapAttr::get(AffineMap::get(rank, 0, map, ctx))))
                ->getResult(0);
    }
    results.push_back(out);
  }
  ret->setOperands(results);
  for (Operation *op : llvm::reverse(originals)) {
    if (!kept.contains(op)) {
      op->erase();
    }
  }
  // Picked values' ops on one packed value, repeated per pick: once.
  {
    IRRewriter rewriter(ctx);
    DominanceInfo dom(func);
    mlir::eliminateCommonSubExpressions(rewriter, dom, func);
  }
  // Then each result's value its own: a value the kernel also reads, or
  // another result returns (per-query norms with one weight are one value
  // after the CSE), goes out as a copy. One output CB is not also read in
  // compute: DstResident dropped such an output's pack, and an FPU
  // operand's CB read by the writer too hung (Inkling 2's local layers).
  for (OpOperand &operand : body.getTerminator()->getOpOperands()) {
    Operation *view = operand.get().getDefiningOp();
    if (!view || view->getNumOperands() != 1 ||
        !mlir::isa<ttir::TileViewOp, ttir::ReshapeOp>(view)) {
      continue;
    }
    Value x = view->getOperand(0);
    if (x.hasOneUse()) {
      continue;
    }
    b.setInsertionPoint(view);
    view->setOperand(0, place(view->getLoc(), x, x, 0, 0, T));
  }
  // The chunk count (one chunk per core: D2M's d2m.chunk_rows grid).
  if (exchanges) {
    func->setAttr("libtt.chunks", b.getI64IntegerAttr(k));
  }
  return true;
}

static bool rewriteHeadMajorSubgraph(func::FuncOp func) {

  if (func.getBody().getBlocks().size() != 1) {
    return false;
  }
  Block &body = func.getBody().front();
  // Unused reshapes (a norm weight's [1, D] -> [D] after its decomposition
  // looked through it) are not part of the subgraph.
  for (Operation &op : llvm::make_early_inc_range(llvm::reverse(body))) {
    if (mlir::isa<mlir::tt::ttir::ReshapeOp>(op) && op.use_empty()) {
      op.erase();
    }
  }
  auto ret = mlir::dyn_cast<func::ReturnOp>(body.getTerminator());
  if (!ret || ret.getNumOperands() != 1) {
    return false;
  }
  auto outReshape =
      ret.getOperand(0).getDefiningOp<mlir::tt::ttir::ReshapeOp>();
  if (!outReshape) {
    return false;
  }
  std::optional<HeadDims> dims = headMerge(
      mlir::cast<RankedTensorType>(outReshape.getInput().getType()).getShape(),
      mlir::cast<RankedTensorType>(outReshape.getType()).getShape());
  if (!dims) {
    return false;
  }
  const int64_t T = dims->T;
  // Classify the input reshapes; everything else must be elementwise or a
  // last-dim reduction on rank-3 values.
  SmallVector<std::pair<mlir::tt::ttir::ReshapeOp, HeadReshape>> inputs;
  for (Operation &op : body.without_terminator()) {
    if (&op == outReshape.getOperation()) {
      continue;
    }
    if (auto reshape = mlir::dyn_cast<mlir::tt::ttir::ReshapeOp>(op)) {
      // Inputs are arguments or rank <= 2 values computed here (e.g. the
      // widening typecast of [T, H*D]).
      std::optional<HeadReshape> kind = classifyHeadReshape(
          mlir::cast<RankedTensorType>(reshape.getInput().getType()).getShape(),
          mlir::cast<RankedTensorType>(reshape.getType()).getShape(), *dims);
      if (!kind) {
        return false;
      }
      inputs.push_back({reshape, *kind});
      continue;
    }
    if (op.getNumResults() != 1 || op.getNumRegions() != 0) {
      return false;
    }
    auto t = mlir::dyn_cast<RankedTensorType>(op.getResult(0).getType());
    // Rank <= 2 elementwise ops (before the split) keep their layout.
    if (t && t.getRank() < 3 &&
        mlir::isa<mlir::tt::ttir::ElementwiseUnary,
                  mlir::tt::ttir::ElementwiseBinary, mlir::tt::ttir::TypecastOp,
                  mlir::tt::ttir::SliceStaticOp>(op) &&
        llvm::all_of(op.getOperandTypes(), [](Type type) {
          return mlir::cast<RankedTensorType>(type).getRank() < 3;
        })) {
      continue;
    }
    if (!t || (t.getRank() != 3 && !mlir::isa<mlir::tt::ttir::FullOp>(op))) {
      return false;
    }
    if (auto sum = mlir::dyn_cast<mlir::tt::ttir::SumOp>(op)) {
      auto sumDims = sum.getDimArg();
      if (!sumDims || sumDims->size() != 1 ||
          (mlir::cast<IntegerAttr>((*sumDims)[0]).getInt() != 2 &&
           mlir::cast<IntegerAttr>((*sumDims)[0]).getInt() != -1)) {
        return false;
      }
      continue;
    }
    if (!mlir::isa<mlir::tt::ttir::ElementwiseUnary,
                   mlir::tt::ttir::ElementwiseBinary, mlir::tt::ttir::FullOp,
                   mlir::tt::ttir::TypecastOp, mlir::tt::ttir::ClampScalarOp>(
            op)) {
      return false;
    }
  }
  if (!llvm::any_of(inputs,
                    [](auto p) { return p.second == HeadReshape::Split; })) {
    return false;
  }
  // Rank-3 arguments keep TTNN's [T, H, D] layout: only single elements.
  for (BlockArgument arg : body.getArguments()) {
    auto t = mlir::cast<RankedTensorType>(arg.getType());
    if (t.getRank() == 3 && t.getNumElements() != 1 && !arg.use_empty()) {
      return false;
    }
  }
  // Group-major: heads of 16 stay in [T, H*D] (two per tile width); last-dim
  // sums become group sums.
  if (dims->D == 16) {
    return rewriteGroupMajor(func, outReshape, inputs, *dims);
  }
  auto swapped = [](RankedTensorType t) {
    if (t.getRank() != 3) {
      return t;
    }
    SmallVector<int64_t> shape{t.getDimSize(1), t.getDimSize(0),
                               t.getDimSize(2)};
    return withShape(t, shape);
  };
  mlir::OpBuilder b(func.getContext());
  for (Operation &op : llvm::make_early_inc_range(body.without_terminator())) {
    if (&op == outReshape.getOperation()) {
      continue;
    }
    auto reshape = mlir::dyn_cast<mlir::tt::ttir::ReshapeOp>(op);
    auto it = llvm::find_if(inputs, [&](auto p) { return p.first == reshape; });
    if (reshape && it != inputs.end()) {
      auto out = mlir::cast<RankedTensorType>(reshape.getType());
      if (it->second == HeadReshape::TokenRow) {
        // [1, T] or [T] -> the column [T, 1] (row_to_column), reinterpreted
        // as the column [1, T, 1].
        b.setInsertionPoint(reshape);
        mlir::Value col =
            rowToColumn(b, reshape.getLoc(), reshape.getInput(), T);
        auto tok =
            reinterpret(b, reshape.getLoc(), col, withShape(out, {1, T, 1}));
        reshape.getResult().replaceAllUsesWith(tok.getResult());
        reshape.erase();
        continue;
      }
      // A token column [T, 1] (column 0 of one tile, made by TTNN) is the
      // column [1, T, 1]; the other inputs swap dims 0 and 1.
      reinterpretInPlace(reshape, it->second == HeadReshape::Token
                                      ? withShape(out, {1, T, 1})
                                      : swapped(out));
      continue;
    }
    retypeResults(op, swapped);
  }
  outReshape->setAttr("libtt.tile_reinterpret", b.getUnitAttr());
  func->setAttr("libtt.head_major", b.getUnitAttr());
  return true;
}

namespace {

// The most rows any value of the subgraph has in a tile row (its
// second-to-last dim; one for rank < 2). At most 16 (decode), the SFPU
// chains of its kernels compute only those rows (D2MToTTKernel); the
// padding rows of the results are left unspecified.
static void annotateTileRows(func::FuncOp f) {
  int64_t rows = 1;
  auto visit = [&](Value v) {
    auto r = mlir::dyn_cast<RankedTensorType>(v.getType());
    if (r && r.getRank() >= 2) {
      rows = std::max(rows, r.getDimSize(r.getRank() - 2));
    }
  };
  // A matmul's weight (and its views) has rows of its own.
  std::function<bool(Value)> weight = [&](Value v) {
    return !v.use_empty() && llvm::all_of(v.getUses(), [&](OpOperand &use) {
      Operation *o = use.getOwner();
      if (mlir::isa<ttir::MatmulOp>(o)) {
        return use.getOperandNumber() == 1;
      }
      return mlir::isa<ttir::TileViewOp, ttir::ReshapeOp>(o) &&
             weight(o->getResult(0));
    });
  };
  for (Value arg : f.getArguments()) {
    if (!weight(arg)) {
      visit(arg);
    }
  }
  f.walk([&](Operation *op) {
    for (Value r : op->getResults()) {
      if (!weight(r)) {
        visit(r);
      }
    }
  });
  if (rows <= 16) {
    f->setAttr("libtt.tile_rows",
               IntegerAttr::get(IntegerType::get(f.getContext(), 64), rows));
  }
}

// A model repeats each subgraph per layer (an Inkling prefill had
// 1573 subgraphs, ~100 distinct); identical ones (after the per-dispatch
// specializations above) are compiled once, every dispatch calling the
// first copy (TTNNCollapseD2M clones the body per dispatch).
static void dedupeSubgraphs(ModuleOp moduleOp) {
  llvm::StringMap<func::FuncOp> canonical;
  llvm::DenseMap<Operation *, func::FuncOp> target;
  SmallVector<func::FuncOp> duplicates;
  moduleOp.walk([&](ttnn::D2MSubgraphOp dispatchOp) {
    func::FuncOp func = dispatchOp.getD2MMainFunc();
    if (!func || func.getBody().empty()) {
      return;
    }
    auto it = target.find(func.getOperation());
    if (it == target.end()) {
      std::string text;
      llvm::raw_string_ostream os(text);
      // Without locations (each layer's are its own).
      func.print(os, OpPrintingFlags()
                         .printGenericOpForm()
                         .enableDebugInfo(/*enable=*/false)
                         .useLocalScope());
      std::string name = func.getSymName().str();
      for (size_t pos = text.find(name); pos != std::string::npos;
           pos = text.find(name, pos)) {
        text.replace(pos, name.size(), "F");
      }
      auto [cit, inserted] = canonical.try_emplace(text, func);
      it = target.try_emplace(func.getOperation(), cit->second).first;
      if (!inserted) {
        duplicates.push_back(func);
      }
    }
    if (it->second != func) {
      dispatchOp.setD2mFuncAttr(
          FlatSymbolRefAttr::get(it->second.getSymNameAttr()));
    }
  });
  for (func::FuncOp func : duplicates) {
    func.erase();
  }
}

static std::optional<std::pair<RankedTensorType, Attribute>>
scalarConstant(ModuleOp moduleOp, Value v) {
  auto load = v.getDefiningOp<ttcore::LoadCachedOp>();
  if (!load || !load.getInputs().empty() || load->getNumResults() != 1) {
    return std::nullopt;
  }
  auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
      load, load.getCalleeAttr());
  if (!callee || callee.getBody().getBlocks().size() != 1) {
    return std::nullopt;
  }
  auto ret = dyn_cast<func::ReturnOp>(callee.getBody().front().getTerminator());
  if (!ret || ret.getNumOperands() != 1) {
    return std::nullopt;
  }
  auto full = ret.getOperand(0).getDefiningOp<ttnn::FullOp>();
  if (!full) {
    return std::nullopt;
  }
  auto type = cast<RankedTensorType>(full.getResult().getType());
  if (type.getNumElements() != 1) {
    return std::nullopt;
  }
  return std::make_pair(type, full.getFillValueAttr());
}

static void inlineScalarConstants(ModuleOp moduleOp) {
  moduleOp.walk([&](ttnn::D2MSubgraphOp dispatchOp) {
    func::FuncOp func = dispatchOp.getD2MMainFunc();
    if (!func || func.getBody().empty()) {
      return;
    }
    OpBuilder b(func.getContext());
    b.setInsertionPointToStart(&func.getBody().front());
    for (auto [i, input] : llvm::enumerate(dispatchOp.getInputs())) {
      auto constant = scalarConstant(moduleOp, input);
      if (!constant || i >= func.getNumArguments()) {
        continue;
      }
      auto [type, fill] = *constant;
      BlockArgument arg = func.getArgument(i);
      auto argType = cast<RankedTensorType>(arg.getType());
      SmallVector<int32_t> shape(argType.getShape().begin(),
                                 argType.getShape().end());
      // The argument's TTNN layout keeps D2M's layout of the broadcast
      // operand as it was; ScalarizeConstTensors makes it a scalar.
      auto full = b.create<ttir::FullOp>(func.getLoc(), argType,
                                         b.getDenseI32ArrayAttr(shape), fill);
      arg.replaceAllUsesWith(full.getResult());
    }
  });
}

} // namespace

LogicalResult finishD2MSubgraphs(ModuleOp module,
                                 ArrayRef<func::FuncOp> d2mSubgraphs) {
  // A subgraph input that is a scalar const-eval constant (a
  // load_cached of a function returning one ttnn.full) becomes a ttir.full
  // inside the subgraph, which D2M applies as a scalar operand instead of
  // loading and broadcasting a tile (eps, thresholds).
  inlineScalarConstants(module);
  // Head-major / group-major per-head subgraphs.
  for (func::FuncOp f : d2mSubgraphs) {
    if (rewriteHeadMajorSubgraph(f) || splitWideRows(f)) {
      continue;
    }
    // D2M's general reshape views are wrong across tiles: a head reshape the
    // rewrite did not take must not be compiled (the outliner admits head
    // reshapes only in groups the rewrite takes; a mismatch is a bug).
    bool viewLeft = false;
    f.walk([&](ttir::ReshapeOp r) {
      auto in = mlir::cast<RankedTensorType>(r.getInput().getType());
      auto out = mlir::cast<RankedTensorType>(r.getType());
      viewLeft |= !r->hasAttr("libtt.tile_reinterpret") && !r->use_empty() &&
                  in.getNumElements() > 1 && in.getRank() != out.getRank();
    });
    if (viewLeft) {
      f.emitError("libtt: D2M subgraph keeps a head reshape the "
                  "head-major rewrite did not take");
      return failure();
    }
  }
  for (func::FuncOp f : d2mSubgraphs) {
    annotateTileRows(f);
  }
  dedupeSubgraphs(module);
  return success();
}

} // namespace mlir::tt::d2m_fusion
