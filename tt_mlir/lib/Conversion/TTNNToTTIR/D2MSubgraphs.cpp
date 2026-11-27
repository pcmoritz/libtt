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
    // One tile row of more than 32 tiles: reduced in k chunks, one per core.
    // x as [k, T, N / k] (the same tiles in the same order), each chunk's sum
    // of squares [k, T, 1], summed over the chunks by an exchange between the
    // cores inside the kernel (ttir.chunk_allreduce), and the normalized
    // chunks viewed back: the whole norm is one row-merged kernel.
    if (auto chunks = wideNormChunks(op, mlir::cast<mlir::RankedTensorType>(
                                             op->getOperand(0).getType()))) {
      const int64_t k = *chunks;
      const int64_t T = inT.getDimSize(rank - 2);
      auto split = retype(inT, {k, T, width / k}, f32);
      auto sums = retype(inT, {k, T, 1}, f32);
      // Views of the arguments (in DRAM), cast after: a view of an
      // intermediate (sharded in L1 by its producer) addresses it wrongly.
      auto view = [&](mlir::Value v, int64_t rows) {
        auto t = mlir::cast<mlir::RankedTensorType>(v.getType());
        return asF32(
            reinterpret(rw, loc, v, withShape(t, {k, rows, width / k})));
      };
      mlir::Value x = view(adaptor.getInput(), T);
      mlir::Value squares = rw.create<ttir::MultiplyOp>(loc, split, x, x);
      mlir::Value part = rw.create<ttir::SumOp>(
          loc, sums, squares, /*keep_dim=*/true, rw.getI32ArrayAttr({2}));
      mlir::Value total = rw.create<ttir::ChunkAllReduceOp>(loc, sums, part);
      mlir::Value mean = rw.create<ttir::MultiplyOp>(
          loc, sums, total, scalar(1.0f / static_cast<float>(width), 3));
      mlir::Value var = rw.create<ttir::AddOp>(loc, sums, mean, scalar(eps, 3));
      // The factors of each element, multiplied in order; the last multiply
      // is FP32 with the result's encoding, so D2M folds the narrowing into
      // the pack (a separate typecast would narrow twice).
      llvm::SmallVector<mlir::Value> factors{
          x, rw.create<ttir::RsqrtOp>(loc, sums, var)};
      if (adaptor.getWeight()) {
        factors.push_back(view(weightArg(), 1));
      }
      if constexpr (std::is_same_v<SrcOp, mlir::tt::ttnn::RMSNormGatedOp>) {
        factors.push_back(
            rw.create<ttir::SiluOp>(loc, split, view(adaptor.getGate(), T)));
      }
      auto outSplit = withShape(outType, {k, T, width / k});
      auto lastType = mlir::RankedTensorType::get(outSplit.getShape(), f32,
                                                  outSplit.getEncoding());
      mlir::Value y = factors.front();
      for (size_t i = 1; i < factors.size(); ++i) {
        bool last = i + 1 == factors.size();
        y = rw.create<ttir::MultiplyOp>(loc, last ? lastType : split, y,
                                        factors[i]);
      }
      if (!outType.getElementType().isF32()) {
        y = rw.create<ttir::TypecastOp>(loc, outSplit, y);
      }
      rw.replaceOp(op, reinterpret(rw, loc, y, outType).getResult());
      return mlir::success();
    }
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
    auto slice = rewriter.replaceOpWithNewOp<mlir::tt::ttir::SliceStaticOp>(
        op, type, adaptor.getInput(), op.getBeginsAttr(), op.getEndsAttr(),
        op.getStepAttr());
    slice->setAttr("libtt.tile_slice", rewriter.getUnitAttr());
    return mlir::success();
  }
};
} // namespace

void populateD2MSubgraphPatterns(MLIRContext *ctx, RewritePatternSet &patterns,
                                 TypeConverter &typeConverter) {
  patterns.add<TTNNToTTIRReshapePattern, TTNNToTTIRSlicePattern,
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

// An elementwise subgraph over one tile row [T, N] (T <= 32) wider
// than a core row (more than 8 tiles; Inkling 2's attention output gate,
// [1, 768]) gets one core row. Computed as [k, T, N / k] (the same tiles in
// the same order, k tile rows: tile reinterpretations), it spreads over k.
static bool rewriteWideRowSubgraph(func::FuncOp func) {
  namespace ttir = mlir::tt::ttir;
  if (func.getBody().getBlocks().size() != 1) {
    return false;
  }
  Block &body = func.getBody().front();
  int64_t T = -1, N = -1;
  auto row = [&](Type type) {
    auto t = mlir::dyn_cast<RankedTensorType>(type);
    if (!t || t.getNumElements() == 1) {
      return t != nullptr;
    }
    if (t.getRank() != 2) {
      return false;
    }
    if (T < 0) {
      T = t.getDimSize(0);
      N = t.getDimSize(1);
    }
    return t.getDimSize(0) == T && t.getDimSize(1) == N;
  };
  for (Operation &op : body) {
    auto slice = mlir::dyn_cast<ttir::SliceStaticOp>(op);
    if (slice && !slice->hasAttr("libtt.tile_slice")) {
      return false;
    }
    if (!slice &&
        !mlir::isa<ttir::ElementwiseUnary, ttir::ElementwiseBinary,
                   ttir::TypecastOp, ttir::FullOp, func::ReturnOp>(op)) {
      return false;
    }
    if (!slice && !llvm::all_of(op.getOperandTypes(), row)) {
      return false;
    }
    if (!llvm::all_of(op.getResultTypes(), row)) {
      return false;
    }
  }
  if (T < 1 || T > 32 || N % 32 != 0) {
    return false;
  }
  const int64_t tiles = N / 32;
  int64_t k = 0;
  for (int64_t d = 2; d <= 8 && !k; ++d) {
    if (tiles % d == 0 && tiles / d <= 8) {
      k = d;
    }
  }
  if (tiles <= 8 || !k) {
    return false;
  }
  SmallVector<int64_t> wide{k, T, N / k};
  auto widen = [&](RankedTensorType t) {
    return t.getNumElements() == 1 ? withShape(t, {1, 1, 1})
                                   : withShape(t, wide);
  };
  mlir::OpBuilder b(func.getContext());
  // Arguments (other than slice sources) and slice results: views.
  b.setInsertionPointToStart(&body);
  for (BlockArgument arg : body.getArguments()) {
    if (llvm::all_of(arg.getUsers(), [](Operation *u) {
          return mlir::isa<ttir::SliceStaticOp>(u);
        })) {
      continue;
    }
    auto t = mlir::cast<RankedTensorType>(arg.getType());
    auto view = reinterpret(b, arg.getLoc(), arg, widen(t));
    arg.replaceUsesWithIf(view.getResult(), [&](OpOperand &use) {
      return use.getOwner() != view &&
             !mlir::isa<ttir::SliceStaticOp>(use.getOwner());
    });
  }
  for (Operation &op : llvm::make_early_inc_range(body.without_terminator())) {
    if (auto slice = mlir::dyn_cast<ttir::SliceStaticOp>(op)) {
      b.setInsertionPointAfter(slice);
      auto view = reinterpret(b, slice.getLoc(), slice.getResult(),
                              widen(slice.getType()));
      slice.getResult().replaceAllUsesExcept(view.getResult(), view);
      continue;
    }
    if (!mlir::isa<ttir::ReshapeOp>(op)) {
      retypeResults(op, widen);
    }
  }
  auto ret = mlir::cast<func::ReturnOp>(body.getTerminator());
  b.setInsertionPoint(ret);
  for (auto [i, v] : llvm::enumerate(ret.getOperands())) {
    auto to = mlir::cast<RankedTensorType>(func.getResultTypes()[i]);
    if (v.getType() != to) {
      ret.setOperand(i, reinterpret(b, ret.getLoc(), v, to).getResult());
    }
  }
  func->setAttr("libtt.wide_rows", b.getUnitAttr());
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
  auto visit = [&](Type t) {
    auto r = mlir::dyn_cast<RankedTensorType>(t);
    if (r && r.getRank() >= 2) {
      rows = std::max(rows, r.getDimSize(r.getRank() - 2));
    }
  };
  for (Type t : f.getArgumentTypes()) {
    visit(t);
  }
  f.walk([&](Operation *op) {
    for (Type t : op->getResultTypes()) {
      visit(t);
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
    if (rewriteHeadMajorSubgraph(f) || rewriteWideRowSubgraph(f)) {
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
