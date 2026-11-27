// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

// libtt's kernel-level D2M rewrites (D2MToTTKernel runs them on each compute
// kernel with FP32 DST, see ttmlir/D2MFusion.h): a whole kernel kept in DST
// (keepKernelInDst), and runs of elementwise SFPU tile ops fused into one SFPI
// function each (fuseSFPUChains).

#include "ttmlir/D2MFusion.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "ttmlir/Conversion/D2MToTTKernel/D2MToTTKernel.h"
#include "ttmlir/Conversion/Passes.h.inc"
#include "ttmlir/Dialect/D2M/Analysis/CBProducerConsumer.h"
#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/Utils/CBUtils.h"
#include "ttmlir/Dialect/D2M/Utils/DMAUtils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetal.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/Support/FormatVariadic.h"
#include <map>
#include <regex>
#include <sstream>
#include <string>

using namespace mlir;
using namespace mlir::tt;

namespace mlir::tt::d2m_fusion {

namespace {
// Fuse runs of elementwise SFPU tile ops into one SFPI function.
//
// tt-metal's SFPU LLKs each make a full pass over DST (load, op, store for
// every row of the tile); a chain of N ops is N passes. A run of supported
// ops between other kernel calls (copies, broadcasts, packs, control flow)
// becomes one function that loads each DST tile once per row, applies the
// whole chain in SFPU registers and stores each written tile once. Only for
// FP32 DST (32-bit CBs), where the per-op LLKs compute and store in FP32 too.
class SFPUChainEmitter {
public:
  // Returns false if `op` cannot be part of a chain.
  bool add(Operation *op) {
    // DST slots this op reads that hold pair-uniform values (equal in the
    // even and odd vector of a band, DstResidentRewriter's libtt.pu_slots).
    curUniformSlots.clear();
    if (auto pu = op->getAttrOfType<DenseI64ArrayAttr>("libtt.pu_slots")) {
      curUniformSlots.insert(pu.asArrayRef().begin(), pu.asArrayRef().end());
    }
    auto un = [&](Value idx, const char *fmt) {
      int k = slot(idx);
      std::string x = read(k);
      write(k, llvm::formatv(fmt, x).str(), isUniform(x));
    };
    auto bin = [&](Value i0, Value i1, Value out, const char *fmt) {
      std::string x = read(slot(i0));
      std::string y = read(slot(i1));
      write(slot(out), llvm::formatv(fmt, x, y).str(),
            isUniform(x) && isUniform(y));
    };
    auto scalar = [&](Value idx, Value p, const char *fmt) {
      int k = slot(idx);
      std::string x = read(k);
      std::string c = "libtt_sfpu::bits(" + param(p) + ")";
      write(k, llvm::formatv(fmt, x, c).str(), isUniform(x));
    };
    auto select = [&](int out, const std::string &code, const std::string &t,
                      bool u) {
      lines.push_back(code);
      lineUniform.push_back(u);
      if (u) {
        uniformTemps.insert(t);
      }
      assign(out, t);
    };
    if (auto m = dyn_cast<ttkernel::MulBinaryTilesOp>(op)) {
      bin(m.getDst0Index(), m.getDst1Index(), m.getOdstIndex(), "{0} * {1}");
    } else if (auto m = dyn_cast<ttkernel::AddBinaryTilesOp>(op)) {
      bin(m.getDst0Index(), m.getDst1Index(), m.getOdstIndex(), "{0} + {1}");
    } else if (auto m = dyn_cast<ttkernel::SubBinaryTilesOp>(op)) {
      bin(m.getDst0Index(), m.getDst1Index(), m.getOdstIndex(), "{0} - {1}");
    } else if (auto m = dyn_cast<ttkernel::BinaryMaxTileOp>(op)) {
      std::string x = read(slot(m.getDst0Index()));
      std::string y = read(slot(m.getDst1Index()));
      std::string t = fresh();
      select(slot(m.getOdstIndex()),
             llvm::formatv("sfpi::vFloat {0} = {1}; v_if({1} < {2}) {{ {0} = "
                           "{2}; } v_endif;",
                           t, x, y)
                 .str(),
             t, isUniform(x) && isUniform(y));
    } else if (auto m = dyn_cast<ttkernel::BinaryMinTileOp>(op)) {
      std::string x = read(slot(m.getDst0Index()));
      std::string y = read(slot(m.getDst1Index()));
      std::string t = fresh();
      select(slot(m.getOdstIndex()),
             llvm::formatv("sfpi::vFloat {0} = {1}; v_if({2} < {1}) {{ {0} = "
                           "{2}; } v_endif;",
                           t, x, y)
                 .str(),
             t, isUniform(x) && isUniform(y));
    } else if (auto m = dyn_cast<ttkernel::MulUnaryTileOp>(op)) {
      scalar(m.getDst0Index(), m.getParam(), "{0} * {1}");
    } else if (auto m = dyn_cast<ttkernel::DivUnaryTileOp>(op)) {
      // The host passes the reciprocal (tt-metal's DIV mode multiplies).
      scalar(m.getDst0Index(), m.getParam(), "{0} * {1}");
    } else if (auto m = dyn_cast<ttkernel::AddUnaryTileOp>(op)) {
      scalar(m.getDst0Index(), m.getParam(), "{0} + {1}");
    } else if (auto m = dyn_cast<ttkernel::SubUnaryTileOp>(op)) {
      scalar(m.getDst0Index(), m.getParam(), "{0} - {1}");
    } else if (auto m = dyn_cast<ttkernel::AbsTileOp>(op)) {
      un(m.getDst0Index(), "sfpi::abs({0})");
    } else if (isa<ttkernel::NegativeTileOp>(op)) {
      un(op->getOperand(0), "-{0}");
    } else if (auto m = dyn_cast<ttkernel::RsqrtTileOp>(op)) {
      un(m.getTileIndex(), "libtt_sfpu::rsqrt({0})");
    } else if (isa<ttkernel::SiluTileOp>(op)) {
      un(op->getOperand(0), "libtt_sfpu::silu({0})");
    } else if (isa<ttkernel::SigmoidTileOp>(op)) {
      un(op->getOperand(0), "libtt_sfpu::sigmoid({0})");
    } else if (isa<ttkernel::ExpTileOp>(op) &&
               !(op->getAttrOfType<BoolAttr>("approx") &&
                 op->getAttrOfType<BoolAttr>("approx").getValue()) &&
               !op->hasAttr("scale")) {
      un(op->getOperand(0), "libtt_sfpu::exp({0})");
    } else if (isa<ttkernel::SqrtTileOp>(op)) {
      un(op->getOperand(0), "libtt_sfpu::sqrt({0})");
    } else if (isa<ttkernel::RecipTileOp>(op)) {
      un(op->getOperand(0), "libtt_sfpu::recip({0})");
    } else if (auto m = dyn_cast<ttkernel::SignTileOp>(op)) {
      int k = slot(m.getTileIndex());
      std::string x = read(k);
      std::string t = fresh();
      select(k,
             llvm::formatv("sfpi::vFloat {0} = 0.0f; v_if({1} > 0.0f) {{ {0} "
                           "= 1.0f; } v_elseif({1} < 0.0f) {{ {0} = -1.0f; } "
                           "v_endif;",
                           t, x)
                 .str(),
             t, isUniform(x));
    } else if (auto m = dyn_cast<ttkernel::ClampScalarTileOp>(op)) {
      int k = slot(m.getDstIndex());
      std::string x = read(k);
      // Bounds at -FLT_MAX / FLT_MAX (min(1, x) as a clamp) need no test.
      auto bound = [](Value v) -> std::optional<int64_t> {
        return getConstantIntValue(v);
      };
      auto lb = bound(m.getMinParam()), ub = bound(m.getMaxParam());
      bool noLo =
          lb && static_cast<int32_t>(*lb) == static_cast<int32_t>(0xff7fffff);
      bool noHi =
          ub && static_cast<int32_t>(*ub) == static_cast<int32_t>(0x7f7fffff);
      std::string t = fresh();
      std::string code = llvm::formatv("sfpi::vFloat {0} = {1};", t, x).str();
      if (!noLo) {
        std::string lo = "libtt_sfpu::bits(" + param(m.getMinParam()) + ")";
        code +=
            llvm::formatv(" v_if({0} < {1}) {{ {0} = {1}; } v_endif;", t, lo)
                .str();
      }
      if (!noHi) {
        std::string hi = "libtt_sfpu::bits(" + param(m.getMaxParam()) + ")";
        code +=
            llvm::formatv(" v_if({0} > {1}) {{ {0} = {1}; } v_endif;", t, hi)
                .str();
      }
      select(k, code, t, isUniform(x));
    } else if (auto m = dyn_cast<ttkernel::LibttGroupSumTileOp>(op)) {
      // Pair op: the 16-column row-half sum needs the band's even and odd
      // vectors (dual emission, @PAIR below); the result is pair-uniform.
      int k = slot(m.getTileIndex());
      std::string x = read(k);
      std::string t = fresh();
      select(k,
             llvm::formatv("sfpi::vFloat {0} = "
                           "libtt_sfpu::sum_lanes8f(@PAIR({1}));",
                           t, x)
                 .str(),
             t, true);
      hasPairOp = true;
    } else if (auto m = dyn_cast<ttkernel::FillTileOp>(op)) {
      int k = slot(m.getDstIndex());
      std::string t = fresh();
      select(k,
             llvm::formatv("sfpi::vFloat {0} = sfpi::vFloat({1});", t,
                           param(m.getValue()))
                 .str(),
             t, true);
    } else if (auto guarded = conditionalFill(op)) {
      // if (cond) fill(k, v): the slice keeps its value unless cond (a
      // reduction's first-iteration seed).
      auto [cond, fill] = *guarded;
      int k = slot(fill.getDstIndex());
      std::string x = read(k);
      std::string t = fresh();
      select(k,
             llvm::formatv("sfpi::vFloat {0} = {1}; if ({2}) {{ {0} = "
                           "sfpi::vFloat({3}); }",
                           t, x, param(cond), param(fill.getValue()))
                 .str(),
             t, isUniform(x));
    } else {
      return false;
    }
    ++numOps;
    return true;
  }

  // scf.if (cond) { fill_tile_init; fill_tile(k, v) } with no else.
  static std::optional<std::pair<Value, ttkernel::FillTileOp>>
  conditionalFill(Operation *op) {
    auto ifOp = dyn_cast<scf::IfOp>(op);
    if (!ifOp || ifOp->getNumResults() != 0 || !ifOp.getElseRegion().empty()) {
      return std::nullopt;
    }
    ttkernel::FillTileOp fill;
    for (Operation &inner : ifOp.getThenRegion().front()) {
      if (auto f = dyn_cast<ttkernel::FillTileOp>(inner)) {
        if (fill) {
          return std::nullopt;
        }
        fill = f;
      } else if (!isa<ttkernel::FillTileInitOp, scf::YieldOp>(inner) &&
                 !(isPure(&inner) && inner.getNumRegions() == 0)) {
        return std::nullopt;
      }
    }
    if (!fill) {
      return std::nullopt;
    }
    // The index and value must be visible outside the if; constants made
    // inside move before it.
    for (Value v : fill->getOperands()) {
      Operation *def = v.getDefiningOp();
      if (def && ifOp->isAncestor(def) &&
          !def->hasTrait<OpTrait::ConstantLike>()) {
        return std::nullopt;
      }
    }
    for (Value v : fill->getOperands()) {
      Operation *def = v.getDefiningOp();
      if (def && ifOp->isAncestor(def)) {
        def->moveBefore(ifOp);
      }
    }
    return std::make_pair(ifOp.getCondition(), fill);
  }

  static bool isInit(Operation *op) {
    return isa<ttkernel::MulBinaryTilesInitOp, ttkernel::AddBinaryTilesInitOp,
               ttkernel::SubBinaryTilesInitOp, ttkernel::BinaryMaxTileInitOp,
               ttkernel::BinaryMinTileInitOp,
               ttkernel::BinopWithScalarTileInitOp, ttkernel::AbsTileInitOp,
               ttkernel::NegativeTileInitOp, ttkernel::SignTileInitOp,
               ttkernel::RsqrtTileInitOp, ttkernel::SqrtTileInitOp,
               ttkernel::RecipTileInitOp, ttkernel::ClampScalarTileInitOp,
               ttkernel::FillTileInitOp, ttkernel::LibttGroupSumTileInitOp,
               ttkernel::SiluTileInitOp, ttkernel::SigmoidTileInitOp,
               ttkernel::ExpTileInitOp>(op);
  }

  int getNumOps() const { return numOps; }
  // Only column 0 of the results is read (DstResidentRewriter's
  // libtt.col0_only): compute the left faces' even vectors only.
  bool col0Only = false;
  // libtt.tile_rows: every value of the kernel has at most this many rows
  // in each tile row (1..16, decode); only those rows are computed.
  int64_t tileRows = 0;

  // Emits the call before `before`; false if the chain is too large.
  bool emit(Operation *before) {
    if (slotValues.size() > 8 || params.size() > 8 || numOps > 32) {
      return false;
    }
    std::string sig;
    SmallVector<Value> args;
    for (auto [k, v] : llvm::enumerate(slotValues)) {
      sig += (k ? ", " : "") + llvm::formatv("uint32_t d{0}", k).str();
      args.push_back(v);
    }
    for (auto [k, v] : llvm::enumerate(params)) {
      sig += (sig.empty() ? "" : ", ") +
             llvm::formatv("{0} p{1}",
                           v.getType().isF32() ? "float" : "uint32_t", k)
                 .str();
      args.push_back(v);
    }
    std::string body;
    for (const std::string &l : lines) {
      body += "      " + l + "\n";
    }
    std::string stores;
    for (auto [k, var] : current) {
      if (written.contains(k)) {
        stores +=
            llvm::formatv("      sfpi::dst_reg[d{0} * 32] = {1};\n", k, var)
                .str();
      }
    }
    body += stores;
    // Dual emission: the even and odd vector of a band in one iteration.
    // Pair-uniform lines (equal in both vectors) run once; the others twice
    // (temps t<n>_e / t<n>_o, the odd vector one register further). Pair ops
    // (group sums) combine the two.
    // With few rows, unrolled vectors replace the loop: pair only for pair
    // ops (which need both vectors).
    bool paired =
        !col0Only &&
        (hasPairOp || (tileRows == 0 && llvm::is_contained(lineUniform, true) &&
                       llvm::is_contained(lineUniform, false)));
    if (col0Only && hasPairOp) {
      return false;
    }
    // Without a pair op: the even vector's lines, then the odd vector's
    // non-uniform lines in a nested scope reusing the uniform temps.
    if (paired && !hasPairOp) {
      std::string odd;
      for (auto [l, u] : llvm::zip(lines, lineUniform)) {
        if (!u) {
          odd += "        " + l + "\n";
        }
      }
      odd += stores;
      body += "      sfpi::dst_reg++;\n      {\n" + odd + "      }\n";
    } else if (paired) {
      std::regex temp("\\bt([0-9]+)\\b");
      auto rename = [&](const std::string &l, const std::string &suffix) {
        std::string out;
        auto begin = std::sregex_iterator(l.begin(), l.end(), temp);
        size_t last = 0;
        for (auto it = begin; it != std::sregex_iterator(); ++it) {
          out += l.substr(last, it->position() - last);
          std::string name = it->str();
          out += uniformTemps.count(name) ? name : name + suffix;
          last = it->position() + it->length();
        }
        return out + l.substr(last);
      };
      auto expandPairs = [&](std::string l) {
        for (size_t pos = l.find("@PAIR("); pos != std::string::npos;
             pos = l.find("@PAIR(")) {
          size_t close = l.find(')', pos);
          std::string x = l.substr(pos + 6, close - pos - 6);
          std::string sum = uniformTemps.count(x)
                                ? "(" + x + " + " + x + ")"
                                : "(" + x + "_e + " + x + "_o)";
          l.replace(pos, close - pos + 1, sum);
        }
        return l;
      };
      auto oddLoad = [](std::string l) {
        size_t pos = l.find(" * 32]");
        if (pos != std::string::npos) {
          l.replace(pos, 6, " * 32 + 1]");
        }
        return l;
      };
      body.clear();
      for (auto [l, u] : llvm::zip(lines, lineUniform)) {
        if (u) {
          body += "      " + expandPairs(l) + "\n";
        } else {
          body += "      " + rename(l, "_e") + "\n";
          body += "      " + oddLoad(rename(l, "_o")) + "\n";
        }
      }
      for (auto [k, var] : current) {
        if (!written.contains(k)) {
          continue;
        }
        bool u = uniformTemps.count(var);
        body += llvm::formatv("      sfpi::dst_reg[d{0} * 32] = {1};\n", k,
                              u ? var : var + "_e")
                    .str();
        body += llvm::formatv("      sfpi::dst_reg[d{0} * 32 + 1] = {1};\n", k,
                              u ? var : var + "_o")
                    .str();
      }
      body += "      sfpi::dst_reg++;\n";
      // SFPU registers: at most a few vectors live at once (the compiler
      // cannot spill: "too few lregs").
      std::regex def("sfpi::vFloat (t[0-9_eo]+)");
      std::regex use("\\bt[0-9]+(_e|_o)?\\b");
      std::vector<std::pair<std::string, std::set<std::string>>> parsed;
      std::istringstream in(body);
      for (std::string l; std::getline(in, l);) {
        std::smatch m;
        std::string d = std::regex_search(l, m, def) ? m[1].str() : "";
        std::set<std::string> uses;
        for (auto it = std::sregex_iterator(l.begin(), l.end(), use);
             it != std::sregex_iterator(); ++it) {
          if (it->str() != d) {
            uses.insert(it->str());
          }
        }
        parsed.push_back({d, uses});
      }
      std::map<std::string, size_t> lastUse;
      for (size_t i = 0; i < parsed.size(); ++i) {
        for (const std::string &u : parsed[i].second) {
          lastUse[u] = i;
        }
      }
      std::set<std::string> live;
      size_t maxLive = 0;
      for (size_t i = 0; i < parsed.size(); ++i) {
        if (!parsed[i].first.empty()) {
          live.insert(parsed[i].first);
        }
        maxLive = std::max(maxLive, live.size());
        for (auto it = live.begin(); it != live.end();) {
          auto lu = lastUse.find(*it);
          it = (lu == lastUse.end() || lu->second <= i) ? live.erase(it)
                                                        : std::next(it);
        }
      }
      if (maxLive > 5) {
        return false;
      }
    }
    const bool unroll = col0Only || tileRows > 0;
    if (unroll) {
      // FP32 DST lane layout: vector d (0..7) of face f is dst_reg offset
      // 8 * f + d of a tile and holds rows 4 * (d / 2) .. + 3, the even
      // (d even) or odd columns of the face (faces 0, 1: rows 0..15, columns
      // 0..15 and 16..31; faces 2, 3: rows 16..31). Column 0: the even
      // vectors of faces 0 and 2; R <= 16 rows: vectors d < 2 * ceil(R / 4)
      // of faces 0 and 1; a pair (even, odd) per even d with pair ops.
      const int vectors = tileRows > 0 ? 2 * ((tileRows + 3) / 4) : 8;
      SmallVector<int> faces =
          tileRows > 0
              ? (col0Only ? SmallVector<int>{0} : SmallVector<int>{0, 1})
              : SmallVector<int>{0, 2};
      const bool evenOnly = col0Only || paired;
      std::string unrolled;
      for (int f : faces) {
        for (int d = 0; d < vectors; d += evenOnly ? 2 : 1) {
          int off = 8 * f + d;
          std::string copy = body;
          auto replaceAll = [&](const std::string &from,
                                const std::string &to) {
            for (size_t pos = copy.find(from); pos != std::string::npos;
                 pos = copy.find(from, pos + to.size())) {
              copy.replace(pos, from.size(), to);
            }
          };
          replaceAll(" * 32 + 1]", llvm::formatv(" * 32 + {0}]", off + 1));
          replaceAll(" * 32]", llvm::formatv(" * 32 + {0}]", off));
          replaceAll("sfpi::dst_reg++;", "");
          unrolled += "  {\n" + copy + "  }\n";
        }
      }
      body = unrolled;
    }
    // The exp-based helpers use most SFPU registers: few chain values may be
    // live across one (the compiler cannot spill: "too few lregs").
    {
      std::regex def("sfpi::vFloat (t[0-9_eo]+)");
      std::regex use("\\bt[0-9]+(_e|_o)?\\b");
      std::vector<std::string> blines;
      std::istringstream in(body);
      for (std::string l; std::getline(in, l);) {
        blines.push_back(l);
      }
      std::map<std::string, size_t> first, last;
      for (size_t i = 0; i < blines.size(); ++i) {
        for (auto it =
                 std::sregex_iterator(blines[i].begin(), blines[i].end(), use);
             it != std::sregex_iterator(); ++it) {
          first.try_emplace(it->str(), i);
          last[it->str()] = i;
        }
      }
      for (size_t i = 0; i < blines.size(); ++i) {
        const std::string &l = blines[i];
        if (l.find("libtt_sfpu::exp(") == std::string::npos &&
            l.find("libtt_sfpu::sigmoid(") == std::string::npos &&
            l.find("libtt_sfpu::silu(") == std::string::npos) {
          continue;
        }
        int across = 0;
        for (auto [t, f] : first) {
          across += f < i && last[t] > i;
        }
        if (across > 2) {
          return false;
        }
      }
    }
    std::string fn =
        unroll ? llvm::formatv("ALWI void NAME({0}) {{\n"
                               "  llk_math_eltwise_unary_sfpu_init<SfpuType::"
                               "unused, DST_ACCUM_MODE>();\n"
                               "  _llk_math_eltwise_sfpu_start_(0);\n"
                               "{1}"
                               "  _llk_math_eltwise_sfpu_done_();\n"
                               "}\n",
                               sig, body)
                     .str()
               : llvm::formatv(
                     "ALWI void NAME({0}) {{\n"
                     "  llk_math_eltwise_unary_sfpu_init<SfpuType::unused, "
                     "DST_ACCUM_MODE>();\n"
                     "  _llk_math_eltwise_sfpu_start_(0);\n"
                     "#pragma GCC unroll 0\n"
                     "  for (int face = 0; face < 4; face++) {{\n"
                     "#pragma GCC unroll 0\n"
                     "    for (int row = 0; row < 8; row += {2}) {{\n"
                     "{1}"
                     "      sfpi::dst_reg++;\n"
                     "    }\n"
                     "    _llk_math_eltwise_sfpu_inc_dst_face_addr_();\n"
                     "  }\n"
                     "  _llk_math_eltwise_sfpu_done_();\n"
                     "}\n",
                     sig, body, paired ? 2 : 1)
                     .str();
    std::string name =
        llvm::formatv("chain_{0:x}",
                      static_cast<uint64_t>(llvm::hash_value(fn)))
            .str();
    size_t pos = fn.find("NAME");
    fn.replace(pos, 4, name);
    std::string def = "#ifdef TRISC_MATH\nnamespace libtt_sfpu {\n" + fn +
                      "} // namespace libtt_sfpu\n#endif";
    std::string call = "MATH((libtt_sfpu::" + name + "(";
    for (size_t i = 0; i < args.size(); ++i) {
      call += i ? ", {}" : "{}";
    }
    call += ")));";
    OpBuilder builder(before);
    OperationState state(before->getLoc(),
                         emitc::VerbatimOp::getOperationName());
    state.addOperands(args);
    state.addAttribute("value", builder.getStringAttr(call));
    state.addAttribute("libtt.sfpu_def", builder.getStringAttr(def));
    builder.create(state);
    return true;
  }

private:
  int slot(Value idx) {
    if (std::optional<int64_t> c = getConstantIntValue(idx)) {
      auto [it, inserted] = constSlots.try_emplace(*c, slotValues.size());
      if (inserted) {
        slotValues.push_back(idx);
      }
      return it->second;
    }
    auto [it, inserted] = valueSlots.try_emplace(idx, slotValues.size());
    if (inserted) {
      slotValues.push_back(idx);
    }
    return it->second;
  }
  std::string param(Value v) {
    auto [it, inserted] = paramIndex.try_emplace(v, params.size());
    if (inserted) {
      params.push_back(v);
    }
    return "p" + std::to_string(it->second);
  }
  std::string fresh() { return "t" + std::to_string(numTemps++); }
  std::string read(int k) {
    auto it = current.find(k);
    if (it != current.end()) {
      return it->second;
    }
    std::string t = fresh();
    lines.push_back(
        llvm::formatv("sfpi::vFloat {0} = sfpi::dst_reg[d{1} * 32];", t, k)
            .str());
    std::optional<int64_t> c = getConstantIntValue(slotValues[k]);
    bool u = c && curUniformSlots.contains(*c);
    lineUniform.push_back(u);
    if (u) {
      uniformTemps.insert(t);
    }
    current[k] = t;
    return t;
  }
  void write(int k, const std::string &expr, bool u = false) {
    std::string t = fresh();
    lines.push_back(llvm::formatv("sfpi::vFloat {0} = {1};", t, expr).str());
    lineUniform.push_back(u);
    if (u) {
      uniformTemps.insert(t);
    }
    assign(k, t);
  }
  bool isUniform(const std::string &t) const {
    return uniformTemps.contains(t);
  }
  std::set<int64_t> curUniformSlots;
  std::set<std::string> uniformTemps;
  std::vector<bool> lineUniform;
  bool hasPairOp = false;
  void assign(int k, const std::string &t) {
    current[k] = t;
    written.insert(k);
  }

  SmallVector<Value> slotValues;
  llvm::DenseMap<int64_t, int> constSlots;
  llvm::DenseMap<Value, int> valueSlots;
  SmallVector<Value> params;
  llvm::DenseMap<Value, int> paramIndex;
  std::map<int, std::string> current;
  llvm::DenseSet<int> written;
  SmallVector<std::string> lines;
  int numTemps = 0;
  int numOps = 0;
};

} // namespace

bool usesFp32Dst(func::FuncOp funcOp) {
  bool fp32 = false;
  auto visit = [&](Type t) {
    if (auto cb = dyn_cast<ttkernel::CBType>(t)) {
      if (auto tile = dyn_cast<ttcore::TileType>(cb.getElementType())) {
        fp32 |= ttcore::getNumberOfBits(tile.getDataType()) == 32;
      }
    }
  };
  funcOp.walk([&](Operation *op) {
    for (Type t : op->getResultTypes()) {
      visit(t);
    }
  });
  for (Type t : funcOp.getArgumentTypes()) {
    visit(t);
  }
  return fp32;
}

void fuseSFPUChains(func::FuncOp funcOp) {
  if (!usesFp32Dst(funcOp)) {
    return;
  }
  SmallVector<Block *> blocks;
  funcOp.walk([&](Block *b) { blocks.push_back(b); });
  int64_t tileRows = 0;
  if (auto rows = funcOp->getAttrOfType<IntegerAttr>("libtt.tile_rows")) {
    tileRows = rows.getInt();
  }
  for (Block *block : blocks) {
    SmallVector<Operation *> run;
    bool runCol0 = false, runHasOps = false;
    auto flush = [&]() {
      bool col0 = runCol0;
      runCol0 = runHasOps = false;
      if (run.empty()) {
        return;
      }
      SFPUChainEmitter emitter;
      emitter.col0Only = col0;
      emitter.tileRows = tileRows;
      Operation *last = nullptr;
      for (Operation *op : run) {
        if (!SFPUChainEmitter::isInit(op)) {
          emitter.add(op);
          last = op;
        }
      }
      bool emitted = last &&
                     (emitter.getNumOps() >= 2 ||
                      ((col0 || tileRows > 0) && emitter.getNumOps() >= 1)) &&
                     emitter.emit(last->getNextNode());
      // A chain with heavy ops (group sums, exp-based ops) that does not fit:
      // chain the parts around them (they stay LLK / helper calls).
      auto heavy = [](Operation *op) {
        return isa<ttkernel::LibttGroupSumTileOp, ttkernel::SiluTileOp,
                   ttkernel::SigmoidTileOp, ttkernel::ExpTileOp>(op);
      };
      auto heavyInit = [](Operation *op) {
        return isa<ttkernel::LibttGroupSumTileInitOp, ttkernel::SiluTileInitOp,
                   ttkernel::SigmoidTileInitOp, ttkernel::ExpTileInitOp>(op);
      };
      if (!emitted && llvm::any_of(run, heavy)) {
        SmallVector<SmallVector<Operation *>> parts(1);
        for (Operation *op : run) {
          if (heavy(op) || heavyInit(op)) {
            if (!parts.back().empty()) {
              parts.emplace_back();
            }
            continue;
          }
          parts.back().push_back(op);
        }
        for (auto &part : parts) {
          SFPUChainEmitter sub;
          sub.col0Only = col0;
          sub.tileRows = tileRows;
          Operation *subLast = nullptr;
          for (Operation *op : part) {
            if (!SFPUChainEmitter::isInit(op)) {
              sub.add(op);
              subLast = op;
            }
          }
          if (subLast && sub.getNumOps() >= 2 &&
              sub.emit(subLast->getNextNode())) {
            for (Operation *op : part) {
              if (op == subLast || op->isBeforeInBlock(subLast)) {
                op->erase();
              }
            }
          }
        }
        run.clear();
        return;
      }
      if (emitted) {
        // Inits after the last fused op belong to whatever follows.
        for (Operation *op : run) {
          if (op == last || op->isBeforeInBlock(last)) {
            op->erase();
          }
        }
      }
      run.clear();
    };
    for (Operation &op : llvm::make_early_inc_range(*block)) {
      if (SFPUChainEmitter::isInit(&op)) {
        run.push_back(&op);
        continue;
      }
      SFPUChainEmitter probe;
      if (probe.add(&op)) {
        bool col0 = op.hasAttr("libtt.col0_only");
        if (runHasOps && col0 != runCol0) {
          // Keep the trailing inits with the op they set up.
          SmallVector<Operation *> inits;
          while (!run.empty() && SFPUChainEmitter::isInit(run.back())) {
            inits.insert(inits.begin(), run.pop_back_val());
          }
          flush();
          run = inits;
        }
        runCol0 = col0;
        runHasOps = true;
        run.push_back(&op);
        continue;
      }
      if (isPure(&op) && op.getNumRegions() == 0) {
        continue;
      }
      // CB bookkeeping does not touch DST: a chain may span it.
      if (isa<ttkernel::CBPopFrontOp, ttkernel::CBWaitFrontOp,
              ttkernel::CBReserveBackOp, ttkernel::CBPushBackOp>(op)) {
        continue;
      }
      flush();
    }
    flush();
  }
}
namespace {
// Keep a whole kernel in DST.
//
// D2M lowers a merged row subgraph phase by phase: each phase loads its
// operands from CBs into DST, computes, and packs results into L1; a later
// phase reloads them. For small rows the values fit in DST for the whole
// kernel (8 FP32 tiles with full-sync DST). This pass fully unrolls the
// kernel's tile loops, follows DST tile values through the straight-line
// code, turns packs into and loads from compute-internal CBs into values that
// stay in DST (a broadcast read becomes an in-DST broadcast), allocates DST
// slices over the whole kernel with one acquire, and packs the kernel's
// outputs at the end. A kernel that does not fit is left unchanged.
struct DstResidentRewriter {
  func::FuncOp funcOp;
  // Move loads from input CBs to the top (after the acquire), so the
  // compute ops form longer SFPU chains. apply() retries without it when the
  // values do not fit.
  bool hoistLoads = false;
  static constexpr int kCapacity = 8;

  enum class Kind { Load, Bcast, Compute, Fill };
  struct Val {
    Operation *def = nullptr;
    int lastUse = -1;
    int slot = -1;
    // Equal in the even and odd vector of each band (constant within each
    // pair of adjacent columns): column broadcasts, fills, group sums and
    // elementwise ops on such values.
    bool pairUniform = false;
  };
  struct Access {
    int readSlot0 = -1, readSlot1 = -1, writeSlot = -1;
    bool inPlace = false;
  };

  std::vector<Val> vals;

  static std::optional<int64_t> ci(Value v) { return getConstantIntValue(v); }

  // DST operand roles of a TTKernel op; nullopt if unknown.
  static std::optional<Access> accessOf(Operation *op) {
    Access a;
    auto idx = [&](Value v) -> int {
      auto c = ci(v);
      return c ? static_cast<int>(*c) : -2;
    };
    if (auto c = dyn_cast<ttkernel::CopyTileOp>(op)) {
      a.writeSlot = idx(c.getTileIndexDst());
      return a;
    }
    if (isa<ttkernel::UnaryBcastTileOp>(op)) {
      a.writeSlot = idx(op->getOperand(2));
      return a;
    }
    if (auto p = dyn_cast<ttkernel::PackTileOp>(op)) {
      a.readSlot0 = idx(p.getDstIndex());
      return a;
    }
    if (auto f = dyn_cast<ttkernel::FillTileOp>(op)) {
      a.writeSlot = idx(f.getDstIndex());
      return a;
    }
    if (!op->hasTrait<ttkernel::TTKernelSFPUOpTrait>()) {
      bool usesIndex = llvm::any_of(op->getOperandTypes(),
                                    [](Type t) { return t.isIndex(); });
      return usesIndex ? std::nullopt : std::optional<Access>(a);
    }
    SmallVector<Value> indices;
    for (Value v : op->getOperands()) {
      if (v.getType().isIndex()) {
        indices.push_back(v);
      }
    }
    if (indices.size() == 3) {
      a.readSlot0 = idx(indices[0]);
      a.readSlot1 = idx(indices[1]);
      a.writeSlot = idx(indices[2]);
      return a;
    }
    if (indices.size() == 1) {
      a.readSlot0 = idx(indices[0]);
      a.writeSlot = a.readSlot0;
      a.inPlace = true;
      return a;
    }
    return std::nullopt;
  }

  static bool isDstOp(Operation *op) {
    return isa<ttkernel::CopyTileOp, ttkernel::UnaryBcastTileOp,
               ttkernel::PackTileOp, ttkernel::FillTileOp>(op) ||
           op->hasTrait<ttkernel::TTKernelSFPUOpTrait>();
  }

  static Value cbOf(Operation *op) {
    if (auto c = dyn_cast<ttkernel::CopyTileOp>(op)) {
      return c.getCb0();
    }
    if (isa<ttkernel::UnaryBcastTileOp>(op)) {
      return op->getOperand(0);
    }
    if (auto p = dyn_cast<ttkernel::PackTileOp>(op)) {
      return p.getOutCb();
    }
    return Value();
  }

  // Moves each load from an input CB (with its unpacker setup ops) to just
  // after the first acquire, in order, when its CB was waited on before the
  // acquire and not popped before the load.
  void hoistInputLoads(Block &block, const llvm::DenseSet<Value> &internal) {
    Operation *acquire = nullptr;
    llvm::DenseMap<Value, int64_t> waited;
    for (Operation &op : block) {
      if (isa<ttkernel::TileRegsAcquireOp>(op)) {
        acquire = &op;
        break;
      }
      if (auto wait = dyn_cast<ttkernel::CBWaitFrontOp>(op)) {
        if (auto n = ci(wait->getOperand(1))) {
          waited[wait->getOperand(0)] = *n;
        }
      }
    }
    if (!acquire) {
      return;
    }
    llvm::DenseSet<Value> popped;
    SmallVector<SmallVector<Operation *>> groups;
    for (Operation *op = acquire->getNextNode(); op; op = op->getNextNode()) {
      if (auto pop = dyn_cast<ttkernel::CBPopFrontOp>(op)) {
        popped.insert(pop->getOperand(0));
        continue;
      }
      if (!isa<ttkernel::CopyTileOp, ttkernel::UnaryBcastTileOp>(op)) {
        continue;
      }
      Value cb = cbOf(op);
      auto tile = ci(op->getOperand(1));
      if (internal.contains(cb) || !waited.contains(cb) ||
          popped.contains(cb) || !tile || *tile >= waited.lookup(cb)) {
        continue;
      }
      SmallVector<Operation *> group{op};
      bool hasInit = false;
      for (Operation *prev = op->getPrevNode(); prev;
           prev = prev->getPrevNode()) {
        if (isa<ttkernel::CopyTileInitOp, ttkernel::UnaryBcastInitOp>(prev) &&
            prev->getOperand(0) == cb) {
          hasInit = true;
          group.insert(group.begin(), prev);
          continue;
        }
        if (isa<ttkernel::ReconfigDataFormatSrcAOp,
                ttkernel::ReconfigDataFormatSrcBOp>(prev) &&
            prev->getOperand(0) == cb) {
          group.insert(group.begin(), prev);
          continue;
        }
        break;
      }
      if (!hasInit) {
        continue;
      }
      if (isa<ttkernel::UnaryBcastTileOp>(op)) {
        Operation *next = op->getNextNode();
        if (!next || !isa<ttkernel::UnaryBcastUninitOp>(next)) {
          continue;
        }
        group.push_back(next);
      }
      groups.push_back(group);
    }
    // Plain copies first, then broadcasts (fewer unpacker reconfigurations:
    // rel 3.16 -> 3.04 us).
    std::stable_partition(groups.begin(), groups.end(), [](auto &group) {
      return llvm::none_of(group, [](Operation *op) {
        return isa<ttkernel::UnaryBcastTileOp>(op);
      });
    });
    Operation *insertAfter = acquire;
    for (auto &group : groups) {
      for (Operation *op : group) {
        for (Value v : op->getOperands()) {
          Operation *def = v.getDefiningOp();
          if (def && def->getBlock() == &block &&
              insertAfter->isBeforeInBlock(def)) {
            if (!isa<arith::ConstantOp>(def)) {
              return;
            }
            def->moveBefore(acquire);
          }
        }
      }
      for (Operation *op : group) {
        op->moveAfter(insertAfter);
        insertAfter = op;
      }
    }
  }

  bool unrollAndFold() {
    // Innermost loops first; give up on large kernels.
    int64_t budget = 256;
    SmallVector<scf::ForOp> loops;
    funcOp.walk([&](scf::ForOp f) { loops.push_back(f); });
    for (scf::ForOp f : loops) {
      auto lb = ci(f.getLowerBound()), ub = ci(f.getUpperBound()),
           st = ci(f.getStep());
      if (!lb || !ub || !st || *st <= 0) {
        return false;
      }
      budget -= (*ub - *lb) / *st;
      if (budget < 0) {
        return false;
      }
    }
    for (scf::ForOp f : loops) {
      if (failed(loopUnrollFull(f))) {
        return false;
      }
    }
    RewritePatternSet patterns(funcOp.getContext());
    scf::IfOp::getCanonicalizationPatterns(patterns, funcOp.getContext());
    (void)applyPatternsGreedily(funcOp, std::move(patterns));
    return true;
  }

  bool run() {
    if (funcOp.getBody().getBlocks().size() != 1) {
      return false;
    }
    Block &block = funcOp.getBody().front();
    // Only kernels with a compute-internal CB (packed and read back here).
    llvm::DenseSet<Value> packed, loaded;
    funcOp.walk([&](Operation *op) {
      if (isa<ttkernel::PackTileOp>(op)) {
        packed.insert(cbOf(op));
      } else if (isa<ttkernel::CopyTileOp, ttkernel::UnaryBcastTileOp>(op)) {
        loaded.insert(cbOf(op));
      }
    });
    llvm::DenseSet<Value> internal;
    for (Value c : packed) {
      if (loaded.contains(c)) {
        internal.insert(c);
      }
    }
    // Without compute-internal CBs, only small kernels (they lose the
    // pack/math overlap of half-sync DST).
    if (internal.empty()) {
      int64_t packs = 0;
      funcOp.walk([&](ttkernel::PackTileOp) { ++packs; });
      bool loops = false;
      funcOp.walk([&](scf::ForOp) { loops = true; });
      if (!hoistLoads || loops || packs > 4) {
        return false;
      }
    }
    // (apply() runs this on a clone and keeps it only on success.)
    if (!unrollAndFold()) {
      return false;
    }
    // A CB is internal only if everything packed into it is read back here:
    // the kernel's last use of it is a read. (A reduction accumulator packed,
    // reloaded and finally packed for the consumer escapes.)
    for (Value cb : internal) {
      Operation *last = nullptr;
      for (Operation &op : block) {
        if (isa<ttkernel::PackTileOp, ttkernel::CBPushBackOp,
                ttkernel::CBPopFrontOp, ttkernel::CopyTileOp,
                ttkernel::UnaryBcastTileOp>(&op) &&
            llvm::is_contained(op.getOperands(), cb)) {
          last = &op;
        }
      }
      if (!last || isa<ttkernel::PackTileOp, ttkernel::CBPushBackOp>(last)) {
        return false;
      }
    }
    SmallVector<Operation *> ops;
    for (Operation &op : block.without_terminator()) {
      if (op.getNumRegions() != 0) {
        return false;
      }
      ops.push_back(&op);
    }

    // Values: slot -> current value; (cb, tile) -> value.
    std::map<int, int> slotVal;
    // Repeated loads of the same CB tile and repeated broadcasts of one value
    // (D2M emits them per output subblock) reuse the first value.
    std::map<std::tuple<const void *, int64_t, int>, int> externalLoads;
    std::map<std::pair<int, int>, int> internalBcasts;
    // Input CBs waited on and popped once: their tiles stay readable until
    // the kernel ends once the pop moves there.
    llvm::DenseSet<Value> reloadable, delayedPops;
    {
      llvm::DenseMap<Value, int> waits, pops;
      for (Operation *op : ops) {
        if (isa<ttkernel::CBWaitFrontOp>(op)) {
          ++waits[op->getOperand(0)];
        } else if (isa<ttkernel::CBPopFrontOp>(op)) {
          ++pops[op->getOperand(0)];
        }
      }
      for (auto [cb, n] : waits) {
        if (n == 1 && pops.lookup(cb) == 1 && !internal.contains(cb)) {
          reloadable.insert(cb);
        }
      }
    }
    auto cbKey = [](Value cb) { return cb.getAsOpaquePointer(); };
    std::map<std::pair<const void *, int64_t>, int> regionByCb;
    struct OpInfo {
      Operation *op;
      Access acc;
      int in0 = -1, in1 = -1, out = -1;
      bool erase = false, finalPack = false, bcastInternal = false;
      // A row (not column) broadcast of an internal value.
      bool bcastRow = false;
    };
    std::vector<OpInfo> infos;
    auto use = [&](int v, int at) {
      if (v >= 0) {
        vals[v].lastUse = std::max(vals[v].lastUse, at);
      }
    };
    int end = static_cast<int>(ops.size());
    for (size_t i = 0; i < ops.size(); ++i) {
      Operation *op = ops[i];
      OpInfo info{op, {}};
      int at = static_cast<int>(i);
      if (isa<ttkernel::TileRegsAcquireOp, ttkernel::TileRegsReleaseOp,
              ttkernel::TileRegsCommitOp, ttkernel::TileRegsWaitOp,
              ttkernel::UnpackStallOnPackOp>(op)) {
        info.erase = true;
        infos.push_back(info);
        continue;
      }
      if (isa<ttkernel::CBPopFrontOp>(op)) {
        const void *key = cbKey(op->getOperand(0));
        for (auto it = externalLoads.begin(); it != externalLoads.end();) {
          it = std::get<0>(it->first) == key ? externalLoads.erase(it)
                                             : std::next(it);
        }
      }
      // CB bookkeeping and format setup of internal CBs disappear.
      if (isa<ttkernel::CBPushBackOp, ttkernel::CBWaitFrontOp,
              ttkernel::CBPopFrontOp, ttkernel::CBReserveBackOp,
              ttkernel::ReconfigDataFormatSrcAOp,
              ttkernel::ReconfigDataFormatSrcBOp,
              ttkernel::PackReconfigDataFormatOp, ttkernel::CopyTileInitOp,
              ttkernel::UnaryBcastInitOp, ttkernel::UnaryBcastUninitOp>(op) &&
          internal.contains(op->getOperand(0))) {
        info.erase = true;
        infos.push_back(info);
        continue;
      }
      if (!isDstOp(op)) {
        if (llvm::any_of(op->getOperandTypes(),
                         [](Type t) { return t.isIndex(); }) &&
            !isPure(op)) {
          // An unknown op reading DST indices.
          if (!isa<ttkernel::CBPushBackOp, ttkernel::CBWaitFrontOp,
                   ttkernel::CBPopFrontOp, ttkernel::CBReserveBackOp>(op)) {
            return false;
          }
        }
        infos.push_back(info);
        continue;
      }
      auto acc = accessOf(op);
      if (!acc || acc->readSlot0 == -2 || acc->readSlot1 == -2 ||
          acc->writeSlot == -2 || acc->writeSlot >= 16 ||
          acc->readSlot0 >= 16 || acc->readSlot1 >= 16) {
        return false;
      }
      info.acc = *acc;
      auto cur = [&](int slot) -> int {
        auto it = slotVal.find(slot);
        return it == slotVal.end() ? -1 : it->second;
      };
      auto fresh = [&](Operation *def) {
        vals.push_back(Val{def, -1, -1});
        return static_cast<int>(vals.size() - 1);
      };
      if (auto copy = dyn_cast<ttkernel::CopyTileOp>(op)) {
        auto tile = ci(copy.getTileIndexCb());
        if (internal.contains(copy.getCb0())) {
          if (!tile) {
            return false;
          }
          auto it = regionByCb.find({cbKey(copy.getCb0()), *tile});
          if (it == regionByCb.end()) {
            return false;
          }
          // A value that is a plain copy of an input tile is reloaded from
          // that input instead of kept in DST (its pop moves to the end).
          auto src =
              dyn_cast_or_null<ttkernel::CopyTileOp>(vals[it->second].def);
          if (src && reloadable.contains(src.getCb0())) {
            OpBuilder rb(op);
            rb.create<ttkernel::ReconfigDataFormatSrcAOp>(op->getLoc(),
                                                          src.getCb0());
            rb.create<ttkernel::CopyTileInitOp>(op->getLoc(), src.getCb0());
            copy->setOperand(0, src.getCb0());
            copy->setOperand(1, src.getTileIndexCb());
            delayedPops.insert(src.getCb0());
            info.out = fresh(op);
            slotVal[acc->writeSlot] = info.out;
            infos.push_back(info);
            continue;
          }
          slotVal[acc->writeSlot] = it->second;
          info.erase = true;
        } else {
          auto key = std::make_tuple(cbKey(copy.getCb0()),
                                     tile ? *tile : int64_t(-1), -1);
          auto it = tile ? externalLoads.find(key) : externalLoads.end();
          if (it != externalLoads.end()) {
            slotVal[acc->writeSlot] = it->second;
            info.erase = true;
          } else {
            info.out = fresh(op);
            slotVal[acc->writeSlot] = info.out;
            if (tile) {
              externalLoads[key] = info.out;
            }
          }
        }
      } else if (isa<ttkernel::UnaryBcastTileOp>(op) &&
                 !internal.contains(op->getOperand(0))) {
        auto tile = ci(op->getOperand(1));
        auto type = op->getAttrOfType<ttkernel::BcastTypeAttr>("bcast_type");
        auto key = std::make_tuple(cbKey(op->getOperand(0)),
                                   tile ? *tile : int64_t(-1),
                                   static_cast<int>(type.getValue()));
        auto it = tile ? externalLoads.find(key) : externalLoads.end();
        if (it != externalLoads.end()) {
          slotVal[acc->writeSlot] = it->second;
          info.erase = true;
        } else {
          info.out = fresh(op);
          slotVal[acc->writeSlot] = info.out;
          if (tile) {
            externalLoads[key] = info.out;
          }
        }
      } else if (isa<ttkernel::UnaryBcastTileOp>(op)) {
        {
          auto tile = ci(op->getOperand(1));
          auto it = tile ? regionByCb.find({cbKey(op->getOperand(0)), *tile})
                         : regionByCb.end();
          auto type = op->getAttrOfType<ttkernel::BcastTypeAttr>("bcast_type");
          if (it == regionByCb.end() || !type ||
              (type.getValue() != ttkernel::BcastType::Col &&
               type.getValue() != ttkernel::BcastType::Row)) {
            return false;
          }
          bool row = type.getValue() == ttkernel::BcastType::Row;
          info.bcastRow = row;
          auto hit = internalBcasts.find({it->second, row ? 2 : 1});
          if (hit != internalBcasts.end()) {
            slotVal[acc->writeSlot] = hit->second;
            info.erase = true;
            infos.push_back(info);
            continue;
          }
          info.in0 = it->second;
          use(info.in0, at);
          info.bcastInternal = true;
        }
        info.out = fresh(op);
        internalBcasts[{info.in0, info.bcastRow ? 2 : 1}] = info.out;
        slotVal[acc->writeSlot] = info.out;
      } else if (auto pack = dyn_cast<ttkernel::PackTileOp>(op)) {
        int v = cur(acc->readSlot0);
        if (v < 0) {
          return false;
        }
        auto tile = ci(pack.getOutIndex());
        if (internal.contains(pack.getOutCb())) {
          if (!tile) {
            return false;
          }
          regionByCb[{cbKey(pack.getOutCb()), *tile}] = v;
          info.erase = true;
        } else {
          info.in0 = v;
          info.finalPack = true;
          use(v, end);
        }
      } else if (isa<ttkernel::FillTileOp>(op)) {
        info.out = fresh(op);
        slotVal[acc->writeSlot] = info.out;
      } else {
        info.in0 = acc->readSlot0 >= 0 ? cur(acc->readSlot0) : -1;
        info.in1 = acc->readSlot1 >= 0 ? cur(acc->readSlot1) : -1;
        if ((acc->readSlot0 >= 0 && info.in0 < 0) ||
            (acc->readSlot1 >= 0 && info.in1 < 0)) {
          return false;
        }
        use(info.in0, at);
        use(info.in1, at);
        info.out = fresh(op);
        slotVal[acc->writeSlot] = info.out;
      }
      infos.push_back(info);
    }

    // Hoisting moves loads after their values are identified (the slots D2M
    // assigned assume the original order); the allocation below follows the
    // new order. Pops of inputs read again later move to the end first.
    for (Operation *op : ops) {
      if (isa<ttkernel::CBPopFrontOp>(op) &&
          delayedPops.contains(op->getOperand(0))) {
        op->moveBefore(block.getTerminator());
      }
    }
    if (hoistLoads || !delayedPops.empty()) {
      if (hoistLoads) {
        hoistInputLoads(block, internal);
      }
      std::stable_sort(infos.begin(), infos.end(),
                       [](const OpInfo &a, const OpInfo &b) {
                         return a.op->isBeforeInBlock(b.op);
                       });
      for (Val &v : vals) {
        v.lastUse = -1;
      }
      for (size_t i = 0; i < infos.size(); ++i) {
        use(infos[i].in0, infos[i].finalPack ? end : static_cast<int>(i));
        use(infos[i].in1, static_cast<int>(i));
      }
    }

    // Column-0 demand: a row reduction's result matters in column 0 only, so
    // do values read only through a column broadcast (and the elementwise
    // ops computing them). Those ops compute column 0 only (col0_only
    // chains); a column broadcast whose result is itself only needed in
    // column 0 is the identity.
    {
      enum { kNone = 0, kCol0 = 1, kFull = 2 };
      std::vector<int> demand(vals.size(), kNone);
      auto need = [&](int v, int d) {
        if (v >= 0) {
          demand[v] = std::max(demand[v], d);
        }
      };
      auto elementwise = [](Operation *op) {
        SFPUChainEmitter probe;
        return !isa<ttkernel::LibttGroupSumTileOp>(op) && probe.add(op);
      };
      for (int i = static_cast<int>(infos.size()) - 1; i >= 0; --i) {
        OpInfo &info = infos[i];
        if (info.finalPack) {
          need(info.in0, kFull);
        } else if (info.bcastInternal) {
          need(info.in0, info.out >= 0 && demand[info.out] == kNone ? kNone
                         : info.bcastRow                            ? kFull
                                                                    : kCol0);
        } else if (info.out >= 0 && demand[info.out] != kNone) {
          int d = demand[info.out];
          if (!elementwise(info.op)) {
            d = kFull;
          }
          need(info.in0, d);
          need(info.in1, d);
        }
      }
      std::vector<int> alias(vals.size());
      for (size_t v = 0; v < vals.size(); ++v) {
        alias[v] = static_cast<int>(v);
      }
      auto resolve = [&](int v) { return v < 0 ? v : alias[v]; };
      for (OpInfo &info : infos) {
        info.in0 = resolve(info.in0);
        info.in1 = resolve(info.in1);
        if (info.bcastInternal && !info.bcastRow && demand[info.out] == kCol0) {
          alias[info.out] = info.in0;
          info.erase = true;
          info.bcastInternal = false;
          info.out = info.in0 = -1;
          continue;
        }
        if (info.out >= 0 && demand[info.out] == kCol0 &&
            elementwise(info.op)) {
          info.op->setAttr("libtt.col0_only",
                           UnitAttr::get(funcOp.getContext()));
        }
      }
      for (Val &v : vals) {
        v.lastUse = -1;
      }
      for (size_t i = 0; i < infos.size(); ++i) {
        use(infos[i].in0, infos[i].finalPack ? end : static_cast<int>(i));
        use(infos[i].in1, static_cast<int>(i));
      }
    }

    // Loads, broadcasts and fills nobody reads (overwritten first) go away.
    for (OpInfo &info : infos) {
      if (info.out >= 0 && vals[info.out].lastUse < 0 &&
          isa<ttkernel::CopyTileOp, ttkernel::UnaryBcastTileOp,
              ttkernel::FillTileOp>(info.op)) {
        info.erase = true;
        info.out = -1;
        info.in0 = -1;
      }
    }

    // Linear-scan allocation over the whole kernel.
    std::vector<bool> busy(kCapacity, false);
    auto take = [&]() -> int {
      for (int s = 0; s < kCapacity; ++s) {
        if (!busy[s]) {
          busy[s] = true;
          return s;
        }
      }
      return -1;
    };
    struct Copy {
      int at, from, to;
    };
    std::vector<Copy> copies;
    for (size_t i = 0; i < infos.size(); ++i) {
      OpInfo &info = infos[i];
      int at = static_cast<int>(i);
      if (info.out >= 0) {
        Val &out = vals[info.out];
        int in = info.acc.inPlace ? info.in0 : -1;
        if (info.acc.inPlace) {
          if (vals[in].lastUse == at) {
            out.slot = vals[in].slot;
          } else {
            out.slot = take();
            if (out.slot < 0) {
              return false;
            }
            copies.push_back({at, vals[in].slot, out.slot});
          }
        } else {
          // A binary op may write over an input dying here.
          for (int v : {info.in0, info.in1}) {
            if (out.slot < 0 && v >= 0 && vals[v].lastUse == at &&
                !info.bcastInternal) {
              out.slot = vals[v].slot;
            }
          }
          if (out.slot < 0) {
            out.slot = take();
          }
          if (out.slot < 0) {
            return false;
          }
        }
      }
      // Free values whose last use is this op (unless the slot passed on).
      for (int v : {info.in0, info.in1}) {
        if (v >= 0 && vals[v].lastUse == at && vals[v].slot >= 0 &&
            (info.out < 0 || vals[info.out].slot != vals[v].slot)) {
          busy[vals[v].slot] = false;
        }
      }
      if (info.out >= 0 && vals[info.out].lastUse < 0) {
        busy[vals[info.out].slot] = false;
      }
    }

    for (OpInfo &info : infos) {
      if (info.out < 0 || info.erase) {
        continue;
      }
      Operation *op = info.op;
      bool u = false;
      auto bcastType = op->getAttrOfType<ttkernel::BcastTypeAttr>("bcast_type");
      if (info.bcastInternal) {
        u = !info.bcastRow;
      } else if (isa<ttkernel::UnaryBcastTileOp>(op)) {
        u = bcastType && (bcastType.getValue() == ttkernel::BcastType::Col ||
                          bcastType.getValue() == ttkernel::BcastType::Scalar);
      } else if (isa<ttkernel::FillTileOp, ttkernel::LibttGroupSumTileOp,
                     ttkernel::LibttRowToColumnTileOp>(op)) {
        u = true;
      } else if (SFPUChainEmitter().add(op)) {
        u = (info.in0 < 0 || vals[info.in0].pairUniform) &&
            (info.in1 < 0 || vals[info.in1].pairUniform);
      }
      vals[info.out].pairUniform = u;
    }

    // Rewrite.
    OpBuilder b(funcOp.getContext());
    auto idxConst = [&](Operation *at, int v) -> Value {
      b.setInsertionPoint(at);
      return b.create<arith::ConstantIndexOp>(at->getLoc(), v);
    };
    auto helperCall = [&](Operation *at, StringRef fn, int from, int to) {
      b.setInsertionPoint(at);
      OperationState state(at->getLoc(), emitc::VerbatimOp::getOperationName());
      state.addAttribute(
          "value", b.getStringAttr(llvm::formatv("MATH((libtt_sfpu::{0}<{1}, "
                                                 "{2}>()));",
                                                 fn, from, to)
                                       .str()));
      state.addAttribute("libtt.sfpu_def",
                         b.getStringAttr("// libtt in-DST tile helpers"));
      b.create(state);
    };
    Operation *terminator = block.getTerminator();
    SmallVector<Operation *> tail;
    Operation *firstAcquire = nullptr;
    for (OpInfo &info : infos) {
      if (isa<ttkernel::TileRegsAcquireOp>(info.op) && !firstAcquire) {
        firstAcquire = info.op;
      }
    }
    if (!firstAcquire) {
      return false;
    }
    b.setInsertionPoint(firstAcquire);
    b.create<ttkernel::TileRegsAcquireOp>(firstAcquire->getLoc());
    for (size_t i = 0; i < infos.size(); ++i) {
      OpInfo &info = infos[i];
      int at = static_cast<int>(i);
      for (const Copy &c : copies) {
        if (c.at == at) {
          helperCall(info.op, "copy_dst", c.from, c.to);
        }
      }
      if (info.erase) {
        continue;
      }
      Operation *op = info.op;
      if (info.bcastInternal) {
        helperCall(op, info.bcastRow ? "bcast_row0" : "bcast_col0",
                   vals[info.in0].slot, vals[info.out].slot);
        info.erase = true;
        continue;
      }
      if (info.finalPack) {
        auto pack = cast<ttkernel::PackTileOp>(op);
        pack->setOperand(0, idxConst(op, vals[info.in0].slot));
        tail.push_back(op);
        continue;
      }
      if (!isDstOp(op)) {
        continue;
      }
      // Renumber this op's DST operands.
      if (auto copy = dyn_cast<ttkernel::CopyTileOp>(op)) {
        copy->setOperand(2, idxConst(op, vals[info.out].slot));
      } else if (isa<ttkernel::UnaryBcastTileOp>(op)) {
        op->setOperand(2, idxConst(op, vals[info.out].slot));
      } else if (auto fill = dyn_cast<ttkernel::FillTileOp>(op)) {
        fill->setOperand(0, idxConst(op, vals[info.out].slot));
      } else {
        SmallVector<OpOperand *> idxOperands;
        for (OpOperand &operand : op->getOpOperands()) {
          if (operand.get().getType().isIndex()) {
            idxOperands.push_back(&operand);
          }
        }
        if (idxOperands.size() == 3) {
          idxOperands[0]->set(idxConst(op, vals[info.in0].slot));
          idxOperands[1]->set(idxConst(op, vals[info.in1].slot));
          idxOperands[2]->set(idxConst(op, vals[info.out].slot));
        } else if (idxOperands.size() == 1) {
          idxOperands[0]->set(idxConst(op, vals[info.out].slot));
        }
        SmallVector<int64_t> uniformSlots;
        for (int v : {info.in0, info.in1}) {
          if (v >= 0 && vals[v].pairUniform) {
            uniformSlots.push_back(vals[v].slot);
          }
        }
        if (!uniformSlots.empty()) {
          op->setAttr("libtt.pu_slots", b.getDenseI64ArrayAttr(uniformSlots));
        }
      }
    }
    // Packs of the kernel's outputs (with their pack reconfig and the
    // following push) at the end, after commit / wait (packs between an
    // acquire and a release are self-managed: TTKernelControlDstSection adds
    // nothing), then the release.
    b.setInsertionPoint(terminator);
    b.create<ttkernel::TileRegsCommitOp>(terminator->getLoc());
    b.create<ttkernel::TileRegsWaitOp>(terminator->getLoc());
    for (Operation *op : tail) {
      Operation *prev = op->getPrevNode();
      while (prev && isPure(prev)) {
        prev = prev->getPrevNode();
      }
      if (prev && isa<ttkernel::PackReconfigDataFormatOp>(prev)) {
        prev->moveBefore(terminator);
      }
      op->moveBefore(terminator);
    }
    // Output CB pushes after the moved packs.
    for (Operation *op : tail) {
      Value cb = cast<ttkernel::PackTileOp>(op).getOutCb();
      for (Operation *user : llvm::make_early_inc_range(cb.getUsers())) {
        if (isa<ttkernel::CBPushBackOp>(user)) {
          user->moveBefore(terminator);
        }
      }
    }
    b.setInsertionPoint(terminator);
    b.create<ttkernel::TileRegsReleaseOp>(terminator->getLoc());
    // A load that goes away (dead, reused or reloaded elsewhere) takes its
    // unpacker setup with it (a leftover reconfig + init cost rel 0.12 us).
    llvm::SetVector<Operation *> setup;
    for (OpInfo &info : infos) {
      if (!info.erase ||
          !isa<ttkernel::CopyTileOp, ttkernel::UnaryBcastTileOp>(info.op)) {
        continue;
      }
      Value cb = info.op->getOperand(0);
      for (Operation *prev = info.op->getPrevNode(); prev;
           prev = prev->getPrevNode()) {
        if (isa<arith::ConstantOp>(prev)) {
          continue;
        }
        if (!isa<ttkernel::ReconfigDataFormatSrcAOp,
                 ttkernel::ReconfigDataFormatSrcBOp, ttkernel::CopyTileInitOp,
                 ttkernel::UnaryBcastInitOp>(prev) ||
            prev->getOperand(0) != cb) {
          break;
        }
        setup.insert(prev);
      }
      Operation *next = info.op->getNextNode();
      if (next && isa<ttkernel::UnaryBcastUninitOp>(next) &&
          next->getOperand(0) == cb) {
        setup.insert(next);
      }
    }
    for (OpInfo &info : infos) {
      if (info.erase) {
        setup.remove(info.op);
        info.op->erase();
      }
    }
    for (Operation *op : setup) {
      op->erase();
    }
    return true;
  }

  // Rewrites a clone of `original`; on success the clone's body replaces the
  // original's.
  static bool apply(func::FuncOp original) {
    func::FuncOp clone = original.clone();
    bool ok = DstResidentRewriter{clone, /*hoistLoads=*/true}.run();
    if (!ok) {
      clone.erase();
      clone = original.clone();
      ok = DstResidentRewriter{clone}.run();
    }
    if (ok) {
      original.getBody().takeBody(clone.getBody());
      original->setAttr("libtt.dst_full_sync",
                        UnitAttr::get(original.getContext()));
    }
    clone.erase();
    return ok;
  }
};
} // namespace

bool keepKernelInDst(func::FuncOp funcOp) {
  return DstResidentRewriter::apply(funcOp);
}

} // namespace mlir::tt::d2m_fusion
