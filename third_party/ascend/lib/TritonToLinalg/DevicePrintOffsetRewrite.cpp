/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */


#include "ascend/include/TritonToLinalg/DevicePrintOffsetRewrite.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <functional>
#include <optional>

#define DEBUG_TYPE "device-print-offset-rewrite"

namespace mlir {
namespace triton {
namespace {

static bool hasOnlyPrintUsers(Value v) {
  if (v.use_empty()) return false;
  for (Operation *u : v.getUsers()) {
    auto callOp = dyn_cast<func::CallOp>(u);
    if (!callOp) return false;
    if (!callOp.getCallee().starts_with("triton_print")) return false;
  }
  return true;
}

static std::optional<int64_t> verifyChain1D(Value v) {
  auto rootTy = dyn_cast<RankedTensorType>(v.getType());
  if (!rootTy || rootTy.getRank() != 1) return std::nullopt;
  int64_t len = rootTy.getShape()[0];

  llvm::DenseSet<Operation *> known;
  std::function<bool(Value)> walk = [&](Value v) -> bool {
    auto ty = dyn_cast<RankedTensorType>(v.getType());
    if (!ty) return true;
    if (ty.getRank() != 1) return false;
    if (ty.getShape()[0] != len) return false;

    Operation *def = v.getDefiningOp();
    if (!def) return false;
    if (known.count(def)) return true;

    bool ok = false;
    if (auto fillOp = dyn_cast<linalg::FillOp>(def)) {
      if (fillOp.getInputs().empty()) ok = false;
      else                            ok = fillOp.getInputs()[0].getType().isIntOrIndex();
    } else if (isa<arith::ConstantOp>(def)) {
      ok = true;
    } else if (auto g = dyn_cast<linalg::GenericOp>(def)) {
      ok = g->hasAttr("tt.from_make_range");
    } else if (isa<tensor::EmptyOp>(def)) {
      ok = true;
    } else if (auto exp = dyn_cast<tensor::ExpandShapeOp>(def)) {
      auto resTy = ::mlir::cast<RankedTensorType>(exp.getResult().getType());
      ok = resTy.getRank() == 1 && walk(exp.getSrc());
    } else if (auto col = dyn_cast<tensor::CollapseShapeOp>(def)) {
      auto resTy = ::mlir::cast<RankedTensorType>(col.getResult().getType());
      ok = resTy.getRank() == 1 && walk(col.getSrc());
    } else if (isa<arith::MulIOp, arith::AddIOp, arith::SubIOp,
                    arith::RemSIOp, arith::DivSIOp,
                    arith::RemUIOp, arith::DivUIOp>(def)) {
      ok = true;
      for (Value operand : def->getOperands())
        if (!walk(operand)) { ok = false; break; }
    }
    if (ok) known.insert(def);
    return ok;
  };

  return walk(v) ? std::optional<int64_t>{len} : std::nullopt;
}

class ScalarChainEmitter {
public:
  ScalarChainEmitter(OpBuilder &bd, Location loc, Value iv)
      : bd(bd), loc(loc), iv(iv) {}

  Value emit(Value v) {
    auto it = memo.find(v);
    if (it != memo.end()) return it->second;
    Value r = emitImpl(v);
    memo[v] = r;
    return r;
  }

private:
  Value emitImpl(Value v) {
    Operation *def = v.getDefiningOp();
    if (!def) return nullptr;

    if (auto cstOp = dyn_cast<arith::ConstantOp>(def))
      return scalarFromConstant(cstOp);
    if (auto fillOp = dyn_cast<linalg::FillOp>(def))
      return scalarFromFill(fillOp);
    if (auto generic = dyn_cast<linalg::GenericOp>(def))
      return scalarFromMakeRange(generic);
    if (auto exp = dyn_cast<tensor::ExpandShapeOp>(def))
      return emit(exp.getSrc());
    if (auto col = dyn_cast<tensor::CollapseShapeOp>(def))
      return emit(col.getSrc());

    if (def->getNumOperands() != 2) return nullptr;
    Value lhs = emit(def->getOperand(0));
    Value rhs = emit(def->getOperand(1));
    if (!lhs || !rhs) return nullptr;

    if (isa<arith::MulIOp>(def))  return bd.create<arith::MulIOp>(loc, lhs, rhs);
    if (isa<arith::AddIOp>(def))  return bd.create<arith::AddIOp>(loc, lhs, rhs);
    if (isa<arith::SubIOp>(def))  return bd.create<arith::SubIOp>(loc, lhs, rhs);
    if (isa<arith::RemSIOp>(def)) return bd.create<arith::RemSIOp>(loc, lhs, rhs);
    if (isa<arith::DivSIOp>(def)) return bd.create<arith::DivSIOp>(loc, lhs, rhs);
    if (isa<arith::RemUIOp>(def)) return bd.create<arith::RemUIOp>(loc, lhs, rhs);
    if (isa<arith::DivUIOp>(def)) return bd.create<arith::DivUIOp>(loc, lhs, rhs);
    return nullptr;
  }

  Value scalarFromConstant(arith::ConstantOp op) {
    auto i32Ty = bd.getI32Type();
    if (auto dense = dyn_cast<DenseIntElementsAttr>(op.getValue()))
      if (dense.isSplat())
        return bd.create<arith::ConstantIntOp>(
            loc, dense.getSplatValue<APInt>().getSExtValue(), i32Ty);
    if (auto intAttr = dyn_cast<IntegerAttr>(op.getValue()))
      return bd.create<arith::ConstantIntOp>(loc, intAttr.getInt(), i32Ty);
    return nullptr;
  }

  Value scalarFromFill(linalg::FillOp op) {
    if (op.getInputs().empty()) return nullptr;
    Value sc = op.getInputs()[0];
    auto i32Ty = bd.getI32Type();
    auto scTy = sc.getType();

    if (auto cstOp = sc.getDefiningOp<arith::ConstantOp>()) {
      if (auto intAttr = dyn_cast<IntegerAttr>(cstOp.getValue()))
        return bd.create<arith::ConstantIntOp>(loc, intAttr.getInt(), i32Ty);
      return nullptr;
    }

    if (scTy.isInteger(32)) return sc;
    if (scTy.isIndex())
      return bd.create<arith::IndexCastOp>(loc, i32Ty, sc);
    if (auto intTy = dyn_cast<IntegerType>(scTy)) {
      if (intTy.getWidth() < 32)
        return bd.create<arith::ExtSIOp>(loc, i32Ty, sc);
      if (intTy.getWidth() > 32)
        return bd.create<arith::TruncIOp>(loc, i32Ty, sc);
    }
    return nullptr;
  }

  Value scalarFromMakeRange(linalg::GenericOp op) {
    auto i32Ty = bd.getI32Type();
    if (!op->hasAttr("tt.from_make_range")) return nullptr;
    int64_t off = 0;
    if (auto a = op->getAttrOfType<IntegerAttr>("tt.make_range_offset"))
      off = a.getInt();
    Value v32 = bd.create<arith::IndexCastOp>(loc, i32Ty, iv);
    if (off != 0) {
      Value oc = bd.create<arith::ConstantIntOp>(loc, off, i32Ty);
      v32 = bd.create<arith::AddIOp>(loc, v32, oc);
    }
    return v32;
  }

  OpBuilder &bd;
  Location loc;
  Value iv;
  llvm::DenseMap<Value, Value> memo;
};

static bool isLinearStructuralOp(Operation *op) {
  if (!op) return false;
  if (isa<linalg::FillOp, linalg::BroadcastOp>(op)) return true;
  if (auto g = dyn_cast<linalg::GenericOp>(op))
    return op->hasAttr("tt.from_make_range");
  if (isa<tensor::EmptyOp, tensor::ExpandShapeOp,
          tensor::CollapseShapeOp>(op))
    return true;
  if (isa<arith::MulIOp, arith::AddIOp, arith::ConstantOp>(op)) return true;
  return false;
}

static bool isLinearChain(Value v) {
  llvm::DenseSet<Operation *> visited;
  std::function<bool(Value)> walk = [&](Value v) -> bool {
    Operation *def = v.getDefiningOp();
    if (!def) return true;
    if (!visited.insert(def).second) return true;
    if (isa<arith::RemSIOp, arith::DivSIOp,
            arith::RemUIOp, arith::DivUIOp>(def))
      return false;
    if (!isLinearStructuralOp(def)) return false;
    for (Value operand : def->getOperands())
      if (!walk(operand)) return false;
    return true;
  };
  return walk(v);
}

static memref::ReinterpretCastOp findMatchingCast(
    func::FuncOp funcOp, ArrayRef<int64_t> targetShape) {
  memref::ReinterpretCastOp best;
  unsigned bestScore = 0;
  funcOp.walk([&](memref::ReinterpretCastOp c) {
    auto resTy = dyn_cast<MemRefType>(c.getResult().getType());
    if (!resTy) return;
    if (resTy.getShape() != targetShape) return;
    unsigned score = 0;
    if (auto blockArg = dyn_cast<BlockArgument>(c.getSource())) {
      unsigned idx = blockArg.getArgNumber();
      if (auto argDict = funcOp.getArgAttrDict(idx))
        if (argDict.contains("tt.tensor_kind")) score = 1;
    }
    if (score >= bestScore) {
      bestScore = score;
      best = c;
    }
  });
  return best;
}

static void emitChain1DForLoop(OpBuilder &builder, Location loc,
                                 func::FuncOp scalarPrintFn,
                                 Value chainRoot, int64_t length) {
  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value upper = builder.create<arith::ConstantIndexOp>(loc, length);
  auto forOp = builder.create<scf::ForOp>(loc, c0, upper, c1);

  OpBuilder bd = builder;
  bd.setInsertionPoint(forOp.getBody()->getTerminator());

  ScalarChainEmitter emitter(bd, loc, forOp.getInductionVar());
  Value scalar = emitter.emit(chainRoot);
  if (!scalar) {
    return;
  }
  bd.create<func::CallOp>(loc, scalarPrintFn, ValueRange{scalar});
}

static void emitCastMultiDimForLoops(OpBuilder &builder, Location loc,
                                       func::FuncOp scalarPrintFn,
                                       ArrayRef<int64_t> shape,
                                       ArrayRef<int64_t> strides,
                                       int64_t baseOffset) {
  auto i32Ty = builder.getI32Type();
  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);

  SmallVector<Value> ivs(shape.size(), c0);
  SmallVector<size_t> activeDims;
  for (size_t k = 0; k < shape.size(); ++k)
    if (shape[k] > 1) activeDims.push_back(k);

  OpBuilder bd = builder;
  for (size_t k : activeDims) {
    Value upper = bd.create<arith::ConstantIndexOp>(loc, shape[k]);
    auto forOp = bd.create<scf::ForOp>(loc, c0, upper, c1);
    ivs[k] = forOp.getInductionVar();
    bd.setInsertionPoint(forOp.getBody()->getTerminator());
  }

  Value acc;
  if (baseOffset != 0)
    acc = bd.create<arith::ConstantIndexOp>(loc, baseOffset);
  for (size_t k = 0; k < shape.size(); ++k) {
    if (shape[k] == 1) continue;
    Value idx = ivs[k];
    if (strides[k] != 1) {
      Value s = bd.create<arith::ConstantIndexOp>(loc, strides[k]);
      idx = bd.create<arith::MulIOp>(loc, idx, s);
    }
    if (!acc) acc = idx;
    else      acc = bd.create<arith::AddIOp>(loc, acc, idx);
  }
  if (!acc) acc = bd.create<arith::ConstantIndexOp>(loc, 0);

  Value v = bd.create<arith::IndexCastOp>(loc, i32Ty, acc);
  bd.create<func::CallOp>(loc, scalarPrintFn, ValueRange{v});
}

enum class Strategy { Chain1D, CastMultiDim };

struct Candidate {
  func::CallOp callOp;
  Value arg;
  Strategy strategy;
  int64_t length;
  SmallVector<int64_t> shape;
  SmallVector<int64_t> strides;
  int64_t baseOffset;
};

static void rewriteOneFunc(func::FuncOp funcOp) {
  auto moduleOp = funcOp->getParentOfType<ModuleOp>();
  if (!moduleOp) return;

  SmallVector<Candidate> candidates;

  funcOp.walk([&](func::CallOp callOp) {
    if (!callOp.getCallee().starts_with("triton_print")) return;
    if (callOp.getNumOperands() != 1) return;
    Value arg = callOp.getOperand(0);
    auto argTy = dyn_cast<RankedTensorType>(arg.getType());
    if (!argTy) return;
    if (!argTy.hasStaticShape()) return;
    if (!hasOnlyPrintUsers(arg)) return;

    if (argTy.getRank() == 1) {
      if (auto len = verifyChain1D(arg)) {
        Candidate c;
        c.callOp = callOp;
        c.arg = arg;
        c.strategy = Strategy::Chain1D;
        c.length = *len;
        candidates.push_back(std::move(c));
        return;
      }
    }

    if (!isLinearChain(arg)) return;
    auto castOp = findMatchingCast(funcOp, argTy.getShape());
    if (!castOp) return;

    bool hasDynamic = false;
    for (int64_t s : castOp.getStaticStrides())
      if (ShapedType::isDynamic(s)) { hasDynamic = true; break; }
    if (!hasDynamic)
      for (int64_t o : castOp.getStaticOffsets())
        if (ShapedType::isDynamic(o)) { hasDynamic = true; break; }
    if (hasDynamic) {
      LLVM_DEBUG(llvm::dbgs() << "[" << funcOp.getSymName()
                              << "] skip: dynamic offset/stride in multi-dim cast\n");
      return;
    }

    Candidate c;
    c.callOp = callOp;
    c.arg = arg;
    c.strategy = Strategy::CastMultiDim;
    c.shape.assign(argTy.getShape().begin(), argTy.getShape().end());
    c.strides.assign(castOp.getStaticStrides().begin(),
                      castOp.getStaticStrides().end());
    c.baseOffset = castOp.getStaticOffsets().empty()
                    ? 0
                    : castOp.getStaticOffsets()[0];
    candidates.push_back(std::move(c));
  });
  if (candidates.empty()) return;

  llvm::DenseMap<Value, SmallVector<Candidate *>> byArg;
  for (Candidate &c : candidates) byArg[c.arg].push_back(&c);

  for (auto &entry : byArg) {
    auto &group = entry.second;
    auto i32Ty = IntegerType::get(funcOp.getContext(), 32);
    for (Candidate *cand : group) {
      auto oldFn = dyn_cast_or_null<func::FuncOp>(
          SymbolTable::lookupSymbolIn(moduleOp,
                                       cand->callOp.getCalleeAttr()));
      if (!oldFn) continue;
      auto newFnTy = FunctionType::get(funcOp.getContext(),
                                         {i32Ty}, {});
      oldFn.setFunctionType(newFnTy);

      OpBuilder b(cand->callOp);
      switch (cand->strategy) {
        case Strategy::Chain1D:
          emitChain1DForLoop(b, cand->callOp.getLoc(), oldFn,
                              cand->arg, cand->length);
          break;
        case Strategy::CastMultiDim:
          emitCastMultiDimForLoops(b, cand->callOp.getLoc(), oldFn,
                                    cand->shape, cand->strides,
                                    cand->baseOffset);
          break;
      }
      cand->callOp.erase();
    }
  }
}

}  // namespace

void rewriteDevicePrintOffsets(ModuleOp moduleOp) {
  moduleOp.walk([&](func::FuncOp funcOp) {
    if (funcOp.isPrivate()) return;
    rewriteOneFunc(funcOp);
  });
}

}  // namespace triton
}  // namespace mlir