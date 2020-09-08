#include "lumen/EIR/IR/EIROps.h"
#include "lumen/EIR/IR/EIRAttributes.h"
#include "lumen/EIR/IR/EIRTypes.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/SMLoc.h"

#include <iterator>
#include <vector>

using namespace lumen;
using namespace lumen::eir;

using ::llvm::ArrayRef;
using ::llvm::SmallVector;
using ::llvm::StringRef;
using ::llvm::dyn_cast_or_null;
using ::llvm::cast;
using ::llvm::isa;
using ::mlir::OpOperand;

namespace lumen {
namespace eir {

/// Matches a ConstantIntOp

/// The matcher that matches a constant numeric operation and binds the constant
/// value.
struct constant_apint_op_binder {
  APIntAttr::ValueType *bind_value;

  /// Creates a matcher instance that binds the value to bv if match succeeds.
  constant_apint_op_binder(APIntAttr::ValueType *bv) : bind_value(bv) {}

  bool match(Operation *op) {
    Attribute attr;
    if (!mlir::detail::constant_op_binder<Attribute>(&attr).match(op))
      return false;
    auto type = op->getResult(0).getType();

    if (auto opaque = type.dyn_cast_or_null<OpaqueTermType>()) {
      if (opaque.isFixnum())
        return mlir::detail::attr_value_binder<APIntAttr>(bind_value)
            .match(attr);
      if (opaque.isBox()) {
        auto box = type.cast<BoxType>();
        if (box.getBoxedType().isInteger())
          return mlir::detail::attr_value_binder<APIntAttr>(bind_value)
              .match(attr);
      }
    }

    return false;
  }
};

/// The matcher that matches a constant numeric operation and binds the constant
/// value.
struct constant_apfloat_op_binder {
  APFloatAttr::ValueType *bind_value;

  /// Creates a matcher instance that binds the value to bv if match succeeds.
  constant_apfloat_op_binder(APFloatAttr::ValueType *bv) : bind_value(bv) {}

  bool match(Operation *op) {
    Attribute attr;
    if (!mlir::detail::constant_op_binder<Attribute>(&attr).match(op))
      return false;
    auto type = op->getResult(0).getType();

    if (auto opaque = type.dyn_cast_or_null<OpaqueTermType>()) {
      if (opaque.isFloat())
        return mlir::detail::attr_value_binder<APFloatAttr>(bind_value)
            .match(attr);
    }

    return false;
  }
};

/// The matcher that matches a constant boolean operation and binds the constant
/// value.
struct constant_bool_op_binder {
  BoolAttr::ValueType *bind_value;

  /// Creates a matcher instance that binds the value to bv if match succeeds.
  constant_bool_op_binder(BoolAttr::ValueType *bv) : bind_value(bv) {}

  bool match(Operation *op) {
    Attribute attr;
    if (!mlir::detail::constant_op_binder<Attribute>(&attr).match(op))
      return false;
    auto type = op->getResult(0).getType();

    if (type.isa<BooleanType>()) {
      return mlir::detail::attr_value_binder<BoolAttr>(bind_value)
          .match(attr);
    }

    if (type.isa<AtomType>()) {
      if (auto atomAttr = attr.dyn_cast<AtomAttr>()) {
        auto id = atomAttr.getValue().getLimitedValue();
        if (id == 0 || id == 1) {
          *bind_value = id == 1;
          return true;
        }
        return false;
      }
    }

    if (type.isInteger(1)) {
      if (auto intAttr = attr.dyn_cast<IntegerAttr>()) {
        auto val = intAttr.getValue().getLimitedValue();
        if (val == 0 || val == 1) {
          *bind_value = val == 1;
          return true;
        }
        return false;
      }

      // We might have a boolean constant with i1 as type
      return mlir::detail::attr_value_binder<BoolAttr>(bind_value)
          .match(attr);
    }

    return false;
  }
};

struct constant_atom_op_binder {
  AtomAttr::ValueType *bind_value;

  /// Creates a matcher instance that binds the value to bv if match succeeds.
  constant_atom_op_binder(AtomAttr::ValueType *bv) : bind_value(bv) {}

  bool match(Operation *op) {
    Attribute attr;
    if (!mlir::detail::constant_op_binder<Attribute>(&attr).match(op))
      return false;
    auto type = op->getResult(0).getType();

    if (type.isa<AtomType>()) {
      return mlir::detail::attr_value_binder<AtomAttr>(bind_value)
          .match(attr);
    }

    if (type.isa<BooleanType>()) {
      if (auto boolAttr = attr.dyn_cast<BoolAttr>()) {
        *bind_value = APInt(64, (uint64_t)boolAttr.getValue(), /*signed=*/false);
        return true;
      }
      return false;
    }

    return false;
  }
};

inline constant_apint_op_binder m_ConstInt(APIntAttr::ValueType *bind_value) {
  return constant_apint_op_binder(bind_value);
}

inline constant_apfloat_op_binder m_ConstFloat(
    APFloatAttr::ValueType *bind_value) {
  return constant_apfloat_op_binder(bind_value);
}

inline constant_bool_op_binder m_ConstBool(
    BoolAttr::ValueType *bind_value) {
  return constant_bool_op_binder(bind_value);
}

inline constant_atom_op_binder m_ConstAtom(
    AtomAttr::ValueType *bind_value) {
  return constant_atom_op_binder(bind_value);
}


//===----------------------------------------------------------------------===//
// eir.func
//===----------------------------------------------------------------------===//

static ParseResult parseFuncOp(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType = [](Builder &builder, ArrayRef<Type> argTypes,
                          ArrayRef<Type> results, mlir::impl::VariadicFlag,
                          std::string &) {
    return builder.getFunctionType(argTypes, results);
  };
  return mlir::impl::parseFunctionLikeOp(
      parser, result, /*allowVariadic=*/false, buildFuncType);
}

static void print(OpAsmPrinter &p, FuncOp &op) {
  FunctionType fnType = op.getType();
  mlir::impl::printFunctionLikeOp(p, op, fnType.getInputs(),
                                  /*isVariadic=*/false, fnType.getResults());
}

FuncOp FuncOp::create(Location location, StringRef name, FunctionType type,
                      ArrayRef<NamedAttribute> attrs) {
  OperationState state(location, FuncOp::getOperationName());
  OpBuilder builder(location->getContext());
  FuncOp::build(builder, state, name, type, attrs);
  return cast<FuncOp>(Operation::create(state));
}

void FuncOp::build(OpBuilder &builder, OperationState &result, StringRef name,
                   FunctionType type, ArrayRef<NamedAttribute> attrs,
                   ArrayRef<MutableDictionaryAttr> argAttrs) {
  result.addRegion();
  result.addAttribute(SymbolTable::getSymbolAttrName(),
                      builder.getStringAttr(name));
  result.addAttribute("type", TypeAttr::get(type));
  result.attributes.append(attrs.begin(), attrs.end());
  if (argAttrs.empty()) {
    return;
  }

  if (argAttrs.empty()) return;

  unsigned numInputs = type.getNumInputs();
  assert(numInputs == argAttrs.size() &&
         "expected as many argument attribute lists as arguments");
  SmallString<8> argAttrName;
  for (unsigned i = 0, e = numInputs; i != e; ++i)
    if (auto argDict = argAttrs[i].getDictionary(builder.getContext()))
      result.addAttribute(getArgAttrName(i, argAttrName), argDict);
}

Block *FuncOp::addEntryBlock() {
  assert(empty() && "function already has an entry block");
  auto *entry = new Block();
  push_back(entry);
  entry->addArguments(getType().getInputs());
  return entry;
}

LogicalResult FuncOp::verifyType() {
  auto type = getTypeAttr().getValue();
  if (!type.isa<FunctionType>())
    return emitOpError("requires '" + getTypeAttrName() +
                       "' attribute of function type");
  return success();
}

Region *FuncOp::getCallableRegion() { return &body(); }

ArrayRef<Type> FuncOp::getCallableResults() {
  assert(!isExternal() && "invalid callable");
  return getType().getResults();
}

//===----------------------------------------------------------------------===//
// eir.call_indirect
//===----------------------------------------------------------------------===//
namespace {
/// Fold indirect calls that have a constant function as the callee operand.
struct SimplifyIndirectCallWithKnownCallee
    : public OpRewritePattern<CallIndirectOp> {
  using OpRewritePattern<CallIndirectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CallIndirectOp indirectCall,
                                PatternRewriter &rewriter) const override {
    // Check that the callee is a constant callee.
    FlatSymbolRefAttr calledFn;
    if (!matchPattern(indirectCall.getCallee(), ::mlir::m_Constant(&calledFn)))
      return failure();

    // Replace with a direct call.
    rewriter.replaceOpWithNewOp<CallOp>(indirectCall, calledFn,
                                        indirectCall.getResultTypes(),
                                        indirectCall.getArgOperands());
    return success();
  }
};
}  // end anonymous namespace.

void CallIndirectOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<SimplifyIndirectCallWithKnownCallee>(context);
}

//===----------------------------------------------------------------------===//
// eir.br
//===----------------------------------------------------------------------===//

/// Given a successor, try to collapse it to a new destination if it only
/// contains a passthrough unconditional branch. If the successor is
/// collapsable, `successor` and `successorOperands` are updated to reference
/// the new destination and values. `argStorage` is an optional storage to use
/// if operands to the collapsed successor need to be remapped.
static LogicalResult collapseBranch(Block *&successor,
                                    ValueRange &successorOperands,
                                    SmallVectorImpl<Value> &argStorage) {
  // Check that the successor only contains a unconditional branch.
  if (std::next(successor->begin()) != successor->end()) return failure();
  // Check that the terminator is an unconditional branch.
  BranchOp successorBranch = dyn_cast<BranchOp>(successor->getTerminator());
  if (!successorBranch) return failure();
  // Check that the arguments are only used within the terminator.
  for (BlockArgument arg : successor->getArguments()) {
    for (Operation *user : arg.getUsers())
      if (user != successorBranch) return failure();
  }
  // Don't try to collapse branches to infinite loops.
  Block *successorDest = successorBranch.getDest();
  if (successorDest == successor) return failure();

  // Update the operands to the successor. If the branch parent has no
  // arguments, we can use the branch operands directly.
  OperandRange operands = successorBranch.getOperands();
  if (successor->args_empty()) {
    successor = successorDest;
    successorOperands = operands;
    return success();
  }

  // Otherwise, we need to remap any argument operands.
  for (Value operand : operands) {
    BlockArgument argOperand = operand.dyn_cast<BlockArgument>();
    if (argOperand && argOperand.getOwner() == successor)
      argStorage.push_back(successorOperands[argOperand.getArgNumber()]);
    else
      argStorage.push_back(operand);
  }
  successor = successorDest;
  successorOperands = argStorage;
  return success();
}

namespace {
/// Simplify a branch to a block that has a single predecessor. This effectively
/// merges the two blocks.
struct SimplifyBrToBlockWithSinglePred : public OpRewritePattern<BranchOp> {
  using OpRewritePattern<BranchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BranchOp op,
                                PatternRewriter &rewriter) const override {
    // Check that the successor block has a single predecessor.
    Block *succ = op.getDest();
    Block *opParent = op.getOperation()->getBlock();
    if (succ == opParent || !llvm::hasSingleElement(succ->getPredecessors()))
      return failure();

    // Merge the successor into the current block and erase the branch.
    rewriter.mergeBlocks(succ, opParent, op.getOperands());
    rewriter.eraseOp(op);
    return success();
  }
};

///   br ^bb1
/// ^bb1
///   br ^bbN(...)
///
///  -> br ^bbN(...)
///
struct SimplifyPassThroughBr : public OpRewritePattern<BranchOp> {
  using OpRewritePattern<BranchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BranchOp op,
                                PatternRewriter &rewriter) const override {
    Block *dest = op.getDest();
    ValueRange destOperands = op.getOperands();
    SmallVector<Value, 4> destOperandStorage;

    // Try to collapse the successor if it points somewhere other than this
    // block.
    if (dest == op.getOperation()->getBlock() ||
        failed(collapseBranch(dest, destOperands, destOperandStorage)))
      return failure();

    // Create a new branch with the collapsed successor.
    rewriter.replaceOpWithNewOp<BranchOp>(op, dest, destOperands);
    return success();
  }
};
}  // namespace

void BranchOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                           MLIRContext *context) {
  results.insert<SimplifyBrToBlockWithSinglePred,
                 SimplifyPassThroughBr>(context);
}

Optional<MutableOperandRange> BranchOp::getMutableSuccessorOperands(
    unsigned index) {
  assert(index == 0 && "invalid successor index");
  return destOperandsMutable();
}

Block *BranchOp::getSuccessorForOperands(ArrayRef<Attribute>) { return dest(); }

//===----------------------------------------------------------------------===//
// eir.cond_br
//===----------------------------------------------------------------------===//

namespace {
/// eir.cond_br true, ^bb1, ^bb2 -> br ^bb1
/// eir.cond_br false, ^bb1, ^bb2 -> br ^bb2
///
struct SimplifyConstCondBranchPred : public OpRewritePattern<CondBranchOp> {
  using OpRewritePattern<CondBranchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CondBranchOp condbr,
                                PatternRewriter &rewriter) const override {
    if (matchPattern(condbr.getCondition(), mlir::m_NonZero())) {
      // True branch taken.
      rewriter.replaceOpWithNewOp<BranchOp>(condbr, condbr.getTrueDest(),
                                            condbr.getTrueOperands());
      return success();
    } else if (matchPattern(condbr.getCondition(), mlir::m_Zero())) {
      // False branch taken.
      rewriter.replaceOpWithNewOp<BranchOp>(condbr, condbr.getFalseDest(),
                                            condbr.getFalseOperands());
      return success();
    }
    return failure();
  }
};

///   cond_br %cond, ^bb1, ^bb2
/// ^bb1
///   br ^bbN(...)
/// ^bb2
///   br ^bbK(...)
///
///  -> cond_br %cond, ^bbN(...), ^bbK(...)
///
struct SimplifyPassThroughCondBranch : public OpRewritePattern<CondBranchOp> {
  using OpRewritePattern<CondBranchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CondBranchOp condbr,
                                PatternRewriter &rewriter) const override {
    Block *trueDest = condbr.trueDest(), *falseDest = condbr.falseDest();
    ValueRange trueDestOperands = condbr.getTrueOperands();
    ValueRange falseDestOperands = condbr.getFalseOperands();
    SmallVector<Value, 4> trueDestOperandStorage, falseDestOperandStorage;

    // Try to collapse one of the current successors.
    LogicalResult collapsedTrue =
        collapseBranch(trueDest, trueDestOperands, trueDestOperandStorage);
    LogicalResult collapsedFalse =
        collapseBranch(falseDest, falseDestOperands, falseDestOperandStorage);
    if (failed(collapsedTrue) && failed(collapsedFalse)) return failure();

    // Create a new branch with the collapsed successors.
    rewriter.replaceOpWithNewOp<CondBranchOp>(condbr, condbr.getCondition(),
                                              trueDest, trueDestOperands,
                                              falseDest, falseDestOperands);
    return success();
  }
};

/// cond_br %cond, ^bb1(A, ..., N), ^bb1(A, ..., N)
///  -> br ^bb1(A, ..., N)
///
/// cond_br %cond, ^bb1(A), ^bb1(B)
///  -> %select = select %cond, A, B
///     br ^bb1(%select)
///
struct SimplifyCondBranchIdenticalSuccessors
    : public OpRewritePattern<CondBranchOp> {
  using OpRewritePattern<CondBranchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CondBranchOp condbr,
                                PatternRewriter &rewriter) const override {
    // Check that the true and false destinations are the same and have the same
    // operands.
    Block *trueDest = condbr.trueDest();
    if (trueDest != condbr.falseDest()) return failure();

    // If all of the operands match, no selects need to be generated.
    OperandRange trueOperands = condbr.getTrueOperands();
    OperandRange falseOperands = condbr.getFalseOperands();
    if (trueOperands == falseOperands) {
      rewriter.replaceOpWithNewOp<BranchOp>(condbr, trueDest, trueOperands);
      return success();
    }

    // Otherwise, if the current block is the only predecessor insert selects
    // for any mismatched branch operands.
    if (trueDest->getUniquePredecessor() != condbr.getOperation()->getBlock())
      return failure();

    // Generate a select for any operands that differ between the two.
    SmallVector<Value, 8> mergedOperands;
    mergedOperands.reserve(trueOperands.size());
    Value condition = condbr.getCondition();
    for (auto it : llvm::zip(trueOperands, falseOperands)) {
      if (std::get<0>(it) == std::get<1>(it))
        mergedOperands.push_back(std::get<0>(it));
      else
        mergedOperands.push_back(rewriter.create<mlir::SelectOp>(
            condbr.getLoc(), condition, std::get<0>(it), std::get<1>(it)));
    }

    rewriter.replaceOpWithNewOp<BranchOp>(condbr, trueDest, mergedOperands);
    return success();
  }
};

struct SimplifyCondBranchToUnreachable : public OpRewritePattern<CondBranchOp> {
  using OpRewritePattern<CondBranchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CondBranchOp condbr,
                                PatternRewriter &rewriter) const override {
    Block *trueDest = condbr.trueDest();
    Block *falseDest = condbr.falseDest();

    // Determine if either branch goes to a block with a single operation
    bool trueIsCandidate = std::next(trueDest->begin()) == trueDest->end();
    bool falseIsCandidate = std::next(falseDest->begin()) == falseDest->end();
    // If neither are candidates for this transformation, we're done
    if (!trueIsCandidate && !falseIsCandidate) return failure();

    // Determine if either branch contains an unreachable
    // Check that the terminator is an unconditional branch.
    Operation *trueOp = trueDest->getTerminator();
    assert(trueOp && "expected terminator");
    Operation *falseOp = falseDest->getTerminator();
    assert(falseOp && "expected terminator");
    UnreachableOp trueUnreachable = dyn_cast<UnreachableOp>(trueOp);
    UnreachableOp falseUnreachable = dyn_cast<UnreachableOp>(falseOp);
    // If neither terminator are unreachables, there is nothing to do
    if (!trueUnreachable && !falseUnreachable) return failure();

    // If both blocks are unreachable, then we can replace this
    // branch operation with an unreachable as well
    if (trueUnreachable && falseUnreachable) {
      rewriter.replaceOpWithNewOp<UnreachableOp>(condbr);
      return success();
    }

    Block *unreachable;
    Block *reachable;
    Operation *reachableOp;
    OperandRange reachableOperands =
        trueUnreachable ? condbr.getFalseOperands() : condbr.getTrueOperands();
    Block *opParent = condbr.getOperation()->getBlock();

    if (trueUnreachable) {
      unreachable = trueDest;
      reachable = falseDest;
      reachableOp = falseOp;
    } else {
      unreachable = falseDest;
      reachable = trueDest;
      reachableOp = trueOp;
    }

    // If the reachable block is a return, we can collapse this operation
    // to a return rather than a branch
    if (auto ret = dyn_cast<ReturnOp>(reachableOp)) {
      if (reachable->getUniquePredecessor() == opParent) {
        // If the reachable block is only reachable from here, merge the blocks
        rewriter.eraseOp(condbr);
        rewriter.mergeBlocks(reachable, opParent, reachableOperands);
        return success();
      } else {
        // If the reachable block has multiple predecessors, but the
        // return only references operands reachable from this block,
        // then replace the condbr with a copy of the return
        SmallVector<Value, 4> destOperandStorage;
        auto retOperands = ret.operands();
        for (Value operand : retOperands) {
          BlockArgument argOperand = operand.dyn_cast<BlockArgument>();
          if (argOperand && argOperand.getOwner() == reachable) {
            // The operand is a block argument in the reachable block,
            // remap it to the successor operand we have in this block
            destOperandStorage.push_back(
                reachableOperands[argOperand.getArgNumber()]);
          } else if (operand.getParentBlock() == reachable) {
            // The operand is constructed in the reachable block,
            // so we can't proceed without cloning the block into
            // this block
            return failure();
          } else {
            // The operand is from parent scope, we can safely reference it
            destOperandStorage.push_back(operand);
          }
        }
        rewriter.replaceOpWithNewOp<ReturnOp>(condbr, destOperandStorage);
        return success();
      }
    }

    // The reachable block doesn't contain a return, so instead replace
    // the condbr with an unconditional branch
    rewriter.replaceOpWithNewOp<BranchOp>(condbr, reachable, reachableOperands);
    return success();
  }
};
}  // end anonymous namespace.

void CondBranchOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<SimplifyConstCondBranchPred, SimplifyPassThroughCondBranch,
                 SimplifyCondBranchIdenticalSuccessors,
                 SimplifyCondBranchToUnreachable>(context);
}

Optional<MutableOperandRange> CondBranchOp::getMutableSuccessorOperands(
    unsigned index) {
  assert(index < getNumSuccessors() && "invalid successor index");
  return index == trueIndex ? trueDestOperandsMutable()
                            : falseDestOperandsMutable();
}

Block *CondBranchOp::getSuccessorForOperands(ArrayRef<Attribute> operands) {
  if (IntegerAttr condAttr = operands.front().dyn_cast_or_null<IntegerAttr>())
    return condAttr.getValue().isOneValue() ? trueDest() : falseDest();
  return nullptr;
}

//===----------------------------------------------------------------------===//
// eir.return
//===----------------------------------------------------------------------===//

static LogicalResult verify(ReturnOp op) {
  auto function = cast<FuncOp>(op.getParentOp());

  // The operand number and types must match the function signature.
  const auto &results = function.getType().getResults();
  if (op.getNumOperands() != results.size())
    return op.emitOpError("has ")
           << op.getNumOperands()
           << " operands, but enclosing function returns " << results.size();

  for (unsigned i = 0, e = results.size(); i != e; ++i)
    if (op.getOperand(i).getType() != results[i])
      return op.emitError()
             << "type of return operand " << i << " ("
             << op.getOperand(i).getType()
             << ") doesn't match function result type (" << results[i] << ")";

  return success();
}

//===----------------------------------------------------------------------===//
// eir.is_type
//===----------------------------------------------------------------------===//

static LogicalResult verify(IsTypeOp op) {
  auto typeAttr = op.getAttrOfType<TypeAttr>("type");
  if (!typeAttr) return op.emitOpError("requires type attribute named 'type'");

  return success();
}

//===----------------------------------------------------------------------===//
// eir.is_tuple
//===----------------------------------------------------------------------===//

static LogicalResult verify(IsTupleOp op) {
  auto numOperands = op.getNumOperands();
  if (numOperands < 1 || numOperands > 2)
    return op.emitOpError("invalid number of operands, expected at least one and no more than 2");

  return success();
}

//===----------------------------------------------------------------------===//
// eir.is_function
//===----------------------------------------------------------------------===//

static LogicalResult verify(IsFunctionOp op) {
  auto numOperands = op.getNumOperands();
  if (numOperands < 1 || numOperands > 2)
    return op.emitOpError("invalid number of operands, expected at least one and no more than 2");

  return success();
}

//===----------------------------------------------------------------------===//
// eir.cast
//===----------------------------------------------------------------------===//

OpFoldResult CastOp::fold(ArrayRef<Attribute> operands) {
  Type srcTy = getSourceType();
  Type targetTy = getTargetType();
  // Identity cast
  if (srcTy == targetTy)
    return input();
  return nullptr;
}

static bool areCastCompatible(OpaqueTermType srcType, OpaqueTermType destType) {
  if (destType.isOpaque()) {
    // Casting an immediate to an opaque term is always allowed
    if (srcType.isImmediate() || srcType.isPid())
      return true;
    // Casting a boxed value to an opaque term is always allowed
    if (srcType.isBox())
      return true;
    // This is redundant, but technically allowed and will be eliminated via canonicalization
    if (srcType.isOpaque())
      return true;
  }
  // Casting an opaque term to any term type is always allowed
  if (srcType.isOpaque()) return true;
  // Box-to-box casts are always allowed
  if (srcType.isBox() & destType.isBox()) return true;
  // Only header types can be boxed
  if (destType.isBox() && !srcType.isBoxable()) return false;
  // A cast must be to an immediate-sized type
  if (!destType.isImmediate()) return false;
  // Only support casts between compatible types
  if (srcType.isNumber() && !destType.isNumber()) return false;
  if (srcType.isInteger() && !destType.isInteger()) return false;
  if (srcType.isFloat() && !destType.isFloat()) return false;
  if (srcType.isList() && !destType.isList()) return false;
  if (srcType.isBinary() && !destType.isBinary()) return false;
  // All other casts are supported
  return true;
}

static LogicalResult verify(CastOp op) {
  auto opType = op.getSourceType();
  auto resType = op.getTargetType();
  if (auto opTermType = opType.dyn_cast_or_null<OpaqueTermType>()) {
    if (auto resTermType = resType.dyn_cast_or_null<OpaqueTermType>()) {
      if (!areCastCompatible(opTermType, resTermType)) {
        return op.emitError("operand type ")
               << opType << " and result type " << resType
               << " are not cast compatible";
      }
      return success();
    }

    if (resType.isa<PtrType>() || resType.isa<RefType>())
      return success();

    if (resType.isIntOrFloat())
      return success();

    return op.emitError("invalid cast type, target type is unsupported");
  }

  if (opType.isa<PtrType>() || resType.isa<RefType>())
    return success();

  if (opType.isIntOrFloat())
    return success();

  return op.emitError("invalid cast type, sourcce type is unsupported");
}

//===----------------------------------------------------------------------===//
// eir.match
//===----------------------------------------------------------------------===//

LogicalResult lowerPatternMatch(OpBuilder &builder, Location loc, Value selector,
                                ArrayRef<MatchBranch> branches) {
  auto numBranches = branches.size();
  assert(numBranches > 0 && "expected at least one branch in a match");

  auto *context = builder.getContext();
  auto *currentBlock = builder.getInsertionBlock();
  auto *region = currentBlock->getParent();
  auto selectorType = selector.getType();

  // Save our insertion point in the current block
  auto startIp = builder.saveInsertionPoint();

  // Create blocks for all match arms
  bool needsFallbackBranch = true;
  SmallVector<Block *, 3> blocks;
  // The first match arm is evaluated in the current block, so we
  // handle it specially
  blocks.reserve(numBranches);
  blocks.push_back(currentBlock);
  if (branches[0].isCatchAll()) {
    needsFallbackBranch = false;
  }
  // All other match arms need blocks for the evaluation of their patterns
  for (auto &branch : branches.take_back(numBranches - 1)) {
    if (branch.isCatchAll()) {
      needsFallbackBranch = false;
    }
    Block *block = builder.createBlock(region);
    block->addArgument(selectorType);
    blocks.push_back(block);
  }

  // Create fallback block, if needed, after all other match blocks, so
  // that after all other conditions have been tried, we branch to an
  // unreachable to force a trap
  Block *failed = nullptr;
  if (needsFallbackBranch) {
    failed = builder.createBlock(region);
    failed->addArgument(selectorType);
    builder.create<eir::UnreachableOp>(loc);
  }

  // Restore our original insertion point
  builder.restoreInsertionPoint(startIp);

  // Save the current insertion point, which we'll restore when lowering is
  // complete
  auto finalIp = builder.saveInsertionPoint();

  // Common types used below
  auto termType = builder.getType<TermType>();
  auto i1Ty = builder.getI1Type();

  // Used whenever we need a set of empty args below
  ArrayRef<Value> emptyArgs{};

  // For each branch, populate its block with the predicate and
  // appropriate conditional branching instruction to either jump
  // to the success block, or to the next branches' block (or in
  // the case of the last branch, the 'failed' block)
  for (unsigned i = 0; i < numBranches; i++) {
    auto &b = branches[i];
    Location branchLoc = b.getLoc();
    bool isLast = i == numBranches - 1;
    Block *block = blocks[i];

    // Set our insertion point to the end of the pattern block
    builder.setInsertionPointToEnd(block);

    // Get the selector value in this block,
    // in the case of the first block, its our original
    // input selector value
    Value selectorArg;
    if (i == 0) {
      selectorArg = selector;
    } else {
      selectorArg = block->getArgument(0);
    }
    ArrayRef<Value> withSelectorArgs{selectorArg};

    // Store the next pattern to try if this one fails
    // If this is the last pattern, we validate that the
    // branch either unconditionally succeeds, or branches to
    // an unreachable op
    Block *nextPatternBlock = nullptr;
    if (!isLast) {
      nextPatternBlock = blocks[i + 1];
    } else if (needsFallbackBranch) {
      nextPatternBlock = failed;
    }

    auto dest = b.getDest();
    auto baseDestArgs = b.getDestArgs();
    auto numBaseDestArgs = baseDestArgs.size();

    // Ensure the destination block argument types are propagated
    for (unsigned i = 0; i < baseDestArgs.size(); i++) {
      BlockArgument arg = dest->getArgument(i);
      auto destArg = baseDestArgs[i];
      auto destArgTy = destArg.getType();
      if (arg.getType() != destArgTy)
        arg.setType(destArgTy);
    }

    switch (b.getPatternType()) {
      case MatchPatternType::Any: {
        // This unconditionally branches to its destination
        builder.create<BranchOp>(branchLoc, dest, baseDestArgs);
        break;
      }

      case MatchPatternType::Cons: {
        assert(nextPatternBlock != nullptr &&
               "last match block must end in unconditional branch");
        auto cip = builder.saveInsertionPoint();
        // 1. Split block, and conditionally branch to split if is_cons,
        // otherwise the next pattern
        Block *split =
            builder.createBlock(region, Region::iterator(nextPatternBlock));
        builder.restoreInsertionPoint(cip);
        auto consType = builder.getType<ConsType>();
        auto boxedConsType = builder.getType<BoxType>(consType);
        auto isConsOp =
            builder.create<IsTypeOp>(branchLoc, selectorArg, boxedConsType);
        auto isConsCond = isConsOp.getResult();
        auto ifOp = builder.create<CondBranchOp>(branchLoc, isConsCond, split,
                                                 emptyArgs, nextPatternBlock,
                                                 withSelectorArgs);
        // 2. In the split, extract head and tail values of the cons cell
        builder.setInsertionPointToEnd(split);
        auto ptrConsType = builder.getType<PtrType>(consType);
        auto castOp =
            builder.create<CastOp>(branchLoc, selectorArg, ptrConsType);
        auto consPtr = castOp.getResult();
        auto getHeadOp =
            builder.create<GetElementPtrOp>(branchLoc, consPtr, 0);
        auto getTailOp =
            builder.create<GetElementPtrOp>(branchLoc, consPtr, 1);
        auto headPointer = getHeadOp.getResult();
        auto tailPointer = getTailOp.getResult();
        auto headLoadOp = builder.create<LoadOp>(branchLoc, headPointer);
        auto headLoadResult = headLoadOp.getResult();
        auto tailLoadOp = builder.create<LoadOp>(branchLoc, tailPointer);
        auto tailLoadResult = tailLoadOp.getResult();
        // 3. Unconditionally branch to the destination, with head/tail as
        // additional destArgs
        unsigned i = numBaseDestArgs > 0 ? numBaseDestArgs - 1 : 0;
        dest->getArgument(i++).setType(headLoadResult.getType());
        dest->getArgument(i).setType(tailLoadResult.getType());
        SmallVector<Value, 2> destArgs(
            {baseDestArgs.begin(), baseDestArgs.end()});
        destArgs.push_back(headLoadResult);
        destArgs.push_back(tailLoadResult);
        builder.create<BranchOp>(branchLoc, dest, destArgs);
        break;
      }

      case MatchPatternType::Tuple: {
        assert(nextPatternBlock != nullptr &&
               "last match block must end in unconditional branch");
        auto *pattern = b.getPatternTypeOrNull<TuplePattern>();
        // 1. Split block, and conditionally branch to split if is_tuple w/arity
        // N, otherwise the next pattern
        auto cip = builder.saveInsertionPoint();
        Block *split =
            builder.createBlock(region, Region::iterator(nextPatternBlock));
        builder.restoreInsertionPoint(cip);
        auto arity = pattern->getArity();
        auto tupleType = builder.getType<eir::TupleType>(arity);
        auto boxedTupleType = builder.getType<BoxType>(tupleType);
        auto isTupleOp = 
            builder.create<IsTypeOp>(branchLoc, selectorArg, boxedTupleType);
        auto isTupleCond = isTupleOp.getResult();
        auto ifOp = builder.create<CondBranchOp>(branchLoc, isTupleCond, split,
                                                 emptyArgs, nextPatternBlock,
                                                 withSelectorArgs);
        // 2. In the split, extract the tuple elements as values
        builder.setInsertionPointToEnd(split);
        auto ptrTupleType = builder.getType<PtrType>(tupleType);
        auto castOp =
            builder.create<CastOp>(branchLoc, selectorArg, ptrTupleType);
        auto tuplePtr = castOp.getResult();
        unsigned ai = numBaseDestArgs > 0 ? numBaseDestArgs - 1 : 0;
        SmallVector<Value, 2> destArgs(
            {baseDestArgs.begin(), baseDestArgs.end()});
        destArgs.reserve(arity);
        for (int64_t i = 0; i < arity; i++) {
          auto getElemOp =
              builder.create<GetElementPtrOp>(branchLoc, tuplePtr, i + 1);
          auto elemPtr = getElemOp.getResult();
          auto elemLoadOp = builder.create<LoadOp>(branchLoc, elemPtr);
          auto elemLoadResult = elemLoadOp.getResult();
          dest->getArgument(ai++).setType(elemLoadResult.getType());
          destArgs.push_back(elemLoadResult);
        }
        // 3. Unconditionally branch to the destination, with the tuple elements
        // as additional destArgs
        builder.create<BranchOp>(branchLoc, dest, destArgs);
        break;
      }

      case MatchPatternType::MapItem: {
        assert(nextPatternBlock != nullptr &&
               "last match block must end in unconditional branch");
        // 1. Split block twice, and conditionally branch to the first split if
        // is_map, otherwise the next pattern
        auto cip = builder.saveInsertionPoint();
        Block *split2 =
            builder.createBlock(region, Region::iterator(nextPatternBlock));
        Block *split = builder.createBlock(region, Region::iterator(split2));
        builder.restoreInsertionPoint(cip);
        auto *pattern = b.getPatternTypeOrNull<MapPattern>();
        auto key = pattern->getKey();
        auto mapType = BoxType::get(builder.getType<MapType>());
        auto isMapOp =
            builder.create<IsTypeOp>(branchLoc, selectorArg, mapType);
        auto isMapCond = isMapOp.getResult();
        auto ifOp = builder.create<CondBranchOp>(
            branchLoc, isMapCond, split, emptyArgs, nextPatternBlock,
            withSelectorArgs);
        // 2. In the split, call runtime function `is_map_key` to confirm
        // existence of the key in the map,
        //    then conditionally branch to the second split if successful,
        //    otherwise the next pattern
        builder.setInsertionPointToEnd(split);
        auto hasKeyOp = builder.create<MapIsKeyOp>(branchLoc, selectorArg, key);
        auto hasKeyCond = hasKeyOp.getResult();
        builder.create<CondBranchOp>(branchLoc, hasKeyCond, split2,
                                     emptyArgs, nextPatternBlock,
                                     withSelectorArgs);
        // 3. In the second split, call runtime function `map_get` to obtain the
        // value for the key
        builder.setInsertionPointToEnd(split2);
        auto mapGetOp =
            builder.create<MapGetKeyOp>(branchLoc, selectorArg, key);
        auto valueTerm = mapGetOp.getResult();
        unsigned i = numBaseDestArgs > 0 ? numBaseDestArgs - 1 : 0;
        dest->getArgument(i).setType(valueTerm.getType());
        // 4. Unconditionally branch to the destination, with the key's value as
        // an additional destArg
        SmallVector<Value, 2> destArgs(baseDestArgs.begin(),
                                       baseDestArgs.end());
        destArgs.push_back(valueTerm);
        builder.create<BranchOp>(branchLoc, dest, destArgs);
        break;
      }

      case MatchPatternType::IsType: {
        assert(nextPatternBlock != nullptr &&
               "last match block must end in unconditional branch");
        // 1. Conditionally branch to destination if is_<type>, otherwise the
        // next pattern
        auto *pattern = b.getPatternTypeOrNull<IsTypePattern>();
        auto expectedType = pattern->getExpectedType();
        auto isTypeOp =
            builder.create<IsTypeOp>(branchLoc, selectorArg, expectedType);
        auto isTypeCond = isTypeOp.getResult();
        builder.create<CondBranchOp>(branchLoc, isTypeCond, dest, baseDestArgs,
                                     nextPatternBlock, withSelectorArgs);
        break;
      }

      case MatchPatternType::Value: {
        assert(nextPatternBlock != nullptr &&
               "last match block must end in unconditional branch");
        // 1. Conditionally branch to dest if the value matches the selector,
        //    passing the value as an additional destArg
        auto *pattern = b.getPatternTypeOrNull<ValuePattern>();
        auto expected = pattern->getValue();
        auto isEq = builder.create<CmpEqOp>(branchLoc, selectorArg, expected,
                                            /*strict=*/true);
        auto isEqCond = isEq.getResult();
        builder.create<CondBranchOp>(branchLoc, isEqCond, dest, baseDestArgs,
                                     nextPatternBlock, withSelectorArgs);
        break;
      }

      case MatchPatternType::Binary: {
        // 1. Split block, and conditionally branch to split if is_bitstring (or
        // is_binary), otherwise the next pattern
        // 2. In the split, conditionally branch to destination if construction
        // of the head value succeeds,
        //    otherwise the next pattern
        // NOTE: The exact semantics depend on the binary specification type,
        // and what is optimal in terms of checks. The success of the overall
        // branch results in two additional destArgs being passed to the
        // destination block, the decoded entry (head), and the rest of the
        // binary (tail)
        auto *pattern = b.getPatternTypeOrNull<BinaryPattern>();
        auto spec = pattern->getSpec();
        auto size = pattern->getSize();
        Operation *op;
        switch (spec.tag) {
          case BinarySpecifierType::Integer: {
            auto payload = spec.payload.i;
            bool isSigned = payload.isSigned;
            auto endianness = payload.endianness;
            auto unit = payload.unit;
            op = builder.create<BinaryMatchIntegerOp>(
                branchLoc, selectorArg, isSigned, endianness, unit, size);
            break;
          }
          case BinarySpecifierType::Utf8: {
            op =
                builder.create<BinaryMatchUtf8Op>(branchLoc, selectorArg, size);
            break;
          }
          case BinarySpecifierType::Utf16: {
            auto endianness = spec.payload.es.endianness;
            op = builder.create<BinaryMatchUtf16Op>(branchLoc, selectorArg,
                                                    endianness, size);
            break;
          }
          case BinarySpecifierType::Utf32: {
            auto endianness = spec.payload.es.endianness;
            op = builder.create<BinaryMatchUtf32Op>(branchLoc, selectorArg,
                                                    endianness, size);
            break;
          }
          case BinarySpecifierType::Float: {
            auto payload = spec.payload.f;
            op = builder.create<BinaryMatchFloatOp>(
                branchLoc, selectorArg, payload.endianness, payload.unit, size);
            break;
          }
          case BinarySpecifierType::Bytes:
          case BinarySpecifierType::Bits: {
            auto payload = spec.payload.us;
            op = builder.create<BinaryMatchRawOp>(branchLoc, selectorArg,
                                                  payload.unit, size);
            break;
          }
          default: {
            auto diagEngine = &builder.getContext()->getDiagEngine();
            diagEngine->emit(branchLoc, DiagnosticSeverity::Error)
              << "unknown binary specifier type tag '" << ((unsigned)spec.tag) << "'";
            return failure();
          }
        }
        Value matched = op->getResult(0);
        Value rest = op->getResult(1);
        Value success = op->getResult(2);
        unsigned i = numBaseDestArgs > 0 ? numBaseDestArgs - 1 : 0;
        dest->getArgument(i++).setType(matched.getType());
        dest->getArgument(i).setType(rest.getType());
        SmallVector<Value, 2> destArgs(baseDestArgs.begin(),
                                       baseDestArgs.end());
        destArgs.push_back(matched);
        destArgs.push_back(rest);
        builder.create<CondBranchOp>(branchLoc, success, dest, destArgs,
                                     nextPatternBlock, withSelectorArgs);
        break;
      }

      default: {
        auto diagEngine = &builder.getContext()->getDiagEngine();
        diagEngine->emit(branchLoc, DiagnosticSeverity::Error) 
          << "unknown match pattern type '" << ((unsigned)b.getPatternType()) << "'";
        return failure();
      }
    }
  }
  builder.restoreInsertionPoint(finalIp);
  return success();
}

//===----------------------------------------------------------------------===//
// cmp.eq
//===----------------------------------------------------------------------===//

// APInt comparisons require equivalent bit width
static bool areAPIntsEqual(APInt &lhs, APInt &rhs) {
  auto lhsBits = lhs.getBitWidth();
  auto rhsBits = rhs.getBitWidth();

  if (lhsBits == rhsBits)
    return lhs == rhs;

  if (lhsBits < rhsBits) {
    APInt temp = lhs.sext(rhsBits);
    return temp == rhs;
  }

  APInt temp = rhs.sext(lhsBits);
  return lhs == temp;
}

static Optional<bool> foldEqualityComparison(Operation *op, ArrayRef<Attribute> operands) {
  assert(operands.size() == 2 && "binary op takes two operands");
    
  Attribute lhs = operands[0];
  Attribute rhs = operands[1];

  if (!lhs || !rhs)
    return llvm::None;

  // Atom-likes
  if (auto lhsAtom = lhs.dyn_cast_or_null<AtomAttr>()) {
    if (auto rhsAtom = rhs.dyn_cast_or_null<AtomAttr>()) {
      return areAPIntsEqual(lhsAtom.getValue(), rhsAtom.getValue());
    }
    if (auto rhsBool = rhs.dyn_cast_or_null<BoolAttr>()) {
      bool areEqual = lhsAtom.getValue().getLimitedValue() == (unsigned)(rhsBool.getValue());
      return areEqual;
    }
    assert(false && "unknown operand combo");
  }

  // Boolean-likes
  if (auto lhsBool = lhs.dyn_cast_or_null<BoolAttr>()) {
    if (auto rhsAtom = rhs.dyn_cast_or_null<AtomAttr>()) {
      bool areEqual = (unsigned)(lhsBool.getValue()) == rhsAtom.getValue().getLimitedValue();
      return areEqual;
    }
    if (auto rhsBool = rhs.dyn_cast_or_null<BoolAttr>()) {
      bool areEqual = lhsBool.getValue() == rhsBool.getValue();
      return areEqual;
    }
    assert(false && "unknown operand combo");
  }

  // Integers
  if (auto lhsInt = lhs.dyn_cast_or_null<APIntAttr>()) {
    if (auto rhsInt = rhs.dyn_cast_or_null<APIntAttr>()) {
      return areAPIntsEqual(lhsInt.getValue(), rhsInt.getValue());
    }
    if (auto rhsInt = rhs.dyn_cast_or_null<IntegerAttr>()) {
      auto rVal = rhsInt.getValue();
      return areAPIntsEqual(lhsInt.getValue(), rVal);
    }
  }

  // Floats
  if (auto lhsFloat = lhs.dyn_cast_or_null<APFloatAttr>()) {
    if (auto rhsFloat = rhs.dyn_cast_or_null<APFloatAttr>()) {
      bool areEqual = lhsFloat.getValue() == rhsFloat.getValue();
      return areEqual;
    }
  }

  return llvm::None;
}


OpFoldResult CmpEqOp::fold(ArrayRef<Attribute> operands) {
  auto maybeConstant = foldEqualityComparison(getOperation(), operands);
  if (maybeConstant.hasValue()) {
    return BoolAttr::get(maybeConstant.getValue(), getContext());
  }

  return nullptr;
}

namespace {
/// Fold negations of constants into negated constants
struct CanonicalizeEqualityComparison : public OpRewritePattern<CmpEqOp> {
  using OpRewritePattern<CmpEqOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CmpEqOp op,
                                PatternRewriter &rewriter) const override {
    auto lhs = op.lhs();
    auto rhs = op.rhs();
    auto i1Ty = rewriter.getI1Type();

    // Check if we can fold this comparison into a constant
    bool lhsVal;
    bool rhsVal;
    auto lhsBoolPattern = m_ConstBool(&lhsVal);
    auto rhsBoolPattern = m_ConstBool(&rhsVal);
    if (matchPattern(lhs, lhsBoolPattern)) {
      // Left-side is a constant boolean value, check right side
      if (matchPattern(rhs, rhsBoolPattern)) {
        // Both operands are boolean, constify
        rewriter.replaceOpWithNewOp<ConstantBoolOp>(op, i1Ty, lhsVal == rhsVal);
        return success();
      }

      // Right-hand side isn't a boolean, but is it a constant?
      // If so, we know this comparison will be false
      if (matchPattern(rhs, mlir::m_Constant())) {
        rewriter.replaceOpWithNewOp<ConstantBoolOp>(op, i1Ty, false);
        return success();
      }
    } else if (!lhs.isa<BlockArgument>() && matchPattern(rhs, rhsBoolPattern)) {
      // Left-hand side isn't a boolean, but is it a constant?
      // If so, we know this comparison will be false
      if (matchPattern(lhs, mlir::m_Constant())) {
        rewriter.replaceOpWithNewOp<ConstantBoolOp>(op, i1Ty, false);
        return success();
      }
    }

    // If one or both sides of the comparison are not constant,
    // make sure the types match. If they don't, insert a cast
    // where appropriate
    auto lhsConst = matchPattern(lhs, mlir::m_Constant());
    auto rhsConst = matchPattern(rhs, mlir::m_Constant());
    if (!lhsConst || !rhsConst) {
      auto lhsTy = lhs.getType();
      auto rhsTy = rhs.getType();
      if (lhsTy != rhsTy) {
        // If both sides are terms and one side is opaque and the other is not,
        // cast to the concrete type. If one side is not a term, cast the non-term
        // type to the term type. If neither side are terms, we raise an error
        if (lhsTy.isa<OpaqueTermType>() && rhsTy.isa<OpaqueTermType>()) {
          auto lTy = lhsTy.cast<OpaqueTermType>();
          auto rTy = rhsTy.cast<OpaqueTermType>();
          if (lTy.isOpaque()) {
            Value newRhs = rewriter.create<CastOp>(op.getLoc(), rhs, lhsTy);
            auto rhsOperands = op.rhsMutable();
            rhsOperands.assign(newRhs);
            return success();
          }
          if (rTy.isOpaque()) {
            Value newLhs = rewriter.create<CastOp>(op.getLoc(), lhs, rhsTy);
            auto lhsOperands = op.lhsMutable();
            lhsOperands.assign(newLhs);
            return success();
          }

          // Comparing immediates to boxed values will always fail
          if ((lTy.isImmediate() && !rTy.isImmediate()) || (!lTy.isImmediate() && rTy.isImmediate())) {
            rewriter.replaceOpWithNewOp<ConstantBoolOp>(op, i1Ty, false);
            return success();
          }

          // Comparing immediates requires no type conversion,
          // and any other term comparisons will be done at runtime
          return success();
        }

        // If only one side is a term, we need to decide if we can cast
        // to the non-term type for efficiency, or cast to term type for
        // a call to the runtime
        if (lhsTy.isa<OpaqueTermType>() || rhsTy.isa<OpaqueTermType>()) {
          Type nonTermTy;
          OpaqueTermType termTy;
          bool lhsTerm = false;
          if (auto lTy = lhsTy.dyn_cast_or_null<OpaqueTermType>()) {
            termTy = lTy;
            nonTermTy = rhsTy;
            lhsTerm = true;
          } else {
            termTy = rhsTy.cast<OpaqueTermType>();
            nonTermTy = lhsTy;
          }

          if (termTy.isBoolean() && nonTermTy.isInteger(1)) {
            if (lhsTerm) {
              Value newLhs = rewriter.create<CastOp>(op.getLoc(), lhs, rhsTy);
              auto lhsOperands = op.lhsMutable();
              lhsOperands.assign(newLhs);
              return success();
            } else {
              Value newRhs = rewriter.create<CastOp>(op.getLoc(), rhs, lhsTy);
              auto rhsOperands = op.rhsMutable();
              rhsOperands.assign(newRhs);
              return success();
            }
          } else {
            // For now, just cast the non-term type to term
            if (lhsTerm) {
              Value newRhs = rewriter.create<CastOp>(op.getLoc(), rhs, lhsTy);
              auto rhsOperands = op.rhsMutable();
              rhsOperands.assign(newRhs);
              return success();
            } else {
              Value newLhs = rewriter.create<CastOp>(op.getLoc(), lhs, rhsTy);
              auto lhsOperands = op.lhsMutable();
              lhsOperands.assign(newLhs);
              return success();
            }
          }
        }

        // Neither side are terms, and have different types this shouldn't ever happen
        op.emitError("invalid operand types for equality comparison, expected at least one operand to be a valid term type");
        return failure();
      } else {
        return success();
      }
    } else {
        // Types match we're done
        return success();
    }

    // Both sides are constant, we can fold this comparison,
    // our fold implementation should handle this, so do nothing
    // for now
    return success();
  }
};
}  // end anonymous namespace

void CmpEqOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                          MLIRContext *context) {
  results.insert<CanonicalizeEqualityComparison>(context);
}

//===----------------------------------------------------------------------===//
// Constant*Op
//===----------------------------------------------------------------------===//

static ParseResult parseConstantOp(OpAsmParser &parser,
                                   OperationState &result) {
  Attribute valueAttr;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseAttribute(valueAttr, "value", result.attributes))
    return failure();

  auto type = valueAttr.getType();

  // Add the attribute type to the list.
  return parser.addTypeToList(type, result.types);
}

template <typename ConstantOp>
static void printConstantOp(OpAsmPrinter &p, ConstantOp &op) {
  p << op.getOperationName() << ' ';
  p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{"value"});

  if (op.getAttrs().size() > 1) p << ' ';
  p << op.getValue();
}

template <typename ConstantOp>
static LogicalResult verifyConstantOp(ConstantOp &) {
  // TODO
  return success();
}

OpFoldResult ConstantIntOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");
  return getValue();
}

OpFoldResult ConstantBigIntOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");
  return getValue();
}

OpFoldResult ConstantFloatOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");
  return getValue();
}

OpFoldResult ConstantBoolOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");
  return getValue();
}

OpFoldResult ConstantAtomOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");
  return getValue();
}

OpFoldResult ConstantNilOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");
  return getValue();
}

OpFoldResult ConstantNoneOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");
  return getValue();
}


//===----------------------------------------------------------------------===//
// eir.neg
//===----------------------------------------------------------------------===//

namespace {
/// Fold negations of constants into negated constants
struct ApplyConstantNegations : public OpRewritePattern<NegOp> {
  using OpRewritePattern<NegOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(NegOp op,
                                PatternRewriter &rewriter) const override {
    auto rhs = op.rhs();

    APInt intVal;
    auto intPattern = mlir::m_Op<CastOp>(m_ConstInt(&intVal));
    if (matchPattern(rhs, intPattern)) {
      auto castOp = dyn_cast<CastOp>(rhs.getDefiningOp());
      auto castType = castOp.getType();
      intVal.negate();
      if (castOp.getSourceType().isa<FixnumType>()) {
        auto newInt = rewriter.create<ConstantIntOp>(op.getLoc(), intVal);
        rewriter.replaceOpWithNewOp<CastOp>(op, newInt.getResult(), castType);
      } else {
        auto newInt = rewriter.create<ConstantBigIntOp>(op.getLoc(), intVal);
        rewriter.replaceOpWithNewOp<CastOp>(op, newInt.getResult(), castType);
      }
      return success();
    }

    APFloat fltVal(0.0);
    auto floatPattern = mlir::m_Op<CastOp>(m_ConstFloat(&fltVal));
    if (matchPattern(rhs, floatPattern)) {
      auto castType = dyn_cast<CastOp>(rhs.getDefiningOp()).getType();
      APFloat newFltVal = -fltVal;
      auto newFlt = rewriter.create<ConstantFloatOp>(op.getLoc(), newFltVal);
      rewriter.replaceOpWithNewOp<CastOp>(op, newFlt.getResult(), castType);
      return success();
    }

    return failure();
  }
};
}  // end anonymous namespace

void NegOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                        MLIRContext *context) {
  results.insert<ApplyConstantNegations>(context);
}

//===----------------------------------------------------------------------===//
// MallocOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(MallocOp op) {
  auto resultType = op.getResult().getType();
  if (!resultType.isa<PtrType>())
    return op.emitOpError("result must be of pointer type");

  auto allocType = op.getAllocType();
  if (auto termType = allocType.dyn_cast_or_null<OpaqueTermType>()) {
    if (!termType.isBoxable())
      return op.emitOpError("cannot malloc an unboxable term type");

    Optional<Value> arityVal = op.arity();
    if (arityVal.hasValue()) {
      if (!termType.hasDynamicExtent())
        return op.emitOpError("it is invalid to specify arity with statically-sized type");
    } else {
      if (termType.hasDynamicExtent())
        return op.emitOpError("cannot malloc a type with dynamic extent without specifying arity");
    }
  } else {
    return op.emitOpError("it is currently unsupported to malloc non-term types");
  }

  return success();
}

namespace {
/// Fold malloc operations with no uses. Malloc has side effects on the heap,
/// but can still be deleted if it has zero uses.
struct SimplifyDeadMalloc : public OpRewritePattern<MallocOp> {
  using OpRewritePattern<MallocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MallocOp alloc,
                                PatternRewriter &rewriter) const override {
    if (alloc.use_empty()) {
      rewriter.eraseOp(alloc);
      return success();
    }
    return failure();
  }
};
}  // end anonymous namespace.

void MallocOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                           MLIRContext *context) {
  results.insert<SimplifyDeadMalloc>(context);
}

//===----------------------------------------------------------------------===//
// eir.invoke
//===----------------------------------------------------------------------===//

Optional<MutableOperandRange> InvokeOp::getMutableSuccessorOperands(
    unsigned index) {
  assert(index < getNumSuccessors() && "invalid successor index");
  return index == okIndex ? llvm::None : Optional(errDestOperandsMutable());
}

//===----------------------------------------------------------------------===//
// eir.yield.check
//===----------------------------------------------------------------------===//

Optional<MutableOperandRange> YieldCheckOp::getMutableSuccessorOperands(
    unsigned index) {
  assert(index < getNumSuccessors() && "invalid successor index");
  return index == trueIndex ? trueDestOperandsMutable()
                            : falseDestOperandsMutable();
}

Block *YieldCheckOp::getSuccessorForOperands(ArrayRef<Attribute> operands) {
  if (IntegerAttr condAttr = operands.front().dyn_cast_or_null<IntegerAttr>())
    return condAttr.getValue().isOneValue() ? trueDest() : falseDest();
  return nullptr;
}

//===----------------------------------------------------------------------===//
// TableGen Output
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "lumen/EIR/IR/EIROps.cpp.inc"

}  // namespace eir
}  // namespace lumen
