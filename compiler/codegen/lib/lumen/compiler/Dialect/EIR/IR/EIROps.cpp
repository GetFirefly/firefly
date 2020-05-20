#include "lumen/compiler/Dialect/EIR/IR/EIROps.h"

#include <iterator>
#include <vector>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/SMLoc.h"
#include "lumen/compiler/Dialect/EIR/IR/EIRAttributes.h"
#include "lumen/compiler/Dialect/EIR/IR/EIRTypes.h"
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
#include "mlir/Support/STLExtras.h"

using namespace lumen;
using namespace lumen::eir;

using ::llvm::ArrayRef;
using ::llvm::SmallVector;
using ::llvm::StringRef;

namespace lumen {
namespace eir {

//===----------------------------------------------------------------------===//
// eir.func
//===----------------------------------------------------------------------===//

static ParseResult parseFuncOp(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType = [](Builder &builder, ArrayRef<Type> argTypes,
                          ArrayRef<Type> results, impl::VariadicFlag,
                          std::string &) {
    return builder.getFunctionType(argTypes, results);
  };
  return impl::parseFunctionLikeOp(parser, result, /*allowVariadic=*/false,
                                   buildFuncType);
}

static void print(OpAsmPrinter &p, FuncOp &op) {
  FunctionType fnType = op.getType();
  impl::printFunctionLikeOp(p, op, fnType.getInputs(), /*isVariadic=*/false,
                            fnType.getResults());
}

FuncOp FuncOp::create(Location location, StringRef name, FunctionType type,
                      ArrayRef<NamedAttribute> attrs) {
  OperationState state(location, FuncOp::getOperationName());
  Builder builder(location->getContext());
  FuncOp::build(&builder, state, name, type, attrs);
  return cast<FuncOp>(Operation::create(state));
}

void FuncOp::build(Builder *builder, OperationState &result, StringRef name,
                   FunctionType type, ArrayRef<NamedAttribute> attrs,
                   ArrayRef<NamedAttributeList> argAttrs) {
  result.addRegion();
  result.addAttribute(SymbolTable::getSymbolAttrName(),
                      builder->getStringAttr(name));
  result.addAttribute("type", TypeAttr::get(type));
  result.attributes.append(attrs.begin(), attrs.end());
  if (argAttrs.empty()) {
    return;
  }

  unsigned numInputs = type.getNumInputs();
  assert(numInputs == argAttrs.size() &&
         "expected as many argument attribute lists as arguments");
  SmallString<8> argAttrName;
  for (unsigned i = 0; i < numInputs; ++i) {
    if (auto argDict = argAttrs[i].getDictionary()) {
      result.addAttribute(getArgAttrName(i, argAttrName), argDict);
    }
  }
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

namespace {
struct SimplifyBrToBlockWithSinglePred : public OpRewritePattern<BranchOp> {
  using OpRewritePattern<BranchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BranchOp op,
                                PatternRewriter &rewriter) const override {
    // Check that the successor block has a single predecessor.
    Block *succ = op.getDest();
    Block *opParent = op.getOperation()->getBlock();
    if (succ == opParent || !has_single_element(succ->getPredecessors()))
      return failure();

    // Merge the successor into the current block and erase the branch.
    rewriter.mergeBlocks(succ, opParent, op.getOperands());
    rewriter.eraseOp(op);
    return success();
  }
};
}  // namespace

void BranchOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                           MLIRContext *context) {
  results.insert<SimplifyBrToBlockWithSinglePred>(context);
}

Optional<OperandRange> BranchOp::getSuccessorOperands(unsigned index) {
  assert(index == 0 && "invalid successor index");
  return getOperands();
}

bool BranchOp::canEraseSuccessorOperand() { return true; }

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
    if (matchPattern(condbr.getCondition(), m_NonZero())) {
      // True branch taken.
      rewriter.replaceOpWithNewOp<BranchOp>(condbr, condbr.getTrueDest(),
                                            condbr.getTrueOperands());
      return success();
    } else if (matchPattern(condbr.getCondition(), m_Zero())) {
      // False branch taken.
      rewriter.replaceOpWithNewOp<BranchOp>(condbr, condbr.getFalseDest(),
                                            condbr.getFalseOperands());
      return success();
    }
    return failure();
  }
};
}  // end anonymous namespace.

void CondBranchOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<SimplifyConstCondBranchPred>(context);
}

Optional<OperandRange> CondBranchOp::getSuccessorOperands(unsigned index) {
  assert(index < getNumSuccessors() && "invalid successor index");
  return index == trueIndex ? getTrueOperands() : getFalseOperands();
}

bool CondBranchOp::canEraseSuccessorOperand() { return true; }

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
// eir.yield_check
//===----------------------------------------------------------------------===//

Optional<OperandRange> YieldCheckOp::getSuccessorOperands(unsigned index) {
  assert(index == 0 && "invalid successor index");
  return getOperands();
}

bool YieldCheckOp::canEraseSuccessorOperand() { return true; }

//===----------------------------------------------------------------------===//
// eir.is_type
//===----------------------------------------------------------------------===//

static LogicalResult verify(IsTypeOp op) {
  auto typeAttr = op.getAttrOfType<TypeAttr>("type");
  if (!typeAttr) return op.emitOpError("requires type attribute named 'type'");

  auto resultType = op.getResultType();
  if (!resultType.isa<BooleanType>() && !resultType.isInteger(1)) {
    return op.emitOpError(
        "requires result type to be of type i1 or !eir.boolean");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// eir.cast
//===----------------------------------------------------------------------===//

static bool areCastCompatible(OpaqueTermType srcType, OpaqueTermType destType) {
  // Casting an immediate to an opaque term is always allowed
  if (destType.isOpaque()) return srcType.isImmediate();
  // Casting an opaque term to any term type is always allowed
  if (srcType.isOpaque()) return true;
  // A cast must be to an immediate-sized type
  if (!destType.isImmediate()) return false;
  // Box-to-box casts are always allowed
  if (srcType.isBox() & destType.isBox()) return true;
  // Only header types can be boxed
  if (destType.isBox() && !srcType.isBoxable()) return false;
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
  auto opType = op.getOperand().getType();
  auto resType = op.getType();
  if (auto opTermType = opType.dyn_cast_or_null<OpaqueTermType>()) {
    if (auto resTermType = resType.dyn_cast_or_null<OpaqueTermType>()) {
      if (!areCastCompatible(opTermType, resTermType)) {
        return op.emitError("operand type ")
               << opType << " and result type " << resType
               << " are not cast compatible";
      }
      return success();
    }
    return op.emitError(
               "invalid cast type for CastOp, expected term type, got ")
           << resType;
  }
  return op.emitError(
             "invalid operand type for CastOp, expected term type, got ")
         << opType;
}

//===----------------------------------------------------------------------===//
// eir.match
//===----------------------------------------------------------------------===//

void lowerPatternMatch(OpBuilder &builder, Location loc, Value selector,
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
        auto castOp =
            builder.create<CastOp>(branchLoc, selectorArg, boxedConsType);
        auto boxedCons = castOp.getResult();
        auto getHeadOp =
            builder.create<GetElementPtrOp>(branchLoc, boxedCons, 0);
        auto getTailOp =
            builder.create<GetElementPtrOp>(branchLoc, boxedCons, 1);
        auto headPointer = getHeadOp.getResult();
        auto tailPointer = getTailOp.getResult();
        auto headLoadOp = builder.create<LoadOp>(branchLoc, headPointer);
        auto tailLoadOp = builder.create<LoadOp>(branchLoc, tailPointer);
        // 3. Unconditionally branch to the destination, with head/tail as
        // additional destArgs
        SmallVector<Value, 2> destArgs(
            {baseDestArgs.begin(), baseDestArgs.end()});
        destArgs.push_back(headLoadOp.getResult());
        destArgs.push_back(tailLoadOp.getResult());
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
        auto castOp =
            builder.create<CastOp>(branchLoc, selectorArg, boxedTupleType);
        auto boxedTuple = castOp.getResult();
        SmallVector<Value, 2> destArgs(
            {baseDestArgs.begin(), baseDestArgs.end()});
        destArgs.reserve(arity);
        for (int64_t i = 0; i < arity; i++) {
          auto getElemOp =
              builder.create<GetElementPtrOp>(branchLoc, boxedTuple, i + 1);
          auto elemPtr = getElemOp.getResult();
          auto elemLoadOp = builder.create<LoadOp>(branchLoc, elemPtr);
          destArgs.push_back(elemLoadOp.getResult());
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
        split2->addArgument(selectorType);
        Block *split = builder.createBlock(region, Region::iterator(split2));
        split->addArgument(selectorType);
        builder.restoreInsertionPoint(cip);
        auto *pattern = b.getPatternTypeOrNull<MapPattern>();
        auto key = pattern->getKey();
        auto mapType = builder.getType<MapType>();
        auto isMapOp =
            builder.create<IsTypeOp>(branchLoc, selectorArg, mapType);
        auto isMapCond = isMapOp.getResult();
        auto ifOp = builder.create<CondBranchOp>(
            branchLoc, isMapCond, split, withSelectorArgs, nextPatternBlock,
            withSelectorArgs);
        // 2. In the split, call runtime function `is_map_key` to confirm
        // existence of the key in the map,
        //    then conditionally branch to the second split if successful,
        //    otherwise the next pattern
        builder.setInsertionPointToEnd(split);
        Value splitSelector = split->getArgument(0);
        ArrayRef<Value> splitSelectorArgs{splitSelector};
        ArrayRef<Type> getKeyResultTypes = {termType};
        ArrayRef<Value> getKeyArgs = {key, splitSelector};
        auto hasKeyOp = builder.create<CallOp>(
            branchLoc, "erlang::is_map_key/2", getKeyResultTypes, getKeyArgs);
        auto hasKeyCondTerm = hasKeyOp.getResult(0);
        auto toBoolOp = builder.create<CastOp>(branchLoc, hasKeyCondTerm,
                                               builder.getType<BooleanType>());
        auto hasKeyCond = toBoolOp.getResult();
        builder.create<CondBranchOp>(branchLoc, hasKeyCond, split2,
                                     splitSelectorArgs, nextPatternBlock,
                                     splitSelectorArgs);
        // 3. In the second split, call runtime function `map_get` to obtain the
        // value for the key
        builder.setInsertionPointToEnd(split2);
        ArrayRef<Type> mapGetResultTypes = {termType};
        ArrayRef<Value> mapGetArgs = {key, split2->getArgument(0)};
        auto mapGetOp = builder.create<CallOp>(branchLoc, "erlang::map_get/2",
                                               mapGetResultTypes, mapGetArgs);
        auto valueTerm = mapGetOp.getResult(0);
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
        assert(false && "binary match patterns are not implemented yet");
        auto *pattern = b.getPatternTypeOrNull<BinaryPattern>();
        break;
      }

      default:
        llvm_unreachable("unexpected match pattern type!");
    }
  }
  builder.restoreInsertionPoint(finalIp);
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

template <typename Op>
OpFoldResult foldConstantOp(Op *op, ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");
  return op->getValue();
}

OpFoldResult ConstantFloatOp::fold(ArrayRef<Attribute> operands) {
  return foldConstantOp(this, operands);
}
OpFoldResult ConstantIntOp::fold(ArrayRef<Attribute> operands) {
  return foldConstantOp(this, operands);
}
OpFoldResult ConstantBigIntOp::fold(ArrayRef<Attribute> operands) {
  return foldConstantOp(this, operands);
}
OpFoldResult ConstantAtomOp::fold(ArrayRef<Attribute> operands) {
  return foldConstantOp(this, operands);
}
OpFoldResult ConstantBinaryOp::fold(ArrayRef<Attribute> operands) {
  return foldConstantOp(this, operands);
}
OpFoldResult ConstantNilOp::fold(ArrayRef<Attribute> operands) {
  return foldConstantOp(this, operands);
}
OpFoldResult ConstantNoneOp::fold(ArrayRef<Attribute> operands) {
  return foldConstantOp(this, operands);
}
OpFoldResult ConstantTupleOp::fold(ArrayRef<Attribute> operands) {
  return foldConstantOp(this, operands);
}
OpFoldResult ConstantConsOp::fold(ArrayRef<Attribute> operands) {
  return foldConstantOp(this, operands);
}
OpFoldResult ConstantListOp::fold(ArrayRef<Attribute> operands) {
  return foldConstantOp(this, operands);
}
OpFoldResult ConstantMapOp::fold(ArrayRef<Attribute> operands) {
  return foldConstantOp(this, operands);
}

//===----------------------------------------------------------------------===//
// MallocOp
//===----------------------------------------------------------------------===//

static void print(OpAsmPrinter &p, MallocOp op) {
  p << MallocOp::getOperationName();

  OpaqueTermType boxedType = op.getAllocType();
  p.printOperands(op.getOperands());
  p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{"type"});
  p << " : " << BoxType::get(boxedType);
}

static ParseResult parseMallocOp(OpAsmParser &parser, OperationState &result) {
  BoxType type;

  llvm::SMLoc loc = parser.getCurrentLocation();
  SmallVector<OpAsmParser::OperandType, 1> opInfo;
  SmallVector<Type, 1> types;
  if (parser.parseOperandList(opInfo)) return failure();
  if (!opInfo.empty() && parser.parseColonTypeList(types)) return failure();

  // Parse the optional dimension operand, followed by a box type.
  if (parser.resolveOperands(opInfo, types, loc, result.operands) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type))
    return failure();

  result.types.push_back(type);
  return success();
}

static LogicalResult verify(MallocOp op) {
  auto type = op.getResult().getType().dyn_cast<BoxType>();
  if (!type) return op.emitOpError("result must be a box type");

  OpaqueTermType boxedType = type.getBoxedType();
  if (!boxedType.isBoxable())
    return op.emitOpError("boxed type must be a boxable type");

  auto operands = op.getOperands();
  if (operands.empty()) {
    // There should be no dynamic dimensions in the boxed type
    if (boxedType.isTuple()) {
      auto tuple = boxedType.cast<TupleType>();
      if (tuple.hasDynamicShape())
        return op.emitOpError(
            "boxed type has dynamic extent, but no dimension operands were "
            "given");
    }
  } else {
    // There should be exactly as many dynamic dimensions as there are operands,
    // and those operands should be of integer type
    if (!boxedType.isTuple())
      return op.emitOpError(
          "only tuples are allowed to have dynamic dimensions");
    auto tuple = boxedType.cast<TupleType>();
    if (tuple.hasStaticShape())
      return op.emitOpError(
          "boxed type has static extent, but dimension operands were given");
    if (tuple.getArity() != operands.size())
      return op.emitOpError(
          "number of dimension operands does not match the number of dynamic "
          "dimensions");
  }

  return success();
}

/// Matches a ConstantIntOp or mlir::ConstantIndexOp.
static mlir::detail::op_matcher<ConstantIntOp> m_ConstantDimension() {
  return mlir::detail::op_matcher<ConstantIntOp>();
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

int64_t calculateAllocSize(unsigned pointerSizeInBits, BoxType boxType) {
  auto boxedType = boxType.getBoxedType();
  if (boxedType.isTuple()) {
    auto tuple = boxedType.cast<TupleType>();
    return tuple.getSizeInBytes();
  } else if (boxedType.isNonEmptyList()) {
    return 2 * (pointerSizeInBits / 8);
  }
  assert(false && "unimplemented boxed type in calculateAllocSize");
}

bool InvokeClosureOp::canEraseSuccessorOperand() { return true; }

Optional<OperandRange> InvokeClosureOp::getSuccessorOperands(unsigned index) {
  switch (index) {
    case 0:
      return llvm::None;
    case 1:
      return getErrOperands();
    default:
      assert(false && "invalid successor index");
  }
}

bool InvokeOp::canEraseSuccessorOperand() { return true; }

Optional<OperandRange> InvokeOp::getSuccessorOperands(unsigned index) {
  switch (index) {
    case 0:
      return llvm::None;
    case 1:
      return getErrOperands();
    default:
      assert(false && "invalid successor index");
  }
}

//===----------------------------------------------------------------------===//
// eir.receive_wait
//===----------------------------------------------------------------------===//

Optional<OperandRange> ReceiveWaitOp::getSuccessorOperands(unsigned index) {
  assert(index < getNumSuccessors() && "invalid successor index");
  return index == timeoutIndex ? getTimeoutOperands() : getCheckOperands();
}

bool ReceiveWaitOp::canEraseSuccessorOperand() { return true; }


//===----------------------------------------------------------------------===//
// TableGen Output
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "lumen/compiler/Dialect/EIR/IR/EIROps.cpp.inc"

}  // namespace eir
}  // namespace lumen
