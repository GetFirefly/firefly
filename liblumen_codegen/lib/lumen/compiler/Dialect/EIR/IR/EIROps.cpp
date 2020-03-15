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

  PatternMatchResult matchAndRewrite(CallIndirectOp indirectCall,
                                     PatternRewriter &rewriter) const override {
    // Check that the callee is a constant callee.
    FlatSymbolRefAttr calledFn;
    if (!matchPattern(indirectCall.getCallee(), ::mlir::m_Constant(&calledFn)))
      return matchFailure();

    // Replace with a direct call.
    rewriter.replaceOpWithNewOp<CallOp>(indirectCall, calledFn,
                                        indirectCall.getResultTypes(),
                                        indirectCall.getArgOperands());
    return matchSuccess();
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

static ParseResult parseBranchOp(OpAsmParser &parser, OperationState &result) {
  Block *dest;
  SmallVector<Value, 4> destOperands;
  if (failed(parser.parseSuccessorAndUseList(dest, destOperands))) {
    return failure();
  }
  result.addSuccessor(dest, destOperands);
  if (failed(parser.parseOptionalAttrDict(result.attributes))) {
    return failure();
  }
  return success();
}

static void print(OpAsmPrinter &p, BranchOp &op) {
  p << op.getOperationName() << ' ';
  p.printSuccessorAndUseList(op.getOperation(), 0);
  p.printOptionalAttrDict(op.getAttrs());
}

static LogicalResult verify(BranchOp op) {
  // TODO
  return success();
}

Block *BranchOp::getDest() { return getOperation()->getSuccessor(0); }

void BranchOp::setDest(Block *block) {
  return getOperation()->setSuccessor(block, 0);
}

void BranchOp::eraseOperand(unsigned index) {
  getOperation()->eraseSuccessorOperand(0, index);
}

//===----------------------------------------------------------------------===//
// eir.cond_br
//===----------------------------------------------------------------------===//

static ParseResult parseCondBranchOp(OpAsmParser &parser,
                                     OperationState &result) {
  SmallVector<Value, 4> destOperands;
  Block *dest;
  OpAsmParser::OperandType condInfo;

  // Parse the condition.
  Type boolTy = parser.getBuilder().getType<BooleanType>();
  if (failed(parser.parseOperand(condInfo)) || failed(parser.parseComma()) ||
      failed(parser.resolveOperand(condInfo, boolTy, result.operands))) {
    return parser.emitError(parser.getNameLoc(),
                            "expected boolean condition type");
  }

  // Parse the true successor.
  if (failed(parser.parseSuccessorAndUseList(dest, destOperands))) {
    return failure();
  }
  result.addSuccessor(dest, destOperands);

  // Parse the false successor.
  destOperands.clear();
  if (failed(parser.parseComma()) ||
      failed(parser.parseSuccessorAndUseList(dest, destOperands))) {
    return failure();
  }
  result.addSuccessor(dest, destOperands);

  if (failed(parser.parseOptionalAttrDict(result.attributes))) {
    return failure();
  }

  return success();
}

static void print(OpAsmPrinter &p, CondBranchOp &op) {
  p << op.getOperationName() << ' ';
  p.printOperand(op.getCondition());
  p << ", ";
  p.printSuccessorAndUseList(op.getOperation(), CondBranchOp::trueIndex);
  p << ", ";
  p.printSuccessorAndUseList(op.getOperation(), CondBranchOp::falseIndex);
  p.printOptionalAttrDict(op.getAttrs());
}

static LogicalResult verify(CondBranchOp op) {
  // TODO
  return success();
}

//===----------------------------------------------------------------------===//
// eir.return
//===----------------------------------------------------------------------===//

static ParseResult parseReturnOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 2> opInfo;
  SmallVector<Type, 2> types;
  llvm::SMLoc loc = parser.getCurrentLocation();
  return failure(parser.parseOperandList(opInfo) ||
                 (!opInfo.empty() && parser.parseColonTypeList(types)) ||
                 parser.resolveOperands(opInfo, types, loc, result.operands));
}

static void print(OpAsmPrinter &p, ReturnOp &op) {
  p << op.getOperationName();
  if (op.getNumOperands() > 0) {
    p << ' ';
    p.printOperands(op.operand_begin(), op.operand_end());
    p.printOptionalAttrDict(op.getAttrs());
    p << " : ";
    interleaveComma(op.getOperandTypes(), p);
  }
}

//===----------------------------------------------------------------------===//
// eir.yield.check
//===----------------------------------------------------------------------===//
static ParseResult parseYieldCheckOp(OpAsmParser &parser,
                                     OperationState &result) {
  SmallVector<Value, 4> destOperands;
  Block *dest;
  OpAsmParser::OperandType condInfo;

  // Parse the true successor.
  if (failed(parser.parseSuccessorAndUseList(dest, destOperands))) {
    return failure();
  }
  result.addSuccessor(dest, destOperands);

  // Parse the false successor.
  destOperands.clear();
  if (failed(parser.parseComma()) ||
      failed(parser.parseSuccessorAndUseList(dest, destOperands))) {
    return failure();
  }
  result.addSuccessor(dest, destOperands);

  if (failed(parser.parseOptionalAttrDict(result.attributes))) {
    return failure();
  }

  return success();
}

static void print(OpAsmPrinter &p, YieldCheckOp &op) {
  p << op.getOperationName() << ' ';
  p.printSuccessorAndUseList(op.getOperation(), CondBranchOp::trueIndex);
  p << ", ";
  p.printSuccessorAndUseList(op.getOperation(), CondBranchOp::falseIndex);
  p.printOptionalAttrDict(op.getAttrs());
}

static LogicalResult verify(YieldCheckOp op) {
  // TODO
  return success();
}

//===----------------------------------------------------------------------===//
// ConsOp
//===----------------------------------------------------------------------===//

void ConsOp::build(Builder *builder, OperationState &result, Value head,
                   Value tail) {
  result.addOperands(head);
  result.addOperands(tail);
  result.addTypes(builder->getType<ConsType>());
}

static LogicalResult verify(ConsOp op) {
  // TODO
  return success();
}

//===----------------------------------------------------------------------===//
// TupleOp
//===----------------------------------------------------------------------===//

void TupleOp::build(Builder *builder, OperationState &result,
                    ArrayRef<Value> elements) {
  SmallVector<Type, 1> elementTypes;
  for (auto val : elements) {
    elementTypes.push_back(val.getType());
  }
  result.addOperands(elements);
  auto tupleType = builder->getType<eir::TupleType>(elementTypes);
  result.addTypes(tupleType);
  result.addAttribute("alloca", builder->getBoolAttr(false));
}

static LogicalResult verify(TupleOp op) {
  // TODO
  return success();
}

//===----------------------------------------------------------------------===//
// ConstructMapOp
//===----------------------------------------------------------------------===//

void ConstructMapOp::build(Builder *builder, OperationState &result,
                           ArrayRef<eir::MapEntry> entries) {
  for (auto &entry : entries) {
    result.addOperands(Value::getFromOpaquePointer(entry.key));
    result.addOperands(Value::getFromOpaquePointer(entry.value));
  }
  result.addTypes(builder->getType<MapType>());
}

static LogicalResult verify(ConstructMapOp op) {
  // TODO
  return success();
}

//===----------------------------------------------------------------------===//
// BinaryPushOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(BinaryPushOp op) {
  // TODO
  return success();
}

//===----------------------------------------------------------------------===//
// IsTypeOp
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
// CastOp
//===----------------------------------------------------------------------===//

static bool areCastCompatible(OpaqueTermType srcType, OpaqueTermType destType) {
  // Casting an immediate to an opaque term is always allowed
  if (destType.isOpaque()) return srcType.isImmediate();
  // Casting an opaque term to any term type is always allowed
  if (srcType.isOpaque()) return true;
  // A cast must be to an immediate-sized type
  if (!destType.isImmediate()) return false;
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
// MatchOp
//===----------------------------------------------------------------------===//

void lowerPatternMatch(OpBuilder &builder, Value selector,
                       ArrayRef<MatchBranch> branches) {
  auto numBranches = branches.size();
  assert(numBranches > 0 && "expected at least one branch in a match");

  auto *context = builder.getContext();
  auto *currentBlock = builder.getInsertionBlock();
  auto *region = currentBlock->getParent();
  auto loc = builder.getUnknownLoc();
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
        builder.create<BranchOp>(loc, dest, baseDestArgs);
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
            builder.create<IsTypeOp>(loc, selectorArg, boxedConsType);
        auto isConsCond = isConsOp.getResult();
        auto ifOp =
            builder.create<CondBranchOp>(loc, isConsCond, split, emptyArgs,
                                         nextPatternBlock, withSelectorArgs);
        // 2. In the split, extract head and tail values of the cons cell
        builder.setInsertionPointToEnd(split);
        auto castOp = builder.create<CastOp>(loc, selectorArg, boxedConsType);
        auto boxedCons = castOp.getResult();
        auto getHeadOp = builder.create<GetElementPtrOp>(loc, boxedCons, 0);
        auto getTailOp = builder.create<GetElementPtrOp>(loc, boxedCons, 1);
        auto headPointer = getHeadOp.getResult();
        auto tailPointer = getTailOp.getResult();
        auto headLoadOp = builder.create<LoadOp>(loc, headPointer);
        auto tailLoadOp = builder.create<LoadOp>(loc, tailPointer);
        // 3. Unconditionally branch to the destination, with head/tail as
        // additional destArgs
        SmallVector<Value, 2> destArgs(
            {baseDestArgs.begin(), baseDestArgs.end()});
        destArgs.push_back(headLoadOp.getResult());
        destArgs.push_back(tailLoadOp.getResult());
        builder.create<BranchOp>(loc, dest, destArgs);
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
            builder.create<IsTypeOp>(loc, selectorArg, boxedTupleType);
        auto isTupleCond = isTupleOp.getResult();
        auto ifOp =
            builder.create<CondBranchOp>(loc, isTupleCond, split, emptyArgs,
                                         nextPatternBlock, withSelectorArgs);
        // 2. In the split, extract the tuple elements as values
        builder.setInsertionPointToEnd(split);
        auto castOp = builder.create<CastOp>(loc, selectorArg, boxedTupleType);
        auto boxedTuple = castOp.getResult();
        SmallVector<Value, 2> destArgs(
            {baseDestArgs.begin(), baseDestArgs.end()});
        destArgs.reserve(arity);
        for (int64_t i = 0; i < arity; i++) {
          auto getElemOp =
              builder.create<GetElementPtrOp>(loc, boxedTuple, i + 1);
          auto elemPtr = getElemOp.getResult();
          auto elemLoadOp = builder.create<LoadOp>(loc, elemPtr);
          destArgs.push_back(elemLoadOp.getResult());
        }
        // 3. Unconditionally branch to the destination, with the tuple elements
        // as additional destArgs
        builder.create<BranchOp>(loc, dest, destArgs);
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
        auto isMapOp = builder.create<IsTypeOp>(loc, selectorArg, mapType);
        auto isMapCond = isMapOp.getResult();
        auto ifOp = builder.create<CondBranchOp>(
            loc, isMapCond, split, withSelectorArgs, nextPatternBlock,
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
        auto hasKeyOp = builder.create<CallOp>(loc, "erlang::is_map_key/2",
                                               getKeyResultTypes, getKeyArgs);
        auto hasKeyCondTerm = hasKeyOp.getResult(0);
        auto toBoolOp = builder.create<CastOp>(loc, hasKeyCondTerm,
                                               builder.getType<BooleanType>());
        auto hasKeyCond = toBoolOp.getResult();
        builder.create<CondBranchOp>(loc, hasKeyCond, split2, splitSelectorArgs,
                                     nextPatternBlock, splitSelectorArgs);
        // 3. In the second split, call runtime function `map_get` to obtain the
        // value for the key
        builder.setInsertionPointToEnd(split2);
        ArrayRef<Type> mapGetResultTypes = {termType};
        ArrayRef<Value> mapGetArgs = {key, split2->getArgument(0)};
        auto mapGetOp = builder.create<CallOp>(loc, "erlang::map_get/2",
                                               mapGetResultTypes, mapGetArgs);
        auto valueTerm = mapGetOp.getResult(0);
        // 4. Unconditionally branch to the destination, with the key's value as
        // an additional destArg
        SmallVector<Value, 2> destArgs(baseDestArgs.begin(),
                                       baseDestArgs.end());
        destArgs.push_back(valueTerm);
        builder.create<BranchOp>(loc, dest, destArgs);
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
            builder.create<IsTypeOp>(loc, selectorArg, expectedType);
        auto isTypeCond = isTypeOp.getResult();
        builder.create<CondBranchOp>(loc, isTypeCond, dest, baseDestArgs,
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
        auto isEq = builder.create<CmpEqOp>(loc, selectorArg, expected,
                                            /*strict=*/true);
        auto isEqCond = isEq.getResult();
        builder.create<CondBranchOp>(loc, isEqCond, dest, baseDestArgs,
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
// TraceCaptureOp
//===----------------------------------------------------------------------===//

ParseResult parseTraceCaptureOp(OpAsmParser &parser, OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes)) return failure();
  return success();
}

static void print(OpAsmPrinter &p, TraceCaptureOp op) {
  p << op.getOperationName();
  p.printOptionalAttrDict(op.getOperation()->getAttrs());
}

static LogicalResult verify(TraceCaptureOp) {
  // TODO
  return success();
}

//===----------------------------------------------------------------------===//
// TraceConstructOp
//===----------------------------------------------------------------------===//

ParseResult parseTraceConstructOp(OpAsmParser &parser, OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes)) return failure();

  auto &builder = parser.getBuilder();
  std::vector<Type> resultType = {TermType::get(builder.getContext())};
  result.addTypes(resultType);
  return success();
}

static void print(OpAsmPrinter &p, TraceConstructOp op) {
  p << op.getOperationName();
  p.printOptionalAttrDict(op.getOperation()->getAttrs());
}

static LogicalResult verify(TraceConstructOp) {
  // TODO
  return success();
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

//===----------------------------------------------------------------------===//
// MallocOp
//===----------------------------------------------------------------------===//

static void print(OpAsmPrinter &p, MallocOp op) {
  p << MallocOp::getOperationName();

  BoxType type = op.getAllocType();
  p.printOperands(op.getOperands());
  p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{"map"});
  p << " : " << type;
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

  PatternMatchResult matchAndRewrite(MallocOp alloc,
                                     PatternRewriter &rewriter) const override {
    if (alloc.use_empty()) {
      rewriter.eraseOp(alloc);
      return matchSuccess();
    }
    return matchFailure();
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

//===----------------------------------------------------------------------===//
// TableGen Output
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "lumen/compiler/Dialect/EIR/IR/EIROps.cpp.inc"

}  // namespace eir
}  // namespace lumen
