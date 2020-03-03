#include "lumen/compiler/Dialect/EIR/IR/EIROps.h"
#include "lumen/compiler/Dialect/EIR/IR/EIRTypes.h"
#include "lumen/compiler/Dialect/EIR/IR/EIRAttributes.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/STLExtras.h"

#include "llvm/Support/Casting.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/SMLoc.h"

#include <iterator>
#include <vector>

using namespace lumen;
using namespace lumen::eir;

using ::llvm::SmallVector;
using ::llvm::ArrayRef;
using ::llvm::StringRef;

namespace lumen {
namespace eir {

//===----------------------------------------------------------------------===//
// eir.func
//===----------------------------------------------------------------------===//

static ParseResult parseFuncOp(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType = [](Builder &builder,
                          ArrayRef<Type> argTypes,
                          ArrayRef<Type> results,
                          impl::VariadicFlag,
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
                   FunctionType type,
                   ArrayRef<NamedAttribute> attrs,
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
// eir.yield
//===----------------------------------------------------------------------===//

static ParseResult parseYieldOp(OpAsmParser &parser, OperationState &result) {
  return parser.parseOptionalAttrDict(result.attributes);
}

static void print(OpAsmPrinter &p, YieldOp &op) {
  p << op.getOperationName();
  p.printOptionalAttrDict(op.getAttrs());
}

//===----------------------------------------------------------------------===//
// ConsOp
//===----------------------------------------------------------------------===//

void ConsOp::build(Builder *builder, OperationState &result, Value head, Value tail) {
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

void TupleOp::build(Builder *builder, OperationState &result, ArrayRef<Value> elements) {
  SmallVector<Type, 1> elementTypes;
  for (auto val : elements) {
    elementTypes.push_back(val.getType());
  }
  result.addOperands(elements);
  auto tupleType = builder->getType<eir::TupleType>(elementTypes);
  result.addTypes(tupleType);
}

static LogicalResult verify(TupleOp op) {
  // TODO
  return success();
}

//===----------------------------------------------------------------------===//
// ConstructMapOp
//===----------------------------------------------------------------------===//

void ConstructMapOp::build(Builder *builder, OperationState &result, ArrayRef<eir::MapEntry> entries) {
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
// IfOp
//===----------------------------------------------------------------------===//

void IfOp::build(Builder *builder, OperationState &result, Value cond,
                 bool withOtherwiseRegion) {
  result.addOperands(cond);

  Region *ifRegion = result.addRegion();
  Region *elseRegion = result.addRegion();
  Region *otherwiseRegion = result.addRegion();

  OpBuilder opBuilder(builder->getContext());

  Block *ifEntry = opBuilder.createBlock(ifRegion);
  Block *elseEntry = opBuilder.createBlock(elseRegion);
  Block *otherwiseEntry = opBuilder.createBlock(otherwiseRegion);
  if (!withOtherwiseRegion) {
    opBuilder.create<eir::UnreachableOp>(result.location);
  }
}

static ParseResult parseIfOp(OpAsmParser &parser,
                             OperationState &result) {
  // Create the regions
  result.regions.reserve(3);
  Region *ifRegion = result.addRegion();
  Region *elseRegion = result.addRegion();
  Region *otherwiseRegion = result.addRegion();

  auto &builder = parser.getBuilder();
  OpAsmParser::OperandType cond;
  Type i1Type = builder.getIntegerType(1);
  if (parser.parseOperand(cond) ||
      parser.resolveOperand(cond, i1Type, result.operands))
    return failure();

  // Parse the 'if' region.
  if (parser.parseRegion(*ifRegion, {}, {}))
    return failure();

  // Parse the 'else' region.
  if (parser.parseKeyword("else") || parser.parseRegion(*elseRegion, {}, {}))
    return failure();

  // If we find an 'otherwise' keyword then parse the 'otherwise' region.
  if (!parser.parseOptionalKeyword("otherwise")) {
    if (parser.parseRegion(*otherwiseRegion, {}, {}))
      return failure();
  }

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}

static void print(OpAsmPrinter &p, IfOp op) {
  p << IfOp::getOperationName() << " " << op.condition();
  p.printRegion(op.ifRegion(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);

  p << " else";
  p.printRegion(op.elseRegion(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
  // Print the 'otherwise' region if it exists and has a block.
  auto &otherwiseRegion = op.otherwiseRegion();
  if (!otherwiseRegion.empty()) {
    p << " otherwise";
    p.printRegion(otherwiseRegion,
                  /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/false);
  }

  p.printOptionalAttrDict(op.getAttrs());
}

static LogicalResult verify(IfOp op) {
  // Verify that the entry of each child region does not have arguments.
  for (auto &region : op.getOperation()->getRegions()) {
    if (region.empty())
      continue;

    // Non-empty regions must contain a single basic block.
    if (std::next(region.begin()) != region.end())
      return op.emitOpError("expects region #")
             << region.getRegionNumber() << " to have 0 or 1 blocks";

    for (auto &b : region) {
      // Verify that the block is not empty
      if (b.empty())
        return op.emitOpError("expects a non-empty block");

      // Verify that the block takes no arguments
      if (b.getNumArguments() != 0)
        return op.emitOpError(
            "requires that child entry blocks have no arguments");

      // Verify that block terminates with valid terminator
      Operation &terminator = b.back();
      if (isa<ReturnOp>(terminator))
        continue;
      else if (isa<BranchOp>(terminator))
        continue;
      else if (isa<CondBranchOp>(terminator))
        continue;
      else if (isa<CallOp>(terminator))
        continue;
      else if (isa<eir::UnreachableOp>(terminator))
        continue;

      return op
          .emitOpError("expects regions to end with 'return', 'br', 'cond_br', "
                       "'eir.unreachable' or 'eir.call', found '" +
                       terminator.getName().getStringRef() + "'")
          .attachNote();
    }
  }

  return success();
}
  
//===----------------------------------------------------------------------===//
// IsTypeOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(IsTypeOp op) {
  auto typeAttr = op.getAttrOfType<TypeAttr>("type");
  if (!typeAttr)
    return op.emitOpError("requires type attribute named 'type'");

  auto resultType = op.getResultType();
  if (!resultType.isa<BooleanType>() && !resultType.isInteger(1)) {
    return op.emitOpError("requires result type to be of type i1 or !eir.boolean");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

static bool areCastCompatible(OpaqueTermType srcType, OpaqueTermType destType) {
  // Casting an immediate to an opaque term is always allowed
  if (destType.isOpaque())
    return srcType.isImmediate();
  // Casting an opaque term to any term type is always allowed
  if (srcType.isOpaque())
    return true;
  // A cast must be to an immediate-sized type
  if (!destType.isImmediate())
    return false;
  // Only header types can be boxed
  if (destType.isBox() && !srcType.isBoxable())
    return false;
  // Only support casts between compatible types
  if (srcType.isNumber() && !destType.isNumber())
    return false;
  if (srcType.isInteger() && !destType.isInteger())
    return false;
  if (srcType.isFloat() && !destType.isFloat())
    return false;
  if (srcType.isList() && !destType.isList())
    return false;
  if (srcType.isBinary() && !destType.isBinary())
    return false;
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
          << opType << " and result type "
          << resType << " are not cast compatible";
      }
      return success();
    }
    return op.emitError("invalid cast type for CastOp, expected term type, got ") << resType;
  }
  return op.emitError("invalid operand type for CastOp, expected term type, got ") << opType;
}


//===----------------------------------------------------------------------===//
// MatchOp
//===----------------------------------------------------------------------===//

void lowerPatternMatch(OpBuilder &builder,
                       Value selector,
                       ArrayRef<MatchBranch> branches) {
  assert(branches.size() > 0 && "expected at least one branch in a match");

  auto *context = builder.getContext();
  auto *currentBlock = builder.getInsertionBlock();
  auto *region = currentBlock->getParent();
  auto loc = builder.getUnknownLoc();

  // Save our insertion point in the current block
  auto startIp = builder.saveInsertionPoint();

  // Create blocks for all branches
  SmallVector<Block *, 2> blocks;
  blocks.reserve(branches.size());
  bool needsFallbackBranch = true;
  for (auto &branch : branches) {
    if (branch.isCatchAll()) {
      needsFallbackBranch = false;
    }
    Block *block = builder.createBlock(region);
    blocks.push_back(block);
  }

  // Create fallback block, if needed, after all other branches, so
  // that after all other conditions have been tried, we branch to an
  // unreachable to force a trap
  Block *failed = nullptr;
  if (needsFallbackBranch) {
    failed = builder.createBlock(region);
    builder.create<eir::UnreachableOp>(loc);
  }

  // Restore our original insertion point
  builder.restoreInsertionPoint(startIp);

  // Unconditionally branch to the first pattern
  builder.create<BranchOp>(loc, blocks[0]);

  // Save the current insertion point, which we'll restore when lowering is complete
  auto finalIp = builder.saveInsertionPoint();

  // For each branch, populate its block with the predicate and
  // appropriate conditional branching instruction to either jump
  // to the success block, or to the next branches' block (or in
  // the case of the last branch, the 'failed' block)
  auto numBranches = branches.size();
  for (unsigned i = 0; i < numBranches; i++) {
    auto &b = branches[i];
    bool isLast = i == numBranches - 1;
    Block *block = blocks[i];

    // Set our insertion point to the end of the pattern block
    builder.setInsertionPointToEnd(block);

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
        assert(nextPatternBlock != nullptr && "last match block must end in unconditional branch");
        auto cip = builder.saveInsertionPoint();
        // 1. Split block, and conditionally branch to split if is_cons, otherwise the next pattern
        Block *split = builder.createBlock(region, Region::iterator(nextPatternBlock));
        builder.restoreInsertionPoint(cip);
        auto consType = builder.getType<ConsType>();
        auto boxedConsType = builder.getType<BoxType>(consType);
        auto isConsOp = builder.create<IsTypeOp>(loc, selector, boxedConsType);
        auto isConsCond = isConsOp.getResult();
        ArrayRef<Value> emptyArgs{};
        auto ifOp = builder.create<CondBranchOp>(loc, isConsCond, split, emptyArgs, nextPatternBlock, emptyArgs);
        // 2. In the split, extract head and tail values of the cons cell
        builder.setInsertionPointToEnd(split);
        auto termType = builder.getType<TermType>();
        auto castOp = builder.create<CastOp>(loc, selector, boxedConsType);
        auto boxedCons = castOp.getResult();
        auto getHeadOp = builder.create<GetElementPtrOp>(loc, boxedCons, 0);
        auto getTailOp = builder.create<GetElementPtrOp>(loc, boxedCons, 1);
        auto headPointer = getHeadOp.getResult();
        auto tailPointer = getTailOp.getResult();
        auto headLoadOp = builder.create<LoadOp>(loc, headPointer);
        auto tailLoadOp = builder.create<LoadOp>(loc, tailPointer);
        // 3. Unconditionally branch to the destination, with head/tail as additional destArgs
        SmallVector<Value, 2> destArgs;
        destArgs.push_back(headLoadOp.getResult());
        destArgs.push_back(tailLoadOp.getResult());
        builder.create<BranchOp>(loc, dest, destArgs);
        break;
      }

      case MatchPatternType::Tuple: {
        assert(nextPatternBlock != nullptr && "last match block must end in unconditional branch");
        // 1. Split block, and conditionally branch to split if is_tuple w/arity N, otherwise the next pattern
        auto cip = builder.saveInsertionPoint();
        Block *split = builder.createBlock(region, Region::iterator(nextPatternBlock));
        builder.restoreInsertionPoint(cip);
        auto *pattern = b.getPatternTypeOrNull<TuplePattern>();
        auto arity = pattern->getArity();
        auto tupleType = builder.getType<eir::TupleType>(arity);
        auto isTupleOp = builder.create<IsTypeOp>(loc, selector, tupleType);
        auto isTupleCond = isTupleOp.getResult();
        ArrayRef<Value> emptyArgs{};
        auto ifOp = builder.create<CondBranchOp>(loc, isTupleCond, split, emptyArgs, nextPatternBlock, emptyArgs);
        // 2. In the split, extract the tuple elements as values
        builder.setInsertionPointToEnd(split);
        auto termType = builder.getType<TermType>();
        auto boxedTupleType = builder.getType<BoxType>(tupleType);
        auto castOp = builder.create<CastOp>(loc, selector, boxedTupleType);
        auto boxedTuple = castOp.getResult();
        SmallVector<Value, 3> destArgs(baseDestArgs.begin(), baseDestArgs.end());
        destArgs.reserve(arity);
        for (int64_t i = 0; i < arity; i++) {
          auto index = builder.create<ConstantIntOp>(loc, i);
          auto getElementOp = builder.create<GetElementPtrOp>(loc, boxedTuple, i);
          auto elementPtr = getElementOp.getResult();
          auto elementLoadOp = builder.create<LoadOp>(loc, elementPtr);
          destArgs.push_back(elementLoadOp.getResult());
        }
        // 3. Unconditionally branch to the destination, with the tuple elements as additional destArgs
        builder.create<BranchOp>(loc, dest, destArgs);
        break;
      }
        
      case MatchPatternType::MapItem: {
        assert(nextPatternBlock != nullptr && "last match block must end in unconditional branch");
        // 1. Split block twice, and conditionally branch to the first split if is_map, otherwise the next pattern
        auto cip = builder.saveInsertionPoint();
        Block *split2 = builder.createBlock(region, Region::iterator(nextPatternBlock));
        Block *split = builder.createBlock(region, Region::iterator(split2));
        builder.restoreInsertionPoint(cip);
        auto *pattern = b.getPatternTypeOrNull<MapPattern>();
        auto key = pattern->getKey();
        auto mapType = builder.getType<MapType>();
        auto isMapOp = builder.create<IsTypeOp>(loc, selector, mapType);
        auto isMapCond = isMapOp.getResult();
        ArrayRef<Value> emptyArgs{};
        auto ifOp = builder.create<CondBranchOp>(loc, isMapCond, split, emptyArgs, nextPatternBlock, emptyArgs);
        // 2. In the split, call runtime function `is_map_key` to confirm existence of the key in the map,
        //    then conditionally branch to the second split if successful, otherwise the next pattern
        builder.setInsertionPointToEnd(split);
        auto termType = builder.getType<TermType>();
        ArrayRef<Type> getKeyResultTypes = {termType};
        ArrayRef<Value> getKeyArgs = {key, selector};
        auto hasKeyOp = builder.create<CallOp>(loc, "erlang::is_map_key/2", getKeyResultTypes, getKeyArgs);
        auto hasKeyCondTerm = hasKeyOp.getResult(0);
        auto toBoolOp = builder.create<CastOp>(loc, hasKeyCondTerm, builder.getType<BooleanType>());
        auto hasKeyCond = toBoolOp.getResult();
        builder.create<CondBranchOp>(loc, hasKeyCond, split2, emptyArgs, nextPatternBlock, emptyArgs);
        // 3. In the second split, call runtime function `map_get` to obtain the value for the key
        builder.setInsertionPointToEnd(split2);
        ArrayRef<Type> mapGetResultTypes = {termType};
        ArrayRef<Value> mapGetArgs = {key, selector};
        auto mapGetOp = builder.create<CallOp>(loc, "erlang::map_get/2", mapGetResultTypes, mapGetArgs);
        auto valueTerm = mapGetOp.getResult(0);
        // 4. Unconditionally branch to the destination, with the key's value as an additional destArg
        SmallVector<Value, 2> destArgs(baseDestArgs.begin(), baseDestArgs.end());
        destArgs.push_back(valueTerm);
        builder.create<BranchOp>(loc, dest, destArgs);
        break;
      }

      case MatchPatternType::IsType: {
        assert(nextPatternBlock != nullptr && "last match block must end in unconditional branch");
        // 1. Conditionally branch to destination if is_<type>, otherwise the next pattern
        auto *pattern = b.getPatternTypeOrNull<IsTypePattern>();
        auto expectedType = pattern->getExpectedType();
        auto isTypeOp = builder.create<IsTypeOp>(loc, selector, expectedType);
        auto isTypeCond = isTypeOp.getResult();
        ArrayRef<Value> emptyArgs{};
        builder.create<CondBranchOp>(loc, isTypeCond, dest, baseDestArgs, nextPatternBlock, emptyArgs);
        break;
      }

      case MatchPatternType::Value: {
        assert(nextPatternBlock != nullptr && "last match block must end in unconditional branch");
        // 1. Conditionally branch to dest if the value matches the selector,
        //    passing the value as an additional destArg
        auto *pattern = b.getPatternTypeOrNull<ValuePattern>();
        auto expected = pattern->getValue();
        auto isEq = builder.create<CmpEqOp>(loc, selector, expected, /*strict=*/true);
        auto isEqCond = isEq.getResult();
        ArrayRef<Value> emptyArgs{};
        builder.create<CondBranchOp>(loc, isEqCond, dest, baseDestArgs, nextPatternBlock, emptyArgs);
        break;
      }

      case MatchPatternType::Binary: {
        // 1. Split block, and conditionally branch to split if is_bitstring (or is_binary), otherwise the next pattern
        // 2. In the split, conditionally branch to destination if construction of the head value succeeds,
        //    otherwise the next pattern
        // NOTE: The exact semantics depend on the binary specification type, and what is optimal in terms of checks.
        // The success of the overall branch results in two additional destArgs being passed to the destination block,
        // the decoded entry (head), and the rest of the binary (tail)
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

ParseResult parseTraceCaptureOp(OpAsmParser &parser,
                                OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
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

ParseResult parseTraceConstructOp(OpAsmParser &parser,
                                  OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

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

  if (op.getAttrs().size() > 1)
    p << ' ';
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

  BoxType type = op.getType();
  p.printOperands(op.getOperands());
  p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{"map"});
  p << " : " << type;
}

static ParseResult parseMallocOp(OpAsmParser &parser, OperationState &result) {
  BoxType type;

  llvm::SMLoc loc = parser.getCurrentLocation();
  SmallVector<OpAsmParser::OperandType, 1> opInfo;
  SmallVector<Type, 1> types;
  if (parser.parseOperandList(opInfo))
    return failure();
  if (!opInfo.empty() && parser.parseColonTypeList(types))
    return failure();

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
  if (!type)
    return op.emitOpError("result must be a box type");

  OpaqueTermType boxedType = type.getBoxedType();
  if (!boxedType.isBoxable())
    return op.emitOpError("boxed type must be a boxable type");

  auto operands = op.getOperands();
  if (operands.empty()) {
    // There should be no dynamic dimensions in the boxed type
    if (boxedType.isTuple()) {
      auto tuple = boxedType.cast<TupleType>();
      if (tuple.hasDynamicShape())
        return op.emitOpError("boxed type has dynamic extent, but no dimension operands were given");
    }
  } else {
    // There should be exactly as many dynamic dimensions as there are operands,
    // and those operands should be of integer type
    if (!boxedType.isTuple())
      return op.emitOpError("only tuples are allowed to have dynamic dimensions");
    auto tuple = boxedType.cast<TupleType>();
    if (tuple.hasStaticShape())
      return op.emitOpError("boxed type has static extent, but dimension operands were given");
    if (tuple.getArity() != operands.size())
      return op.emitOpError("number of dimension operands does not match the number of dynamic dimensions");
  }

  return success();
}

/// Matches a ConstantIntOp or mlir::ConstantIndexOp.
static mlir::detail::op_matcher<ConstantIntOp> m_ConstantDimension() {
  return mlir::detail::op_matcher<ConstantIntOp>();
}

namespace {
/// Fold constant dimensions into an alloc operation.
struct SimplifyMallocConst : public OpRewritePattern<MallocOp> {
  using OpRewritePattern<MallocOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(MallocOp alloc,
                                     PatternRewriter &rewriter) const override {
    auto boxType = alloc.getType();
    auto boxedType = boxType.getBoxedType();
    if (!boxedType.isTuple())
      return matchFailure();

    auto tuple = boxedType.cast<TupleType>();
    auto numOperands = alloc.getNumOperands();
    if (tuple.hasStaticShape()) {
      // We can remove any operands as the shape is known
      if (numOperands > 0) {
        auto alignment = alloc.alignment();
        if (alignment) {
          auto alignTy = rewriter.getIntegerType(64);
          auto align = rewriter.getIntegerAttr(alignTy, alignment.getValue());
          rewriter.replaceOpWithNewOp<MallocOp>(alloc, boxType, align);
        } else {
          rewriter.replaceOpWithNewOp<MallocOp>(alloc, boxType);
        }
        return matchSuccess();
      } else {
        return matchFailure();
      }
    }

    // Check to see if any dimensions operands are constants.  If so, we can
    // substitute and drop them.
    if (llvm::none_of(alloc.getOperands(), [](Value operand) {
          return matchPattern(operand, m_ConstantDimension());
        }))
      return matchFailure();

    assert(numOperands > 1 && "malloc op only permits one level of dynamic dimensionality");
    SmallVector<Value, 1> newOperands;

    auto *defOp = alloc.getOperand(0).getDefiningOp();
    auto constantIntOp = cast<ConstantIntOp>(defOp);
    auto arityAP = constantIntOp.getValue().cast<IntegerAttr>().getValue();
    auto arity = (unsigned)arityAP.getLimitedValue();
    auto newTuple = TupleType::get(tuple.getContext(), arity);
    auto newType = BoxType::get(newTuple);
    auto alignment = alloc.alignment();

    if (alignment) {
      auto alignTy = rewriter.getIntegerType(64);
      auto align = rewriter.getIntegerAttr(alignTy, alignment.getValue());
      rewriter.replaceOpWithNewOp<MallocOp>(alloc, newType, align);
    } else {
      rewriter.replaceOpWithNewOp<MallocOp>(alloc, newType);
    }
    return matchSuccess();
  }
};

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
} // end anonymous namespace.

void MallocOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                           MLIRContext *context) {
  results.insert<SimplifyMallocConst, SimplifyDeadMalloc>(context);
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

} // namespace eir
} // namespace lumen
