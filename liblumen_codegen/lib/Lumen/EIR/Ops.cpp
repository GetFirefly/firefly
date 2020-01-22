#include "lumen/LLVM.h"

#include "eir/Ops.h"
#include "eir/Types.h"
#include "eir/Attributes.h"

#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"

#include "llvm/Support/Casting.h"

#include <iterator>
#include <vector>

using namespace eir;

namespace L = llvm;
namespace M = mlir;

using L::SmallVector;
using OperandType = M::OpAsmParser::OperandType;
using OperandIterator = M::Operation::operand_iterator;
using Delimiter = M::OpAsmParser::Delimiter;

template <typename Iterator, typename F, typename K>
void chunked(Iterator begin, Iterator end, K k, F f) {
  Iterator chunk_begin = begin;
  Iterator chunk_end = begin;

  do {
    if (std::distance(chunk_end, end) < k) {
      chunk_end = end;
    } else {
      std::advance(chunk_end, k);
    }
    f(chunk_begin, chunk_end);
    chunk_begin = chunk_end;
  } while (std::distance(chunk_begin, end) > 0);
}

namespace eir {

// Parses ': (ty1, ..tyN) ->'
M::ParseResult parseColonTypeArgumentList(M::OpAsmParser &parser, SmallVectorImpl<M::Type> &types) {
  if (parser.parseColon() || parser.parseLParen())
    return M::failure();

  M::Type first;
  if (parser.parseType(first))
    return M::failure();
  types.push_back(first);
  
  do {
    // Parse leading comma, or ')' if no comma is present
    if (parser.parseComma()) {
      // Check if this is the end of the arg list
      if (parser.parseRParen()) {
        parser.emitError(parser.getCurrentLocation(), "expected ')'");
        return M::failure();
      } else {
        // We're done with the argument list
        break;
      }
    }
    // Parse type
    M::Type ty;
    auto loc = parser.getCurrentLocation();
    if (parser.parseType(ty)) {
      parser.emitError(loc, "expected type");
      return M::failure();
    } else {
      types.push_back(ty);
    }
  } while(true);

  // Parse -> 
  if (parser.parseArrow())
    return M::failure();

  return M::success();
}

//===----------------------------------------------------------------------===//
// ConsOp
//===----------------------------------------------------------------------===//

void ConsOp::build(M::Builder *builder, M::OperationState &result, M::Value head, M::Value tail) {
  result.addOperands(head);
  result.addOperands(tail);
  result.addTypes(builder->getType<ConsType>());
}

static M::ParseResult parseConsOp(M::OpAsmParser &parser,
                                  M::OperationState &result) {
  SmallVector<M::OpAsmParser::OperandType, 2> consInfo;
  SmallVector<M::Type, 2> typeInfo;
  M::Type resultType;
  L::SMLoc loc = parser.getCurrentLocation();
  return M::failure(parser.parseOperandList(consInfo, /*required_num*/2, M::OpAsmParser::Delimiter::Paren) ||
                    (!consInfo.empty() && parseColonTypeArgumentList(parser, typeInfo)) ||
                    parser.resolveOperands(consInfo, typeInfo, loc, result.operands) ||
                    parser.parseType(resultType) ||
                    parser.addTypeToList(resultType, result.types) ||
                    parser.parseOptionalAttrDict(result.attributes));
}
  
static void print(M::OpAsmPrinter &p, ConsOp op) {
  p << ConsOp::getOperationName() << "(";
  p << op.getOperands() << ") : (";
  p << op.getOperandTypes() << ") -> ";
  p << op.getResult().getType();

  p.printOptionalAttrDict(op.getAttrs());
}

static M::LogicalResult verify(ConsOp op) {
  // TODO
  return M::success();
}
 
//===----------------------------------------------------------------------===//
// TupleOp
//===----------------------------------------------------------------------===//

void TupleOp::build(M::Builder *builder, M::OperationState &result, ArrayRef<M::Value> elements) {
  auto shape = Shape::infer(elements);
  result.addOperands(elements);
  result.addTypes(builder->getType<eir::TupleType>(shape));
}

static M::ParseResult parseTupleOp(M::OpAsmParser &parser,
                                   M::OperationState &result) {
  SmallVector<M::OpAsmParser::OperandType, 2> elementsInfo;
  SmallVector<M::Type, 2> elementsTypeInfo;
  M::Type resultType;
  L::SMLoc loc = parser.getCurrentLocation();
  
  return M::failure(parser.parseOperandList(elementsInfo, /*required_num*/1, M::OpAsmParser::Delimiter::Paren) ||
                    (!elementsInfo.empty() && parseColonTypeArgumentList(parser, elementsTypeInfo)) ||
                    parser.resolveOperands(elementsInfo, elementsTypeInfo, loc, result.operands) ||
                    parser.parseType(resultType) ||
                    parser.addTypeToList(resultType, result.types) ||
                    parser.parseOptionalAttrDict(result.attributes));
}
  
static void print(M::OpAsmPrinter &p, TupleOp op) {
  p << TupleOp::getOperationName() << " ";
  p << "(" << op.getOperands() << ") : (";
  p << op.getOperandTypes() << ") -> ";
  p << op.getResult().getType();

  p.printOptionalAttrDict(op.getAttrs());
}

static M::LogicalResult verify(TupleOp op) {
  // TODO
  return M::success();
}

//===----------------------------------------------------------------------===//
// ConstructMapOp
//===----------------------------------------------------------------------===//

void ConstructMapOp::build(M::Builder *builder, M::OperationState &result, ArrayRef<eir::MapEntry> entries) {
  for (auto &entry : entries) {
    result.addOperands(M::Value::getFromOpaquePointer(entry.key));
    result.addOperands(M::Value::getFromOpaquePointer(entry.value));
  }
  result.addTypes(builder->getType<MapType>());
}

static M::ParseResult parseConstructMapOp(M::OpAsmParser &parser,
                                          M::OperationState &result) {
  SmallVector<M::OpAsmParser::OperandType, 2> entriesInfo;
  SmallVector<M::Type, 2> entriesTypeInfo;
  M::Type resultType;
  L::SMLoc loc = parser.getCurrentLocation();
  
  return M::failure(parser.parseOperandList(entriesInfo, /*required_num*/-1, M::OpAsmParser::Delimiter::Paren) ||
                    parseColonTypeArgumentList(parser, entriesTypeInfo) ||
                    parser.resolveOperands(entriesInfo, entriesTypeInfo, loc, result.operands) ||
                    parser.parseType(resultType) ||
                    parser.addTypeToList(resultType, result.types) ||
                    parser.parseOptionalAttrDict(result.attributes));
}
  
static void print(M::OpAsmPrinter &p, ConstructMapOp op) {
  p << ConstructMapOp::getOperationName() << " ";
  p << "(" << op.getOperands() << ") : (";
  p << op.getOperandTypes() << ") -> ";
  p << op.getResultTypes();

  p.printOptionalAttrDict(op.getAttrs());
}

static M::LogicalResult verify(ConstructMapOp op) {
  // TODO
  return M::success();
}
   
//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

void IfOp::build(M::Builder *builder, M::OperationState &result, M::Value cond,
                 bool withOtherwiseRegion) {
  result.addOperands(cond);

  M::Region *ifRegion = result.addRegion();
  M::Region *elseRegion = result.addRegion();
  M::Region *otherwiseRegion = result.addRegion();

  M::OpBuilder opBuilder(builder->getContext());

  M::Block *ifEntry = opBuilder.createBlock(ifRegion);
  M::Block *elseEntry = opBuilder.createBlock(elseRegion);
  M::Block *otherwiseEntry = opBuilder.createBlock(otherwiseRegion);
  if (!withOtherwiseRegion) {
    opBuilder.create<eir::UnreachableOp>(result.location);
  }
}

static M::ParseResult parseIfOp(M::OpAsmParser &parser,
                                M::OperationState &result) {
  // Create the regions
  result.regions.reserve(3);
  M::Region *ifRegion = result.addRegion();
  M::Region *elseRegion = result.addRegion();
  M::Region *otherwiseRegion = result.addRegion();

  auto &builder = parser.getBuilder();
  M::OpAsmParser::OperandType cond;
  M::Type i1Type = builder.getIntegerType(1);
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

static void print(M::OpAsmPrinter &p, IfOp op) {
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

static M::LogicalResult verify(IfOp op) {
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
      M::Operation &terminator = b.back();
      if (isa<M::ReturnOp>(terminator))
        continue;
      else if (isa<M::BranchOp>(terminator))
        continue;
      else if (isa<M::CondBranchOp>(terminator))
        continue;
      else if (isa<M::CallOp>(terminator))
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

  return M::success();
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

M::ParseResult parseCastOp(M::OpAsmParser &parser, M::OperationState &result) {
  M::OpAsmParser::OperandType srcInfo;
  M::Type srcType, dstType;
  return M::failure(parser.parseOperand(srcInfo) ||
                    parser.parseOptionalAttrDict(result.attributes) ||
                    parser.parseColonType(srcType) ||
                    parser.resolveOperand(srcInfo, srcType, result.operands) ||
                    parser.parseKeywordType("to", dstType) ||
                    parser.addTypeToList(dstType, result.types));
}

static void print(M::OpAsmPrinter &p, CastOp op) {
  p << op.getOperationName() << ' ';
  p << op.getOperand() << " : " << op.getInputType();
  p << " to ";
  p << op.getResultType();
}
  
static M::LogicalResult verify(CastOp op) {
  auto inType = op.getInputType();
  auto outType = op.getResultType();
  if (!CastOp::areCastCompatible(inType, outType))
    return op.emitError("operand type ") << inType << " cannot be cast to "
                                         << outType;

  return M::success();
}

bool CastOp::areCastCompatible(M::Type inType, M::Type outType) {
  // Input terms must be a generic term, an immediate or boxed terms
  if (auto inTerm = inType.dyn_cast_or_null<TermBase>()) {
    auto kind = inTerm.getImplKind();
    // Terms can be cast to any type for now, we may restrict this in the future
    if (kind == EirTypes::Term)
      return true;

    // Fixnums can be cast to integers with a bitmask + bitcast
    if (kind == EirTypes::Fixnum && outType.isa<M::IntegerType>())
      return true;
    else
      return false;

    // Booleans can be cast to i1 with a bitmask + bitcast
    if (kind == EirTypes::Boolean && outType.isInteger(1))
      return true;
    else
      return false;

    // Floats (non-packed) can be cast to f64 directly
    if (kind == EirTypes::Float && outType.isF64())
      return true;
    else
      return false;

    // Boxed terms can be cast to pointers to their element types with a bitmask + bitcast
    if (auto boxType = inType.dyn_cast_or_null<BoxType>()) {
      if (outType.getKind() == boxType.getEleTy().getKind())
        return true;
      return false;
    }

    return false;
  }
  // Raw boolean values can be cast to boolean term with a bitwise-OR
  if (inType.isInteger(1) && outType.isa<BooleanType>())
    return true;

  // Raw integer values can be cast to fixnum term with bit operations
  if (inType.isa<M::IntegerType>() && outType.isa<FixnumType>())
    return true;

  // Raw float values can be cast to float (non-packed) terms with a bitcast
  if (inType.isF64() && outType.isa<FloatType>())
    return true;

  return false;
}
  
//===----------------------------------------------------------------------===//
// GetElementOp
//===----------------------------------------------------------------------===//

M::ParseResult parseGetElementOp(M::OpAsmParser &parser, M::OperationState &result) {
  M::OpAsmParser::OperandType srcInfo;
  M::Type srcType, dstType;

  // Parse aggregate operand and type
  if (parser.parseOperand(srcInfo) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(srcType) ||
      parser.resolveOperand(srcInfo, srcType, result.operands))
    return M::failure();

  // Parse " at index "
  if (parser.parseKeyword("at") || parser.parseKeyword("index"))
    return M::failure();

  // Parse index value
  M::IntegerAttr index;
  M::Type indexType = parser.getBuilder().getIndexType();
  if (parser.parseAttribute(index, indexType, "index", result.attributes))
    return M::failure();

  // Parse result type
  if (parser.parseColonType(dstType) || parser.addTypeToList(dstType, result.types))
    return M::failure();

  return M::success();
}

static void print(M::OpAsmPrinter &p, GetElementOp op) {
  p << op.getOperationName() << ' ';
  p << op.getOperand() << " : ";
  p << op.getAggregateType();
  p << " at index " << op.getIndex();
  p << " : " << op.getResultType();
}
  
static M::LogicalResult verify(GetElementOp op) {
  auto aggregateType = op.getAggregateType();
  if (!aggregateType.isa<ConsType>() && !aggregateType.isa<TupleType>())
    return op.emitOpError("input type to get_element must be a cons or tuple type");

  auto resultType = op.getResultType();
  if (auto consType = aggregateType.dyn_cast_or_null<ConsType>()) {
    if (!resultType.isa<TermType>())
      return op.emitOpError("result type for get_element on cons cells must be term");
    return M::success();
  }

  auto tupleType = aggregateType.cast<TupleType>();
  auto index = op.getIndex();
  auto shape = tupleType.getShape();
  auto arity = shape.arity();
  if (shape.isKnown()) {
    if (index >= arity)
      return op.emitOpError("out of bounds index used in get_element with tuple of arity ") << arity;

    auto elementTypeOpt = shape.getType(index);
    auto elementType = elementTypeOpt.getValue();
    if (resultType.getKind() != elementType.getKind())
      return op.emitOpError("result type does not match type of element at index ") << index << ", expected " << elementType;
  } else {
    if (!resultType.isa<TermType>())
      return op.emitOpError("result type for dynamically shaped tuples must be term");
  }
  return M::success();
}
  
//===----------------------------------------------------------------------===//
// IsTypeOp
//===----------------------------------------------------------------------===//

M::ParseResult parseIsTypeOp(M::OpAsmParser &parser, M::OperationState &result) {
  M::OpAsmParser::OperandType op;
  L::SmallVector<NamedAttribute, 1> attrs;
  Type valueType;
  Type resultType;
  if (parser.parseOperand(op) ||
      parser.parseColonType(valueType) ||
      parser.parseOptionalAttrDict(attrs) ||
      parser.parseColonType(resultType))
    return M::failure();

  if (!resultType.isa<BooleanType>() && !resultType.isInteger(1)) {
    return parser.emitError(parser.getNameLoc(),
                            "expected result to be valid boolean type, either i1 or !eir.boolean");
  }

  if (parser.resolveOperand(op, valueType, result.operands) ||
      parser.addTypeToList(resultType, result.types))
    return M::failure();

  return M::success();
}

static void print(M::OpAsmPrinter &p, IsTypeOp op) {
  p << op.getOperationName() << ' ';
  p << op.getOperand();
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.getResultType();
}

static M::LogicalResult verify(IsTypeOp op) {
  auto typeAttr = op.getAttrOfType<M::TypeAttr>("type");
  if (!typeAttr)
    return op.emitOpError("requires type attribute named 'type'");

  auto resultType = op.getResultType();
  if (!resultType.isa<BooleanType>() && !resultType.isInteger(1)) {
    return op.emitOpError("requires result type to be of type i1 or !eir.boolean");
  }

  return M::success();
}

//===----------------------------------------------------------------------===//
// MatchOp
//===----------------------------------------------------------------------===//

void MatchOp::build(M::Builder *builder, M::OperationState &result,
                    M::Value selector,
                    ArrayRef<MatchBranch> branches,
                    ArrayRef<NamedAttribute> attributes) {

  M::OpBuilder opBuilder(builder->getContext());
  assert(branches.size() > 0 && "expected at least one branch in a match");

  // We only have one "real" operand, the selector
  result.addOperands(selector);

  // Create the region which holds the match blocks
  M::Region *body = result.addRegion();

  // Create blocks for all branches
  L::SmallVector<M::Block *, 2> blocks;
  blocks.reserve(branches.size());
  bool needsFallbackBranch = true;
  for (auto it = branches.begin(); it + 1 != branches.end(); ++it) {
    if (it->isCatchAll()) {
      needsFallbackBranch = false;
    }
    M::Block *block = opBuilder.createBlock(body);
    blocks.push_back(block);
  }

  // Create fallback block, if needed, after all other branches, so
  // that after all other conditions have been tried, we branch to an
  // unreachable to force a trap
  M::Block *failed = nullptr;
  if (needsFallbackBranch) {
    failed = opBuilder.createBlock(body);
    opBuilder.create<eir::UnreachableOp>(result.location);
  }

  // For each branch, populate its block with the predicate and
  // appropriate conditional branching instruction to either jump
  // to the success block, or to the next branches' block (or in
  // the case of the last branch, the 'failed' block)
  for (auto b = branches.begin(); b + 1 != branches.end(); ++b) {
    bool isLast = b == branches.end();
    M::Block *block = opBuilder.createBlock(body);
    M::OpBuilder blockBuilder(block);

    // Store the next pattern to try if this one fails
    // If this is the last pattern, we validate that the
    // branch either unconditionally succeeds, or branches to
    // an unreachable op
    M::Block *nextPatternBlock = nullptr;
    if (!isLast) {
      auto nextBranch = b + 1;
      nextPatternBlock = nextBranch->getDest();
    }

    auto dest = b->getDest();
    auto baseDestArgs = b->getDestArgs();
    
    switch (b->getPatternType()) {
      case MatchPatternType::Any:
        // This unconditionally branches to its destination
        result.addSuccessor(dest, baseDestArgs);
        blockBuilder.create<M::BranchOp>(result.location, dest, baseDestArgs);
        break;

      case MatchPatternType::Cons: {
        assert(nextPatternBlock != nullptr && "last match block must end in unconditional branch");
        // 1. Split block, and conditionally branch to split if is_cons, otherwise the next pattern
        M::Block *split = opBuilder.createBlock(nextPatternBlock);
        auto consType = opBuilder.getType<ConsType>();
        auto isConsOp = blockBuilder.create<IsTypeOp>(result.location, selector, consType);
        auto isConsCond = isConsOp.getResult();
        ArrayRef<M::Value> emptyArgs{};
        auto ifOp = blockBuilder.create<M::CondBranchOp>(result.location, isConsCond, split, emptyArgs, nextPatternBlock, emptyArgs);
        // 2. In the split, extract head and tail values of the cons cell
        M::OpBuilder splitBuilder(split);
        auto boxedConsType = splitBuilder.getType<BoxType>(consType);
        auto castOp = splitBuilder.create<CastOp>(result.location, selector, boxedConsType);
        auto boxedCons = castOp.getResult();
        auto getHeadOp = splitBuilder.create<GetElementOp>(result.location, boxedCons, 0);
        auto getTailOp = splitBuilder.create<GetElementOp>(result.location, boxedCons, 1);
        auto headPointer = getHeadOp.getResult();
        auto tailPointer = getTailOp.getResult();
        auto headLoadOp = splitBuilder.create<LoadOp>(result.location, headPointer);
        auto tailLoadOp = splitBuilder.create<LoadOp>(result.location, tailPointer);
        // 3. Unconditionally branch to the destination, with head/tail as additional destArgs
        L::SmallVector<M::Value, 3> destArgs(baseDestArgs.begin(), baseDestArgs.end());
        destArgs.push_back(headLoadOp.getResult());
        destArgs.push_back(tailLoadOp.getResult());
        result.addSuccessor(dest, destArgs);
        splitBuilder.create<M::BranchOp>(result.location, dest, destArgs);
        break;
      }

      case MatchPatternType::Tuple: {
        assert(nextPatternBlock != nullptr && "last match block must end in unconditional branch");
        // 1. Split block, and conditionally branch to split if is_tuple w/arity N, otherwise the next pattern
        M::Block *split = opBuilder.createBlock(nextPatternBlock);
        auto *pattern = b->getPatternTypeOrNull<TuplePattern>();
        auto arity = pattern->getArity();
        Shape shape(opBuilder.getType<TermType>(), arity);
        auto tupleType = opBuilder.getType<TupleType>(shape);
        auto isTupleOp = blockBuilder.create<IsTypeOp>(result.location, selector, tupleType);
        auto isTupleCond = isTupleOp.getResult();
        ArrayRef<M::Value> emptyArgs{};
        auto ifOp = blockBuilder.create<M::CondBranchOp>(result.location, isTupleCond, split, emptyArgs, nextPatternBlock, emptyArgs);
        // 2. In the split, extract the tuple elements as values
        M::OpBuilder splitBuilder(split);
        auto boxedTupleType = splitBuilder.getType<BoxType>(tupleType);
        auto castOp = splitBuilder.create<CastOp>(result.location, selector, boxedTupleType);
        auto boxedTuple = castOp.getResult();
        L::SmallVector<M::Value, 3> destArgs(baseDestArgs.begin(), baseDestArgs.end());
        destArgs.reserve(arity);
        for (unsigned i = 0; i + 1 != arity; ++i) {
          auto getElementOp = splitBuilder.create<GetElementOp>(result.location, boxedTuple, 0);
          auto elementPtr = getElementOp.getResult();
          auto elementLoadOp = splitBuilder.create<LoadOp>(result.location, elementPtr);
          destArgs.push_back(elementLoadOp.getResult());
        }
        // 3. Unconditionally branch to the destination, with the tuple elements as additional destArgs
        result.addSuccessor(dest, destArgs);
        splitBuilder.create<M::BranchOp>(result.location, dest, destArgs);
        break;
      }
        
      case MatchPatternType::MapItem: {
        assert(nextPatternBlock != nullptr && "last match block must end in unconditional branch");
        // 1. Split block twice, and conditionally branch to the first split if is_map, otherwise the next pattern
        M::Block *split2 = opBuilder.createBlock(nextPatternBlock);
        M::Block *split = opBuilder.createBlock(split2);
        auto *pattern = b->getPatternTypeOrNull<MapPattern>();
        auto key = pattern->getKey();
        auto mapType = opBuilder.getType<MapType>();
        auto isMapOp = blockBuilder.create<IsTypeOp>(result.location, selector, mapType);
        auto isMapCond = isMapOp.getResult();
        ArrayRef<M::Value> emptyArgs{};
        auto ifOp = blockBuilder.create<M::CondBranchOp>(result.location, isMapCond, split, emptyArgs, nextPatternBlock, emptyArgs);
        // 2. In the split, call runtime function `is_map_key` to confirm existence of the key in the map,
        //    then conditionally branch to the second split if successful, otherwise the next pattern
        M::OpBuilder splitBuilder(split);
        auto termType = splitBuilder.getType<TermType>();
        ArrayRef<M::Type> getKeyResultTypes = {termType};
        ArrayRef<M::Value> getKeyArgs = {key, selector};
        auto hasKeyOp = splitBuilder.create<M::CallOp>(result.location, "erlang_is_map_key_2", getKeyResultTypes, getKeyArgs);
        auto hasKeyCondTerm = hasKeyOp.getResult(0);
        auto toBoolOp = splitBuilder.create<CastOp>(result.location, hasKeyCondTerm, opBuilder.getI1Type());
        auto hasKeyCond = toBoolOp.getResult();
        blockBuilder.create<M::CondBranchOp>(result.location, hasKeyCond, split2, emptyArgs, nextPatternBlock, emptyArgs);
        // 3. In the second split, call runtime function `map_get` to obtain the value for the key
        M::OpBuilder split2Builder(split2);
        ArrayRef<M::Type> mapGetResultTypes = {termType};
        ArrayRef<M::Value> mapGetArgs = {key, selector};
        auto mapGetOp = split2Builder.create<M::CallOp>(result.location, "erlang_map_get_2", mapGetResultTypes, mapGetArgs);
        auto valueTerm = mapGetOp.getResult(0);
        // 4. Unconditionally branch to the destination, with the key's value as an additional destArg
        L::SmallVector<M::Value, 2> destArgs(baseDestArgs.begin(), baseDestArgs.end());
        destArgs.push_back(valueTerm);
        result.addSuccessor(dest, destArgs);
        split2Builder.create<M::BranchOp>(result.location, dest, destArgs);
        break;
      }

      case MatchPatternType::IsType: {
        assert(nextPatternBlock != nullptr && "last match block must end in unconditional branch");
        // 1. Conditionally branch to destination if is_<type>, otherwise the next pattern
        auto *pattern = b->getPatternTypeOrNull<IsTypePattern>();
        auto expectedType = pattern->getExpectedType();
        auto isTypeOp = blockBuilder.create<IsTypeOp>(result.location, selector, expectedType);
        auto isTypeCond = isTypeOp.getResult();
        result.addSuccessor(dest, baseDestArgs);
        ArrayRef<M::Value> emptyArgs{};
        blockBuilder.create<M::CondBranchOp>(result.location, isTypeCond, dest, baseDestArgs, nextPatternBlock, emptyArgs);
        break;
      }

      case MatchPatternType::Value: {
        // 1. Unconditionally branch to destination, passing the value as an additional destArg
        auto *pattern = b->getPatternTypeOrNull<ValuePattern>();
        L::SmallVector<M::Value, 3> destArgs(baseDestArgs.begin(), baseDestArgs.end());
        destArgs.push_back(selector);
        destArgs.push_back(pattern->getValue());
        result.addSuccessor(dest, destArgs);
        blockBuilder.create<M::BranchOp>(result.location, dest, destArgs);
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
        auto *pattern = b->getPatternTypeOrNull<BinaryPattern>();
        break;
      }

      default:
        llvm_unreachable("unexpected match pattern type!");
    }
  }
}

static M::ParseResult parseMatchOp(M::OpAsmParser &parser,
                                   M::OperationState &result) {

  // TODO: Need to properly parse this operation
  return parser.parseRegion(*result.addRegion(),
                            /*arguments=*/{},
                            /*argTypes=*/{});
}

static void print(OpAsmPrinter &p, MatchOp matchOp) {
  auto *op = matchOp.getOperation();

  p << matchOp.getOperationName() << ' ';
  p.printOperand(matchOp.getSelector());
  p << " : " << matchOp.getSelector().getType() << ' ';
  p.printRegion(op->getRegion(0),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}

static M::LogicalResult verify(MatchOp matchOp) {
  auto *op = matchOp.getOperation();

  auto &region = op->getRegion(0);
  // Allow empty region as a degenerate case, which can come from
  // optimizations.
  if (region.empty())
    return success();

  // The last block must end with an unreachable, or unconditional branch
  // to a block outside the region
  auto &lastBlock = region.back();
  auto &lastBlockOp = lastBlock.back();
  if (!isa<UnreachableOp>(lastBlockOp)) {
    if (auto brOp = dyn_cast<M::BranchOp>(lastBlockOp)) {
      auto brParent = brOp.getDest()->getParent();
      if (brParent == &region) {
        return matchOp.emitOpError(
          "invalid branch target in last block: target must be a block outside of the match region");
      }
    } else {
      return matchOp.emitOpError(
        "last block must terminate with 'eir.unreachable', or 'br'");
    }
  }
  
  // Every block must branch or end in 'eir.unreachable'
  for (auto &blk : region) {
    auto &lastOp = blk.back();
    if (isa<UnreachableOp>(lastOp) || isa<IfOp>(lastOp) || isa<M::BranchOp>(lastOp) || isa<M::CondBranchOp>(lastOp)) {
      continue;
    } else {
      return matchOp.emitOpError("all match region blocks must end in a branch, 'eir.if', or 'eir.unreachable'");
    }
  }

  return success();
}

void MatchOp::getCanonicalizationPatterns(M::OwningRewritePatternList &results,
                                          M::MLIRContext *context) {
  // TODO: if statically resolvable, collapse conditions into unconditional
  // branches, and potentially fold out the match altogether
  return;
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

static M::ParseResult parseCallOp(M::OpAsmParser &parser,
                                  M::OperationState &result) {
  L::SmallVector<M::OpAsmParser::OperandType, 8> operands;

  if (parser.parseOperandList(operands))
    return M::failure();
  bool isDirect = operands.empty();
  SmallVector<NamedAttribute, 4> attrs;
  M::SymbolRefAttr funcAttr;

  if (isDirect)
    if (parser.parseAttribute(funcAttr, "callee", attrs))
      return M::failure();
  Type type;

  if (parser.parseOperandList(operands, M::OpAsmParser::Delimiter::Paren) ||
      parser.parseOptionalAttrDict(attrs) || parser.parseColon() ||
      parser.parseType(type))
    return M::failure();

  auto funcType = type.dyn_cast<M::FunctionType>();
  if (!funcType)
    return parser.emitError(parser.getNameLoc(), "expected function type");
  if (isDirect) {
    if (parser.resolveOperands(operands, funcType.getInputs(),
                               parser.getNameLoc(), result.operands))
      return M::failure();
  } else {
    auto funcArgs =
        L::ArrayRef<M::OpAsmParser::OperandType>(operands).drop_front();
    L::SmallVector<M::Value , 8> resultArgs(
        result.operands.begin() + (result.operands.empty() ? 0 : 1),
        result.operands.end());
    if (parser.resolveOperand(operands[0], funcType, result.operands) ||
        parser.resolveOperands(funcArgs, funcType.getInputs(),
                               parser.getNameLoc(), resultArgs))
      return M::failure();
  }
  result.addTypes(funcType.getResults());
  result.attributes = attrs;
  return M::success();
}

static void print(M::OpAsmPrinter &p, CallOp op) {
  auto callee = op.callee();
  bool isDirect = callee.hasValue();
  p << op.getOperationName() << ' ';
  if (isDirect)
    p << callee.getValue();
  else
    p << op.getOperand(0);
  p << '(';
  p.printOperands(L::drop_begin(op.getOperands(), isDirect ? 0 : 1));
  p << ')';
  p.printOptionalAttrDict(op.getAttrs(), {"callee"});
  M::OpBuilder builder(op.getContext());
  L::SmallVector<Type, 1> resultTypes({builder.getType<TermType>()});
  L::SmallVector<Type, 8> argTypes(
      L::drop_begin(op.getOperandTypes(), isDirect ? 0 : 1));
  p << " : " << FunctionType::get(argTypes, resultTypes, op.getContext());
}

static M::LogicalResult verify(CallOp op) {
  auto callee = op.callee();
  if (callee.hasValue()) {
    // Direct, check that the callee attribute was specified
    auto fnAttr = op.getAttrOfType<M::FlatSymbolRefAttr>("callee");
    if (!fnAttr)
      return op.emitOpError("expected a 'callee' symbol reference attribute");
    auto fn = op.getParentOfType<M::ModuleOp>().lookupSymbol<M::FuncOp>(
        fnAttr.getValue());
    if (!fn)
      return op.emitOpError()
             << "'" << fnAttr.getValue()
             << "' does not reference a valid function in this module";

    // Verify that the operand and result types match the callee.
    auto fnType = fn.getType();
    if (fnType.getNumInputs() != op.getNumOperands())
      return op.emitOpError("incorrect number of operands for callee");

    for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i)
      if (op.getOperand(i).getType() != fnType.getInput(i))
        return op.emitOpError("operand type mismatch");

  } else {
    // Indirect
    // The callee must be a function.
    auto fnType = op.callee()->getType().dyn_cast<M::FunctionType>();
    if (!fnType)
      return op.emitOpError("callee must have function type");

    // Verify that the operand and result types match the callee.
    if (fnType.getNumInputs() != op.getNumOperands() - 1)
      return op.emitOpError("incorrect number of operands for callee");

    for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i)
      if (op.getOperand(i + 1).getType() != fnType.getInput(i))
        return op.emitOpError("operand type mismatch");
  }

  return M::success();
}

M::FunctionType CallOp::getCalleeType() {
  SmallVector<M::Type, 1> resultTypes;
  SmallVector<M::Type, 8> argTypes(getOperandTypes());
  return M::FunctionType::get(argTypes, resultTypes, getContext());
}

M::CallInterfaceCallable CallOp::getCallableForCallee() {
  return getAttrOfType<M::SymbolRefAttr>("callee");
}

M::Operation::operand_range CallOp::getArgOperands() { return arguments(); }

//===----------------------------------------------------------------------===//
// UnreachableOp
//===----------------------------------------------------------------------===//

static M::LogicalResult verify(UnreachableOp) {
  // TODO
  return M::success();
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

M::ParseResult parseLoadOp(M::OpAsmParser &parser, M::OperationState &result) {
  M::Type type;
  M::OpAsmParser::OperandType oper;

  if (parser.parseOperand(oper) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type) ||
      parser.resolveOperand(oper, type, result.operands))
    return M::failure();

  M::Type eleTy;
  if (eir::LoadOp::getElementOf(eleTy, type) ||
      parser.addTypeToList(eleTy, result.types))
    return M::failure();
  return M::success();
}

static void print(M::OpAsmPrinter &p, eir::LoadOp op) {
  p << op.getOperationName() << ' ';
  p.printOperand(op.memref());
  p.printOptionalAttrDict(op.getOperation()->getAttrs(), {});
  p << " : " << op.memref().getType();
}

static M::LogicalResult verify(eir::LoadOp) { return M::success(); }

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

M::ParseResult parseStoreOp(M::OpAsmParser &parser, M::OperationState &result) {
  M::Type type;
  M::OpAsmParser::OperandType oper;
  M::OpAsmParser::OperandType store;

  if (parser.parseOperand(oper) || parser.parseKeyword("to") ||
      parser.parseOperand(store) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type) ||
      parser.resolveOperand(oper, StoreOp::elementType(type),
                            result.operands) ||
      parser.resolveOperand(store, type, result.operands))
    return M::failure();
  return M::success();
}

static void print(M::OpAsmPrinter &p, eir::StoreOp op) {
  p << op.getOperationName() << ' ';
  p.printOperand(op.value());
  p << " to ";
  p.printOperand(op.memref());
  p.printOptionalAttrDict(op.getOperation()->getAttrs(), {});
  p << " : " << op.memref().getType();
}

static M::LogicalResult verify(eir::StoreOp) { return M::success(); }

/// Get the element type of a reference like type; otherwise null
M::Type elementTypeOf(M::Type ref) {
  if (auto r = ref.dyn_cast_or_null<RefType>())
    return r.getEleTy();
  if (auto r = ref.dyn_cast_or_null<BoxType>())
    return r.getEleTy();
  return {};
}

M::ParseResult LoadOp::getElementOf(M::Type &ele, M::Type ref) {
  if (M::Type r = elementTypeOf(ref)) {
    ele = r;
    return M::success();
  }
  return M::failure();
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

M::ParseResult parseUndefOp(M::OpAsmParser &parser, M::OperationState &result) {
  M::Type intype;
  if (parser.parseType(intype) || parser.addTypeToList(intype, result.types))
    return M::failure();
  return M::success();
}

static void print(M::OpAsmPrinter &p, UndefOp op) {
  p << op.getOperationName() << ' ' << op.getType();
}

static M::LogicalResult verify(UndefOp op) {
  if (auto ref = op.getType().dyn_cast<eir::RefType>())
    return op.emitOpError("undefined values of type !eir.ref not allowed");
  return M::success();
}

M::Type StoreOp::elementType(M::Type refType) {
  if (auto ref = refType.dyn_cast<RefType>())
    return ref.getEleTy();
  if (auto ref = refType.dyn_cast<BoxType>())
    return ref.getEleTy();
  return {};
}

//===----------------------------------------------------------------------===//
// AllocaOp
//===----------------------------------------------------------------------===//

M::Type AllocaOp::getAllocatedType() {
  return getType().cast<RefType>().getEleTy();
}

/// Create a legal memory reference as return type
M::Type AllocaOp::wrapResultType(M::Type intype) {
  // memory references to memory references are disallowed
  if (intype.dyn_cast<RefType>())
    return {};
  return RefType::get(intype.getContext(), intype);
}

M::Type AllocaOp::getRefTy(M::Type ty) {
  return RefType::get(ty.getContext(), ty);
}

//===----------------------------------------------------------------------===//
// MallocOp
//===----------------------------------------------------------------------===//

static M::LogicalResult verify(MallocOp op) {
  M::Type outType = op.getType();
  if (!outType.dyn_cast<BoxType>())
    return op.emitOpError("must be a !eir.box type");
  return M::success();
}

M::Type MallocOp::getAllocatedType() {
  return getType().cast<BoxType>().getEleTy();
}

M::Type MallocOp::getRefTy(M::Type ty) {
  return BoxType::get(ty.getContext(), ty);
}

/// Create a legal heap reference as return type
M::Type MallocOp::wrapResultType(M::Type intype) {
  // one may not allocate a memory reference value
  if (intype.dyn_cast<RefType>() || intype.dyn_cast<BoxType>() ||
      intype.dyn_cast<FunctionType>())
    return {};
  return BoxType::get(intype.getContext(), intype);
}

//===----------------------------------------------------------------------===//
// TraceCaptureOp
//===----------------------------------------------------------------------===//

M::ParseResult parseTraceCaptureOp(M::OpAsmParser &parser,
                                   M::OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return M::failure();
  return M::success();
}

static void print(M::OpAsmPrinter &p, TraceCaptureOp op) {
  p << op.getOperationName();
  p.printOptionalAttrDict(op.getOperation()->getAttrs());
}

static M::LogicalResult verify(TraceCaptureOp) {
  // TODO
  return M::success();
}

//===----------------------------------------------------------------------===//
// TraceConstructOp
//===----------------------------------------------------------------------===//

M::ParseResult parseTraceConstructOp(M::OpAsmParser &parser,
                                     M::OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return M::failure();

  auto &builder = parser.getBuilder();
  std::vector<M::Type> resultType = {TermType::get(builder.getContext())};
  result.addTypes(resultType);
  return M::success();
}

static void print(M::OpAsmPrinter &p, TraceConstructOp op) {
  p << op.getOperationName();
  p.printOptionalAttrDict(op.getOperation()->getAttrs());
}

static M::LogicalResult verify(TraceConstructOp) {
  // TODO
  return M::success();
}

//===----------------------------------------------------------------------===//
// Logical Ops
//===----------------------------------------------------------------------===//

// Parses `eir.<opname> %lhs : ty, %rhs : ty {...} : ty`
M::ParseResult parseLogicalOp(M::OpAsmParser &parser,
                              M::OperationState &result) {
  OperandType lhs;
  OperandType rhs;
  M::Type lhsType;
  M::Type rhsType;
  M::Builder &builder = parser.getBuilder();

  if (parser.parseOperand(lhs) ||
      parser.parseColonType(lhsType) ||
      parser.resolveOperand(lhs, lhsType, result.operands) ||
      parser.parseComma() ||
      parser.parseOperand(rhs) ||
      parser.parseColonType(rhsType) ||
      parser.resolveOperand(rhs, rhsType, result.operands) ||
      parser.parseOptionalAttrDict(result.attributes))
    return M::failure();

  M::Type resultType;
  if (parser.parseColonType(resultType)) {
    result.addTypes(builder.getI1Type());
  }  else {
    result.addTypes(resultType);
  }

  return M::success();
}

template <typename O>
void printLogicalOp(M::OpAsmPrinter &p, O &op) {
  p << op.getOperationName() << ' ';
  p << op.lhs() << ", ";
  p << op.rhs();
  p.printOptionalAttrDict(op.getOperation()->getAttrs());
  p << " : " << op.getResult().getType();
}

template <typename O>
static M::LogicalResult verifyLogicalOp(O) {
  // TODO
  return M::success();
}

//===----------------------------------------------------------------------===//
// Comparison Ops
//===----------------------------------------------------------------------===//

// Parses `eir.<opname> %lhs : ty, %rhs : ty {...} : ty`
M::ParseResult parseComparisonOp(M::OpAsmParser &parser,
                                 M::OperationState &result) {
  OperandType lhs;
  OperandType rhs;
  M::Type lhsType;
  M::Type rhsType;
  M::Builder &builder = parser.getBuilder();

  if (parser.parseOperand(lhs) ||
      parser.parseColonType(lhsType) ||
      parser.resolveOperand(lhs, lhsType, result.operands) ||
      parser.parseComma() ||
      parser.parseOperand(rhs) ||
      parser.parseColonType(rhsType) ||
      parser.resolveOperand(rhs, rhsType, result.operands) ||
      parser.parseOptionalAttrDict(result.attributes))
    return M::failure();

  M::Type resultType;
  if (parser.parseColonType(resultType)) {
    result.addTypes(builder.getI1Type());
  }  else {
    result.addTypes(resultType);
  }

  return M::success();
}

template <typename O>
void printComparisonOp(M::OpAsmPrinter &p, O &op) {
  p << op.getOperationName() << ' ';
  p << op.lhs() << ", ";
  p << op.rhs();
  p.printOptionalAttrDict(op.getOperation()->getAttrs());
  p << " : " << op.getResult().getType();
}

template <typename O>
static M::LogicalResult verifyComparisonOp(O) {
  // TODO
  return M::success();
}

//===----------------------------------------------------------------------===//
// MapPutOps (MapInsertOp, MapUpdateOp)
//===----------------------------------------------------------------------===//

// Parses `eir.<opname> %map, [(%key : <type>, %value : <type>), ...]`
M::ParseResult parseMapPutOp(M::OpAsmParser &parser,
                             M::OperationState &result) {
  OperandType map;
  L::SmallVector<L::SmallVector<M::Value, 2>, 1> updates;
  M::Builder &builder = parser.getBuilder();
  MapType mapType = MapType::get(builder.getContext());

  // Parse: %map,
  if (parser.parseOperand(map) ||
      parser.resolveOperand(map, mapType, result.operands) ||
      parser.parseComma())
    return M::failure();

  // Parse: [(%k, %v), ..]
  while (true) {
    OperandType keyOperand;
    OperandType valueOperand;
    M::Type keyType;
    M::Type valueType;
    if (parser.parseLParen() || parser.parseOperand(keyOperand) ||
        parser.parseColonType(keyType) ||
        parser.resolveOperand(keyOperand, keyType, result.operands) ||
        parser.parseComma() || parser.parseOperand(valueOperand) ||
        parser.parseColonType(valueType) ||
        parser.resolveOperand(valueOperand, valueType, result.operands) ||
        parser.parseRParen()) {
      return M::failure();
    }
  }

  // MapPut operations return the updated map and a bitflag which is set in case
  // of an error
  std::vector<M::Type> resultTypes = {
      MapType::get(builder.getContext()),
      builder.getI1Type(),
  };
  result.addTypes(resultTypes);

  return M::success();
}

template <typename O>
void printMapPutOp(M::OpAsmPrinter &p, O &op) {
  p << op.getOperationName() << ' ' << op.map() << '[';

  auto end = op.operand_end();
  chunked(std::next(op.operand_begin()), end, 2,
          [&](OperandIterator &k, OperandIterator &v) {
            auto last = v == end;
            p << "( ";
            p.printOperand(*k);
            p << ", ";
            p.printOperand(*v);
            p << ')';
            if (!last)
              p << ',';
          });

  p << ']';
  p.printOptionalAttrDict(op.getOperation()->getAttrs());
}

template <typename O>
static M::LogicalResult verifyMapPutOp(O) {
  // TODO
  return M::success();
}

//===----------------------------------------------------------------------===//
// BinaryPushOp
//===----------------------------------------------------------------------===//

// Parses: eir.binary_push %bin : <ty>, %val : <ty> { type = integer, signed =
// true, endian = big, unit = 1 }
M::ParseResult parseBinaryPushOp(M::OpAsmParser &parser,
                                 M::OperationState &result) {
  // Parse: %bin, %val : <ty>
  OperandType bin;
  OperandType val;
  M::Type binType;
  M::Type valType;
  if (parser.parseOperand(bin) || parser.parseColonType(binType) ||
      parser.resolveOperand(bin, binType, result.operands) ||
      parser.parseComma() || parser.parseOperand(val) ||
      parser.parseColonType(valType) ||
      parser.resolveOperand(val, valType, result.operands))
    return M::failure();

  // Parse: { key = val, ... }
  SmallVector<M::NamedAttribute, 4> attrs;
  if (parser.parseOptionalAttrDict(attrs))
    return M::failure();

  if (attrs.empty()) {
    parser.emitError(parser.getNameLoc(),
                     "expected binary specification attributes");
    return M::failure();
  }

  // TODO: verify attributes before adding them
  result.addAttributes(attrs);

  // This op is multi-result, returning the updated binary and an error flag
  auto builder = parser.getBuilder();
  std::vector<M::Type> resultTypes = {
      binType,
      builder.getI1Type(),
  };
  result.addTypes(resultTypes);
  return M::success();
}

static void print(M::OpAsmPrinter &p, BinaryPushOp op) {
  p << op.getOperationName();
  p.printOperand(op.bin());
  p << " : ";
  p.printType(op.bin().getType());
  p << ", ";
  p.printOperand(op.val());
  p << " : ";
  p.printType(op.val().getType());
  p.printOptionalAttrDict(op.getOperation()->getAttrs());
}

static M::LogicalResult verify(BinaryPushOp) {
  // TODO
  return M::success();
}

//===----------------------------------------------------------------------===//
// AllocatableOp
//===----------------------------------------------------------------------===//

template <typename O>
static M::ParseResult parseAllocatableOp(M::OpAsmParser &parser,
                                         M::OperationState &result) {
  M::Type intype;
  if (parser.parseType(intype))
    return M::failure();

  result.addAttribute("in_type", M::TypeAttr::get(intype));

  L::SmallVector<M::OpAsmParser::OperandType, 8> operands;
  L::SmallVector<M::Type, 8> typeVec;

  auto &builder = parser.getBuilder();

  bool hasOperands = false;
  if (!parser.parseOptionalLParen()) {
    if (parser.parseOperandList(operands, M::OpAsmParser::Delimiter::None) ||
        parser.parseRParen())
      return M::failure();
    auto lens = builder.getI32IntegerAttr(operands.size());
    result.addAttribute(O::lenpName(), lens);
    hasOperands = true;
  }

  if (!parser.parseOptionalComma()) {
    if (parser.parseOperandList(operands, M::OpAsmParser::Delimiter::None))
      return M::failure();
    hasOperands = true;
  }

  if (hasOperands &&
      parser.resolveOperands(operands, builder.getIndexType(),
                             parser.getNameLoc(), result.operands))
    return M::failure();

  M::Type restype = O::wrapResultType(intype);
  if (!restype) {
    parser.emitError(parser.getNameLoc(), "invalid allocate type: ") << intype;
    return M::failure();
  }

  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.addTypeToList(restype, result.types))
    return M::failure();
  return M::success();
}

template <typename O>
static void printAllocatableOp(OpAsmPrinter &p, O allocOp) {
  p << allocOp.getOperationName() << ' ' << allocOp.getAttr("in_type");
  if (allocOp.hasLenParams()) {
    p << '(';
    p.printOperands(allocOp.getLenParams());
    p << ')';
  }
  for (auto sh : allocOp.getShapeOperands()) {
    p << ", ";
    p.printOperand(sh);
  }
  auto *op = allocOp.getOperation();
  p.printOptionalAttrDict(op->getAttrs(), {"in_type", allocOp.lenpName()});
}

//===----------------------------------------------------------------------===//
// Constant*Op
//===----------------------------------------------------------------------===//

static M::ParseResult parseConstantOp(M::OpAsmParser &parser,
                                      M::OperationState &result) {
  M::Attribute valueAttr;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseAttribute(valueAttr, "value", result.attributes))
    return failure();

  // If the attribute is a symbol reference, then we expect a trailing type.
  M::Type type;
  if (!valueAttr.isa<M::SymbolRefAttr>())
    type = valueAttr.getType();
  else if (parser.parseColonType(type))
    return failure();

  // Add the attribute type to the list.
  return parser.addTypeToList(type, result.types);
}

static void print(M::OpAsmPrinter &p, ConstantOp &op) {
  p << op.getOperationName() << ' ';
  p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{"value"});

  if (op.getAttrs().size() > 1)
    p << ' ';
  p << op.getValue();

  // If the value is a symbol reference, print a trailing type.
  if (op.getValue().isa<M::SymbolRefAttr>())
    p << " : " << op.getType();
}

/// The constant op requires an attribute, and furthermore requires that it
/// matches the return type.
static M::LogicalResult verify(ConstantOp &op) {
  auto value = op.getValue();
  if (!value)
    return op.emitOpError("requires a 'value' attribute");

  auto type = op.getType();
  if (!value.getType().isa<M::NoneType>() && type != value.getType())
    return op.emitOpError() << "requires attribute's type (" << value.getType()
                            << ") to match op's return type (" << type << ")";

  if (type.isa<M::IndexType>() || type.isa<NilType>() || value.isa<M::BoolAttr>())
    return success();

  if (type.isa<AtomType>()) {
    if (!value.isa<M::StringAttr>())
      return op.emitOpError("requires 'value' to be a string constant");
    return success();
  }

  if (type.isa<FixnumType>() || type.isa<BigIntType>()) {
    if (!value.isa<M::IntegerAttr>())
      return op.emitOpError("requires 'value' to be an integer constant");
    return success();
  }

  if (type.isa<BinaryType>()) {
    if (!value.isa<BinaryAttr>())
      return op.emitOpError("requires 'value' to be a string constant");
    return success();
  }

  if (auto intAttr = value.dyn_cast<M::IntegerAttr>()) {
    // If the type has a known bitwidth we verify that the value can be
    // represented with the given bitwidth.
    auto bitwidth = type.cast<M::IntegerType>().getWidth();
    auto intVal = intAttr.getValue();
    if (!intVal.isSignedIntN(bitwidth) && !intVal.isIntN(bitwidth))
      return op.emitOpError("requires 'value' to be an integer within the "
                            "range of the integer result type");
    return success();
  }

  if (type.isa<M::FloatType>() || type.isa<FloatType>() || type.isa<PackedFloatType>()) {
    if (!value.isa<M::FloatAttr>())
      return op.emitOpError("requires 'value' to be a floating point constant");
    return success();
  }

  if (type.isa<ConsType>() || type.isa<TupleType>() || type.isa<MapType>()) {
    if (!value.isa<SeqAttr>())
      return op.emitOpError("requires 'value' to be a sequence constant");
    return success();
  }

  if (type.isa<M::ShapedType>()) {
    if (!value.isa<M::ElementsAttr>())
      return op.emitOpError("requires 'value' to be a shaped constant");
    return success();
  }

  if (type.isa<M::FunctionType>()) {
    auto fnAttr = value.dyn_cast<M::FlatSymbolRefAttr>();
    if (!fnAttr)
      return op.emitOpError("requires 'value' to be a function reference");

    // Try to find the referenced function.
    auto fn =
        op.getParentOfType<M::ModuleOp>().lookupSymbol<M::FuncOp>(fnAttr.getValue());
    if (!fn)
      return op.emitOpError("reference to undefined function 'bar'");

    // Check that the referenced function has the correct type.
    if (fn.getType() != type)
      return op.emitOpError("reference to function with mismatched type");

    return success();
  }

  if (type.isa<M::NoneType>() && value.isa<M::UnitAttr>())
    return success();

  return op.emitOpError("unsupported 'value' attribute: ") << value;
}

M::OpFoldResult ConstantOp::fold(ArrayRef<M::Attribute> operands) {
  assert(operands.empty() && "constant has no operands");
  return getValue();
}

void ConstantOp::getAsmResultNames(
    function_ref<void(M::Value, StringRef)> setNameFn) {
  M::Type type = getType();
  if (auto intCst = getValue().dyn_cast<M::IntegerAttr>()) {
    M::IntegerType intTy = type.dyn_cast<M::IntegerType>();

    // Sugar i1 constants with 'true' and 'false'.
    if (intTy && intTy.getWidth() == 1)
      return setNameFn(getResult(), (intCst.getInt() ? "true" : "false"));

    // Otherwise, build a complex name with the value and type.
    SmallString<32> specialNameBuffer;
    L::raw_svector_ostream specialName(specialNameBuffer);
    specialName << 'c' << intCst.getInt();
    if (intTy)
      specialName << '_' << type;
    setNameFn(getResult(), specialName.str());

  } else if (type.isa<M::FunctionType>()) {
    setNameFn(getResult(), "f");
  } else {
    setNameFn(getResult(), "cst");
  }
}

/// Returns true if a constant operation can be built with the given value and
/// result type.
bool ConstantOp::isBuildableWith(M::Attribute value, M::Type type) {
  // SymbolRefAttr can only be used with a function type.
  if (value.isa<M::SymbolRefAttr>())
    return type.isa<M::FunctionType>();

  // Nil constants, value must be the integer 0
  if (type.isa<NilType>()) {
    if (auto intAttr = value.dyn_cast<M::IntegerAttr>()) {
      if (intAttr.getInt() == 0) {
        return true;
      }
    }
    return false;
  }

  // Atom constants must be strings
  if (type.isa<AtomType>())
    return value.isa<M::StringAttr>();

  // Binary constants must be binaries
  if (value.isa<BinaryAttr>())
    return type.isa<BinaryType>();

  // Fixnum & BigInt constants must be integers
  if (type.isa<FixnumType>() || type.isa<BigIntType>())
    return value.isa<M::IntegerAttr>();

  // Float constants must be floats
  if (type.isa<FloatType>() || type.isa<PackedFloatType>() || type.isa<M::FloatType>())
    return value.isa<M::FloatAttr>();

  // Cons & Tuple constants must be arrays
  if (type.isa<ConsType>() || type.isa<TupleType>() || type.isa<MapType>())
    return value.isa<SeqAttr>();

  // Otherwise, the attribute must have the same type as 'type'.
  if (value.getType() != type)
    return false;

  // Finally, check that the attribute kind is handled.
  return value.isa<M::BoolAttr>() || value.isa<M::IntegerAttr>() ||
         value.isa<M::FloatAttr>() || value.isa<M::ElementsAttr>() ||
         value.isa<M::UnitAttr>();
}

void ConstantFloatOp::build(M::Builder *builder, M::OperationState &result,
                            const L::APFloat &value, M::Type type) {
  ConstantOp::build(builder, result, type, builder->getFloatAttr(type, value));
}

bool ConstantFloatOp::classof(M::Operation *op) {
  auto type = op->getResult(0).getType();
  return ConstantOp::classof(op) &&
    (type.isa<FloatType>() || type.isa<PackedFloatType>());
}

void ConstantIntOp::build(M::Builder *builder, M::OperationState &result,
                          int64_t value, unsigned width) {
  Type type = builder->getIntegerType(width);
  ConstantOp::build(builder, result, type,
                    builder->getIntegerAttr(type, value));
}

bool ConstantIntOp::classof(M::Operation *op) {
  return ConstantOp::classof(op) &&
         op->getResult(0).getType().isa<FixnumType>();
}

void ConstantBigIntOp::build(M::Builder *builder, M::OperationState &result,
                             const L::APInt &value) {
  auto type = builder->getType<BigIntType>();
  ConstantOp::build(builder, result, type,
                    builder->getIntegerAttr(type, value));
}

bool ConstantBigIntOp::classof(M::Operation *op) {
  return ConstantOp::classof(op) &&
         op->getResult(0).getType().isa<BigIntType>();
}

void ConstantAtomOp::build(M::Builder *builder, M::OperationState &result,
                           StringRef value, L::APInt id) {
  auto type = builder->getType<AtomType>();
  auto attr = AtomAttr::get(value, id, type);
  ConstantOp::build(builder, result, type, attr);
}

bool ConstantAtomOp::classof(M::Operation *op) {
  return ConstantOp::classof(op) &&
         op->getResult(0).getType().isa<AtomType>();
}

void ConstantBinaryOp::build(M::Builder *builder, M::OperationState &result,
                             ArrayRef<char> value, uint64_t header, uint64_t flags, unsigned width) {
  auto type = builder->getType<BinaryType>();
  auto attr = BinaryAttr::get(value, header, flags, width, type);
  ConstantOp::build(builder, result, type, attr);
}

bool ConstantBinaryOp::classof(M::Operation *op) {
  return ConstantOp::classof(op) &&
         op->getResult(0).getType().isa<BinaryType>();
}

void ConstantNilOp::build(M::Builder *builder, M::OperationState &result,
                          int64_t value, unsigned width) {
  auto nilType = builder->getType<NilType>();
  Type type = builder->getIntegerType(width);
  ConstantOp::build(builder, result, nilType,
                    builder->getIntegerAttr(type, value));
}

bool ConstantNilOp::classof(M::Operation *op) {
  return ConstantOp::classof(op) &&
         op->getResult(0).getType().isa<NilType>();
}

void ConstantSeqOp::build(M::Builder *builder, M::OperationState &result,
                          ArrayRef<M::Attribute> elements, M::Type type) {
  auto attr = SeqAttr::get(elements, type);
  ConstantOp::build(builder, result, type, attr);
}

bool ConstantSeqOp::classof(M::Operation *op) {
  return ConstantOp::classof(op) &&
    op->getAttr("value").isa<SeqAttr>();
}

//===----------------------------------------------------------------------===//
// TableGen Output
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "eir/Ops.cpp.inc"

} // namespace eir
