#include "lumen/LLVM.h"

#include "eir/Ops.h"
#include "eir/Types.h"

#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"

#include <iterator>
#include <vector>

using namespace eir;

namespace L = llvm;
namespace M = mlir;

using L::SmallVector;
using OperandType = M::OpAsmParser::OperandType;
using OperandIterator = M::OperandIterator;
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

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

void IfOp::build(M::Builder *builder, M::OperationState &result, M::Value *cond,
                 bool withElseRegion) {
  result.addOperands(cond);
  M::Region *thenRegion = result.addRegion();
  M::Region *elseRegion = result.addRegion();
}

static M::ParseResult parseIfOp(M::OpAsmParser &parser,
                                M::OperationState &result) {
  // Create the regions for 'then'.
  result.regions.reserve(2);
  M::Region *thenRegion = result.addRegion();
  M::Region *elseRegion = result.addRegion();

  auto &builder = parser.getBuilder();
  M::OpAsmParser::OperandType cond;
  M::Type i1Type = builder.getIntegerType(1);
  if (parser.parseOperand(cond) ||
      parser.resolveOperand(cond, i1Type, result.operands))
    return failure();

  // Parse the 'then' region.
  if (parser.parseRegion(*thenRegion, {}, {}))
    return failure();

  // If we find an 'else' keyword then parse the 'else' region.
  if (!parser.parseOptionalKeyword("else")) {
    if (parser.parseRegion(*elseRegion, {}, {}))
      return failure();
  }

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}

static void print(M::OpAsmPrinter &p, IfOp op) {
  p << IfOp::getOperationName() << " " << *op.condition();
  p.printRegion(op.thenRegion(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);

  // Print the 'else' regions if it exists and has a block.
  auto &elseRegion = op.elseRegion();
  if (!elseRegion.empty()) {
    p << " else";
    p.printRegion(elseRegion,
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

      return op
          .emitOpError("expects regions to end with 'return', 'br', 'cond_br', "
                       "or 'eir.call', found '" +
                       terminator.getName().getStringRef() + "'")
          .attachNote();
    }
  }

  return M::success();
}

//===----------------------------------------------------------------------===//
// MergeOp
//===----------------------------------------------------------------------===//

static ParseResult parseMergeOp(M::OpAsmParser &parser,
                                M::OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return M::failure();
  return M::success();
}

static void print(M::OpAsmPrinter &p, MergeOp op) {
  p << op.getOperationName();
  p.printOptionalAttrDict(op.getOperation()->getAttrs());
}

static M::LogicalResult verify(MergeOp mergeOp) {
  M::Block &parentLastBlock = mergeOp.getParentRegion()->back();
  if (mergeOp.getOperation() != parentLastBlock.getTerminator())
    return mergeOp.emitOpError(
        "can only be used in the last block of 'eir.match'");
  return M::success();
}

static inline bool isMergeBlock(M::Block &block) {
  return !block.empty() && std::next(block.begin()) == block.end() &&
         isa<MergeOp>(block.front());
}

//===----------------------------------------------------------------------===//
// MatchOp
//===----------------------------------------------------------------------===//

static M::ParseResult parseMatchOp(M::OpAsmParser &parser,
                                   M::OperationState &result) {
  return parser.parseRegion(*result.addRegion(),
                            /*arguments=*/{},
                            /*argTypes=*/{});
}

static void print(OpAsmPrinter &p, MatchOp matchOp) {
  auto *op = matchOp.getOperation();

  p << MatchOp::getOperationName();
  p.printRegion(op->getRegion(0),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}

static M::LogicalResult verify(MatchOp matchOp) {
  auto *op = matchOp.getOperation();

  // We need to verify that the blocks follow the following layout:
  //
  //                     +--------------+
  //                     | header block |
  //                     +--------------+
  //                          / | \
  //                           ...
  //
  //
  //         +---------+   +---------+   +---------+
  //         | case #0 |   | case #1 |   | case #2 |  ...
  //         +---------+   +---------+   +---------+
  //
  //
  //                           ...
  //                          \ | /
  //                            v
  //                     +-------------+
  //                     | merge block |
  //                     +-------------+

  auto &region = op->getRegion(0);
  // Allow empty region as a degenerate case, which can come from
  // optimizations.
  if (region.empty())
    return success();

  // The last block is the merge block.
  if (!isMergeBlock(region.back()))
    return matchOp.emitOpError(
        "last block must be the merge block with only one 'eir.merge' op");

  if (std::next(region.begin()) == region.end())
    return matchOp.emitOpError("must have a match header block");

  return success();
}

M::Block *MatchOp::getHeaderBlock() {
  assert(!body().empty() && "op region should not be empty!");
  // The first block is the header block.
  return &body().front();
}

Block *MatchOp::getMergeBlock() {
  assert(!body().empty() && "op region should not be empty!");
  // The last block is the merge block.
  return &body().back();
}

void MatchOp::addMergeBlock() {
  assert(body().empty() && "entry and merge block already exist");
  auto *mergeBlock = new M::Block();
  body().push_back(mergeBlock);
  M::OpBuilder builder(mergeBlock);

  // Add a eir.merge op into the merge block.
  builder.create<MergeOp>(getLoc());
}

void MatchOp::getCanonicalizationPatterns(M::OwningRewritePatternList &results,
                                          M::MLIRContext *context) {
  // TODO?
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
    L::SmallVector<M::Value *, 8> resultArgs(
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
    p << *op.getOperand(0);
  p << '(';
  p.printOperands(L::drop_begin(op.getOperands(), isDirect ? 0 : 1));
  p << ')';
  p.printOptionalAttrDict(op.getAttrs(), {"callee"});
  L::SmallVector<Type, 1> resultTypes(op.getResultTypes());
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
      if (op.getOperand(i)->getType() != fnType.getInput(i))
        return op.emitOpError("operand type mismatch");

    if (fnType.getNumResults() != op.getNumResults())
      return op.emitOpError("incorrect number of results for callee");

    for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i)
      if (op.getResult(i)->getType() != fnType.getResult(i))
        return op.emitOpError("result type mismatch");
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
      if (op.getOperand(i + 1)->getType() != fnType.getInput(i))
        return op.emitOpError("operand type mismatch");

    if (fnType.getNumResults() != op.getNumResults())
      return op.emitOpError("incorrect number of results for callee");

    for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i)
      if (op.getResult(i)->getType() != fnType.getResult(i))
        return op.emitOpError("result type mismatch");
  }

  return M::success();
}

M::FunctionType CallOp::getCalleeType() {
  SmallVector<M::Type, 4> resultTypes(getResultTypes());
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
  p << " : " << op.memref()->getType();
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
  p << " : " << op.memref()->getType();
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
// MapPutOps (MapInsertOp, MapUpdateOp)
//===----------------------------------------------------------------------===//

// Parses `eir.<opname> %map, [(%key : <type>, %value : <type>), ...]`
M::ParseResult parseMapPutOp(M::OpAsmParser &parser,
                             M::OperationState &result) {
  OperandType map;
  L::SmallVector<L::SmallVector<M::Value *, 2>, 1> updates;
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
  p << op.getOperationName() << ' ' << *op.map() << '[';

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
  p.printType(op.bin()->getType());
  p << ", ";
  p.printOperand(op.val());
  p << " : ";
  p.printType(op.val()->getType());
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
// TableGen Output
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "eir/Ops.cpp.inc"

} // namespace eir
