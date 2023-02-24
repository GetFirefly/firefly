#include "CIR/Ops.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"

using namespace mlir;
using namespace mlir::cir;

#define GET_OP_CLASSES
#include "CIR/CIR.cpp.inc"

struct CIRInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  /// All call operations within cir can be inlined.
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }

  /// All operations within cir can be inlined.
  bool isLegalToInline(Operation *, Region *, bool,
                       BlockAndValueMapping &) const final {
    return true;
  }

  // All functions within cir can be inlined.
  bool isLegalToInline(Region *, Region *, bool,
                       BlockAndValueMapping &) const final {
    return true;
  }

  //===--------------------------------------------------------------------===//
  // Transformation Hooks
  //===--------------------------------------------------------------------===//

  /// Handle the given inlined terminator by replacing it with a new
  /// operation if necessary.
  void handleTerminator(Operation *,
                        ArrayRef<Value> _valuesToRepl) const final {
    // Our only terminators are currently EnterOp and RaiseOp, which don't need
    // to be replaced as they remain terminators even after inlining
  }

  /// Attempts to materialize a conversion for a type mismatch between a call
  /// from this dialect, and a callable region. This method should generate an
  /// operation that takes 'input' as the only operand, and produces a single
  /// result of 'resultType'. If a conversion can not be generated, nullptr
  /// should be returned.
  Operation *materializeCallConversion(OpBuilder &builder, Value input,
                                       Type resultType,
                                       Location loc) const final {
    return builder.create<cir::CastOp>(loc, resultType, input);
  }
};

Operation *CIRDialect::materializeConstant(mlir::OpBuilder &builder,
                                           mlir::Attribute value,
                                           mlir::Type type,
                                           mlir::Location loc) {
  return builder.create<cir::ConstantOp>(loc, type, value);
}

void CIRDialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "CIR/CIR.cpp.inc"
      >();
}

void CIRDialect::registerInterfaces() { addInterfaces<CIRInlinerInterface>(); }

//===----------------------------------------------------------------------===//
// DispatchTableOp
//===----------------------------------------------------------------------===//

LogicalResult DispatchTableOp::verifyRegions() {
  llvm::SmallSet<std::pair<StringRef, unsigned>, 16> entries;
  for (auto &op : getBody()) {
    if (isa<CirEndOp>(op))
      continue;
    if (!isa<DispatchEntryOp>(op))
      return op.emitOpError("invalid operation within dispatch table, expected "
                            "cir.dispatch_entry or cir.end");
    auto entry = cast<DispatchEntryOp>(op);
    StringRef name = entry.getFunction();
    unsigned arity = entry.getArity();
    auto value = std::make_pair(name, arity);
    auto inserted = entries.insert(value);
    if (!std::get<bool>(inserted)) {
      return op.emitOpError("conflicting entry found in dispatch table");
    }
  }
  return mlir::success();
}

void DispatchTableOp::appendTableEntry(Operation *op) {
  assert(isa<DispatchEntryOp>(op) && "operation must be a DispatchEntryOp");
  auto &body = getBody();
  body.push_front(op);
  // body.getOperations().insert(body.end(), op);
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

LogicalResult CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the callee attribute was specified.
  auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!fnAttr)
    return emitOpError("requires a 'callee' symbol reference attribute");
  auto fn = symbolTable.lookupNearestSymbolFrom<func::FuncOp>(*this, fnAttr);
  if (!fn)
    return emitOpError() << "'" << fnAttr.getValue()
                         << "' does not reference a valid function";

  // Verify that the operand and result types match the callee.
  auto fnType = fn.getFunctionType();
  if (fnType.getNumInputs() != getNumOperands())
    return emitOpError("incorrect number of operands for callee");

  for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i) {
    Type srcTy = getOperand(i).getType();
    Type dstTy = fnType.getInput(i);
    if (srcTy == dstTy)
      continue;
    else if (srcTy.isa<TermType>() && dstTy.isa<TermType>())
      continue;
    else if (srcTy.isa<CIRExceptionType>() && dstTy.isa<CIROpaqueTermType>())
      continue;
    else
      return emitOpError("operand type mismatch: expected operand type ")
             << fnType.getInput(i) << ", but provided "
             << getOperand(i).getType() << " for operand number " << i;
  }

  if (fnType.getNumResults() != getNumResults())
    return emitOpError("incorrect number of results for callee");

  for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i) {
    Type srcTy = getResult(i).getType();
    Type dstTy = fnType.getResult(i);
    if (srcTy == dstTy)
      continue;
    else if (srcTy.isa<TermType>() && dstTy.isa<TermType>())
      continue;
    else {
      auto diag = emitOpError("result type mismatch at index ") << i;
      diag.attachNote() << "      op result types: " << getResultTypes();
      diag.attachNote() << "function result types: " << fnType.getResults();
      return diag;
    }
  }

  return success();
}

FunctionType CallOp::getCalleeType() {
  return FunctionType::get(getContext(), getOperandTypes(), getResultTypes());
}

//===----------------------------------------------------------------------===//
// EnterOp
//===----------------------------------------------------------------------===//

LogicalResult EnterOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the callee attribute was specified.
  auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!fnAttr)
    return emitOpError("requires a 'callee' symbol reference attribute");
  auto fn = symbolTable.lookupNearestSymbolFrom<func::FuncOp>(*this, fnAttr);
  if (!fn)
    return emitOpError() << "'" << fnAttr.getValue()
                         << "' does not reference a valid function";

  // Verify that the operand and result types match the callee.
  auto fnType = fn.getFunctionType();
  if (fnType.getNumInputs() != getNumOperands())
    return emitOpError("incorrect number of operands for callee");

  for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i) {
    Type srcTy = getOperand(i).getType();
    Type dstTy = fnType.getInput(i);
    if (srcTy == dstTy)
      continue;
    else if (srcTy.isa<TermType>() && dstTy.isa<TermType>())
      continue;
    else if (srcTy.isa<CIRExceptionType>() && dstTy.isa<CIROpaqueTermType>())
      continue;
    else
      return emitOpError("operand type mismatch: expected operand type ")
             << fnType.getInput(i) << ", but provided "
             << getOperand(i).getType() << " for operand number " << i;
  }

  auto parent = cast<func::FuncOp>((*this)->getParentOp());
  auto parentType = parent.getFunctionType();
  if (fnType.getNumResults() != parentType.getNumResults())
    return emitOpError(
        "callee result type does not match the containing function");

  for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i) {
    Type dstTy = parentType.getResult(i);
    Type srcTy = fnType.getResult(i);
    if (srcTy == dstTy)
      continue;
    else if (srcTy.isa<TermType>() && dstTy.isa<TermType>())
      continue;
    else {
      auto diag = emitOpError("result type mismatch at index ") << i;
      diag.attachNote() << "expected result types: " << parentType.getResults();
      diag.attachNote() << "  callee result types: " << fnType.getResults();
      return diag;
    }
  }

  return success();
}

FunctionType EnterOp::getCalleeType() {
  auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!fnAttr)
    return nullptr;
  auto fn = cast<func::FuncOp>(
      mlir::SymbolTable::lookupNearestSymbolFrom(*this, fnAttr));
  return fn.getFunctionType();
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

bool canCastBetween(Type input, Type output) {
  // Identity casts are trivially supported
  if (input == output)
    return true;

  auto isTermOutput = isTermType(output);
  // Opaque terms are always castable to a term type, numeric, or tuple type
  if (input.isa<CIROpaqueTermType>() && isTermOutput)
    return true;

  // Opaque terms are always castable to special primop pointer types
  if (input.isa<CIROpaqueTermType>()) {
    if (auto ptrTy = output.dyn_cast_or_null<PtrType>()) {
      auto innerTy = ptrTy.getElementType();
      // *mut BitVec
      if (innerTy.isa<CIRBinaryBuilderType>())
        return true;
      // *mut MatchContext
      if (innerTy.isa<CIRMatchContextType>())
        return true;
      // *mut ErlangException
      if (innerTy.isa<CIRExceptionType>())
        return true;
    }

    // Casts from opaque term to i64 are intended as a bitcast from term
    // type to native machine integer representation WITHOUT decoding the value
    if (output.isInteger(64))
      return true;
  }

  // All term, numeric or tuple types are castable to an opaque term
  auto isTermInput = isTermType(input);
  auto isInputNumeric = isTypePrimitiveNumeric(input);
  auto isInputTuple = false;
  auto isInputBox = false;
  if (auto boxTy = input.dyn_cast<CIRBoxType>()) {
    isInputBox = true;
    isInputTuple = boxTy.getElementType().isa<TupleType>();
  }
  if ((isTermInput || isInputNumeric || isInputTuple) &&
      output.isa<CIROpaqueTermType>())
    return true;

  // None can be cast to term/Exception
  if (input.isa<CIRNoneType>() &&
      (isTermOutput || output.isa<CIRExceptionType>()))
    return true;
  // Exceptions can be cast to term for comparison against None
  if (input.isa<CIRExceptionType>() && output.isa<CIROpaqueTermType>())
    return true;
  // Otherwise special types cannot be cast to
  if (output.isa<CIRNoneType, CIRExceptionType, CIRTraceType,
                 CIRRecvContextType, CIRBinaryBuilderType>())
    return false;

  auto isOutputNumeric = isTypePrimitiveNumeric(output);
  auto isInputBool = input.isInteger(1) || input.isa<CIRBoolType>();
  auto isOutputBool = output.isInteger(1) || output.isa<CIRBoolType>();

  // All primitive numeric types are interchangeable
  if (isInputNumeric && isOutputNumeric)
    return true;

  // All bool types are interchangeable
  if (isInputBool && isOutputBool)
    return true;

  // Primitive numerics are castable to bigint or boolean
  if (isInputNumeric) {
    if (output.isInteger(1) || output.isa<CIRBoolType>())
      return true;
    if (auto boxTy = output.dyn_cast<CIRBoxType>())
      return boxTy.getElementType().isa<CIRBigIntType>();
  }

  // Bools are interchangeable with atoms
  if (isInputBool && output.isa<CIRAtomType>())
    return true;
  else if (input.isa<CIRAtomType>() && isOutputBool)
    return true;

  // Boxes can be cast to pointers and vice versa, as long as the contents are
  // cast-compatible
  if (isInputBox) {
    auto inputBoxType = input.cast<CIRBoxType>();
    if (auto outputBoxType = output.dyn_cast<CIRBoxType>())
      return canCastBetween(inputBoxType.getElementType(),
                            outputBoxType.getElementType());
    if (auto outputPtrType = output.dyn_cast<PtrType>())
      return canCastBetween(inputBoxType.getElementType(),
                            outputPtrType.getElementType());
  } else if (auto inputPtrType = input.dyn_cast<PtrType>()) {
    if (auto outputBoxType = output.dyn_cast<CIRBoxType>())
      return canCastBetween(inputPtrType.getElementType(),
                            outputBoxType.getElementType());
    if (auto outputPtrType = output.dyn_cast<PtrType>())
      return canCastBetween(inputPtrType.getElementType(),
                            outputPtrType.getElementType());
  }

  // Unrecognized cast
  return false;
}

bool cir::CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  // We only support input/output casts with the same number of values
  if (inputs.size() != outputs.size())
    return false;

  // We zip up the types and check that each cast is valid, i.e. that the input
  // type is actually convertible to the output type. For non-CIR types, we are
  // primarily interested in conversions of integer/float types to their CIR
  // representation, or vice-versa
  for (auto it : llvm::zip(inputs, outputs)) {
    Type inputType = std::get<0>(it);
    Type outputType = std::get<1>(it);

    if (!canCastBetween(inputType, outputType))
      return false;
  }
  return true;
}

LogicalResult cir::CastOp::fold(ArrayRef<Attribute> attrOperands,
                                SmallVectorImpl<OpFoldResult> &foldResults) {
  OperandRange operands = getInputs();
  ResultRange results = getOutputs();

  if (operands.getType() == results.getType()) {
    foldResults.append(operands.begin(), operands.end());
    return success();
  }

  if (operands.empty())
    return failure();

  // Check that the input is a cast with results that all feed into this
  // operation, and operand types that directly match the result types of this
  // operation.
  Value firstInput = operands.front();
  auto inputOp = firstInput.getDefiningOp<cir::CastOp>();
  if (!inputOp || inputOp.getResults() != operands ||
      inputOp.getOperandTypes() != results.getTypes())
    return failure();

  // If everything matches up, we can fold the passthrough.
  foldResults.append(inputOp->operand_begin(), inputOp->operand_end());
  return success();
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

bool cir::ConstantOp::isBuildableWith(Attribute value, Type type) {
  // Must be a concrete term type
  if (type.isa<CIRIntegerType, CIRNumberType, CIROpaqueTermType, CIRNoneType>())
    return false;
  // Must not be a box or fun type
  if (type.isa<CIRFunType, CIRBoxType>())
    return false;
  return true;
}

OpFoldResult cir::ConstantOp::fold(ArrayRef<Attribute> operands) {
  return getValue();
}

//===----------------------------------------------------------------------===//
// ConstantNullOp
//===----------------------------------------------------------------------===//

bool cir::ConstantNullOp::isBuildableWith(Attribute value, Type type) {
  return true;
}

//===----------------------------------------------------------------------===//
// MallocOp
//===----------------------------------------------------------------------===//

LogicalResult cir::MallocOp::inferReturnTypes(
    MLIRContext *context, llvm::Optional<Location> location,
    ValueRange operands, DictionaryAttr attributes, RegionRange regions,
    llvm::SmallVectorImpl<Type> &inferredReturnTypes) {
  auto allocTypeAttr = attributes.get("allocType").cast<TypeAttr>();
  auto allocType = allocTypeAttr.getValue();
  if (auto boxTy = allocType.dyn_cast<CIRBoxType>()) {
    auto returnType = PtrType::get(boxTy.getElementType());
    inferredReturnTypes.assign({returnType});
  } else {
    auto returnType = PtrType::get(allocType);
    inferredReturnTypes.assign({returnType});
  }
  return success();
}

//===----------------------------------------------------------------------===//
// MakeFunOp
//===----------------------------------------------------------------------===//

LogicalResult cir::MakeFunOp::inferReturnTypes(
    MLIRContext *context, llvm::Optional<Location> location,
    ValueRange operands, DictionaryAttr attributes, RegionRange regions,
    llvm::SmallVectorImpl<Type> &inferredReturnTypes) {
  auto funTypeAttr = attributes.get("funType").cast<TypeAttr>();
  auto returnType = CIRBoxType::get(funTypeAttr.getValue());
  auto i1Type = IntegerType::get(context, 1);
  inferredReturnTypes.assign({i1Type, returnType});
  return success();
}
