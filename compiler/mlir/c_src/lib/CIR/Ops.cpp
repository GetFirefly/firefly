#include "CIR/Ops.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/Optional.h"
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
    // Our only terminator currently is RaiseOp, which doesn't need to
    // be replaced as it remains a terminator even after inlining
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
// CastOp
//===----------------------------------------------------------------------===//

bool canCastBetween(Type input, Type output) {
  auto isCirOutput = isCIRType(output);
  // Opaque terms are always castable to a term type, numeric, or tuple type
  if (input.isa<CIRTermType>() && isCirOutput)
    return true;

  // All term, numeric or tuple types are castable to an opaque term
  auto isCirInput = isCIRType(input);
  auto isInputNumeric = isTypePrimitiveNumeric(input);
  auto isInputTuple = false;
  auto isInputBox = false;
  if (auto boxTy = input.dyn_cast<CIRBoxType>()) {
    isInputBox = true;
    isInputTuple = boxTy.getElementType().isa<TupleType>();
  }
  if ((isCirInput || isInputNumeric || isInputTuple) &&
      output.isa<CIRTermType>())
    return true;

  // Special types can be cast to Term, but not vice-versa
  if (output.isa<CIRNoneType, CIRExceptionType, CIRTraceType,
                 CIRRecvContextType, CIRBinaryBuilderType>())
    return false;

  auto isOutputNumeric = isTypePrimitiveNumeric(output);
  auto isInputBool = input.isInteger(1) || input.isa<CIRBoolType>();
  auto isOutputBool = output.isInteger(1) || output.isa<CIRBoolType>();

  // All CIR types or primitive numerics are castable to an opaque term
  if (output.isa<CIRTermType>() && (isCirInput || isInputNumeric))
    return true;

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
  OperandRange operands = inputs();
  ResultRange results = outputs();

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
  // The types must match
  auto valueType = value.getType();
  if (valueType != type)
    return false;
  // Must be a concrete term type
  if (valueType.isa<CIRIntegerType, CIRNumberType, CIRTermType, CIRNoneType>())
    return false;
  // Must not be a box or fun type
  if (valueType.isa<CIRFunType, CIRBoxType>())
    return false;
  return true;
}

OpFoldResult cir::ConstantOp::fold(ArrayRef<Attribute> operands) {
  return value();
}

//===----------------------------------------------------------------------===//
// MallocOp
//===----------------------------------------------------------------------===//

LogicalResult cir::MallocOp::inferReturnTypes(
    MLIRContext *context, llvm::Optional<Location> location,
    ValueRange operands, DictionaryAttr attributes, RegionRange regions,
    llvm::SmallVectorImpl<Type> &inferredReturnTypes) {
  auto allocTypeAttr = attributes.get("allocType").cast<TypeAttr>();
  auto returnType = PtrType::get(allocTypeAttr.getValue());
  inferredReturnTypes.assign({returnType});
  return success();
}

//===----------------------------------------------------------------------===//
// CaptureFunOp
//===----------------------------------------------------------------------===//

LogicalResult cir::CaptureFunOp::inferReturnTypes(
    MLIRContext *context, llvm::Optional<Location> location,
    ValueRange operands, DictionaryAttr attributes, RegionRange regions,
    llvm::SmallVectorImpl<Type> &inferredReturnTypes) {
  auto funTypeAttr = attributes.get("funType").cast<TypeAttr>();
  auto returnType = CIRBoxType::get(funTypeAttr.getValue());
  inferredReturnTypes.assign({returnType});
  return success();
}
