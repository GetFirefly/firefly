#include "TypeDetail.h"

#include "CIR/Types.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::cir;

//===----------------------------------------------------------------------===//
// TableGen
//===----------------------------------------------------------------------===//

#include "CIR/CIRTypeInterfaces.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "CIR/CIRTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

void CIRDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "CIR/CIRTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// CIRBoxType
//===----------------------------------------------------------------------===//

LogicalResult CIRBoxType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 Type elementType) {
  if (!isBoxableType(elementType))
    return emitError() << "invalid element type for box: " << elementType;
  return success();
}

//===----------------------------------------------------------------------===//
// CIRFunType
//===----------------------------------------------------------------------===//

unsigned CIRFunType::getArity() const { return getCalleeType().getNumInputs(); }

FunctionType CIRFunType::getCalleeType() const {
  return getImpl()->getCalleeType();
}

ArrayRef<Type> CIRFunType::getEnvTypes() const {
  return getImpl()->getEnvTypes();
}

void CIRFunType::walkImmediateSubElements(
    function_ref<void(Attribute)> walkAttrsFn,
    function_ref<void(Type)> walkTypesFn) const {
  FunctionType calleeType = getCalleeType();
  auto subElementInterface =
      llvm::dyn_cast<mlir::SubElementTypeInterface>(calleeType);
  subElementInterface.walkImmediateSubElements(walkAttrsFn, walkTypesFn);
  ArrayRef<Type> envTypes = getEnvTypes();
  for (Type type : envTypes)
    walkTypesFn(type);
}

Type CIRFunType::parse(AsmParser &parser) {
  auto context = parser.getContext();
  if (parser.parseLess() || parser.parseLParen())
    return Type();
  SmallVector<Type, 1> envTypes;
  if (parser.parseTypeList(envTypes) || parser.parseRParen() ||
      parser.parseArrow())
    return Type();
  // If there is no '(', we're parsing a thin closure
  if (parser.parseLParen()) {
    SmallVector<Type, 1> resultTypes;
    if (parser.parseTypeList(resultTypes) || parser.parseGreater())
      return Type();
    return get(context, FunctionType::get(context, envTypes, resultTypes),
               ArrayRef<Type>{});
  } else {
    SmallVector<Type, 1> inputTypes;
    SmallVector<Type, 1> resultTypes;
    if (parser.parseTypeList(inputTypes) || parser.parseRParen() ||
        parser.parseArrow() || parser.parseTypeList(resultTypes))
      return Type();
    return get(context, FunctionType::get(context, inputTypes, resultTypes),
               envTypes);
  }
}

void CIRFunType::print(AsmPrinter &p) const {
  auto envTys = getEnvTypes();
  auto calleeTy = getCalleeType();
  if (envTys.size() == 0) {
    p << getCalleeType() << ">";
    return;
  }
  p << '[';
  llvm::interleaveComma(getEnvTypes(), p,
                        [&](Type envTy) { p.printType(envTy); });
  p << ']';
  p.printFunctionalType(calleeTy.getInputs(), calleeTy.getResults());
}

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

bool mlir::cir::isCIRType(Type type) {
  // clang-format off
  if (type.isa<
      CIRNoneType,
      CIROpaqueTermType,
      CIRNumberType,
      CIRIntegerType,
      CIRFloatType,
      CIRAtomType,
      CIRBoolType,
      CIRIsizeType,
      CIRBigIntType,
      CIRNilType,
      CIRConsType,
      TupleType,
      CIRMapType,
      CIRFunType,
      CIRBitsType,
      CIRBinaryType,
      CIRBoxType,
      CIRPidType,
      CIRReferenceType,
      CIRExceptionType,
      CIRTraceType,
      CIRRecvContextType,
      CIRBinaryBuilderType,
      PtrType
    >()) {
    // clang-format on
    return true;
  }
  return false;
}

bool mlir::cir::isTermType(Type type) {
  if (type.isa<TermType, TupleType>()) {
    return true;
  }
  return false;
}

bool mlir::cir::isImmediateType(Type type) { return type.isa<ImmediateType>(); }

bool mlir::cir::isBoxableType(Type type) {
  if (type.isa<BoxedType, TupleType, CIRIntegerType, CIRNumberType>()) {
    return true;
  }
  return false;
}

bool mlir::cir::isTypePrimitiveNumeric(Type ty) {
  if (ty.isIntOrIndexOrFloat())
    return true;
  if (ty.isa<CIRNumberType, CIRIntegerType, CIRFloatType, CIRIsizeType>())
    return true;
  return false;
}
