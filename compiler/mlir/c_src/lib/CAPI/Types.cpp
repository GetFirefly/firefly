#include "CIR-c/Dialects.h"

#include "CIR/Dialect.h"
#include "CIR/Types.h"

#include "mlir/CAPI/IR.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::cir;

MlirType mlirCirNoneTypeGet(MlirContext ctx) {
  return wrap(CIRNoneType::get(unwrap(ctx)));
}

bool mlirCirIsANoneType(MlirType type) {
  return unwrap(type).isa<CIRNoneType>();
}

MlirType mlirCirTermTypeGet(MlirContext ctx) {
  return wrap(CIROpaqueTermType::get(unwrap(ctx)));
}

bool mlirCirIsATermType(MlirType type) {
  return unwrap(type).isa<CIROpaqueTermType>();
}

MlirType mlirCirNumberTypeGet(MlirContext ctx) {
  return wrap(CIRNumberType::get(unwrap(ctx)));
}

bool mlirCirIsANumberType(MlirType type) {
  return unwrap(type).isa<CIRNumberType>();
}

MlirType mlirCirIntegerTypeGet(MlirContext ctx) {
  return wrap(CIRIntegerType::get(unwrap(ctx)));
}

bool mlirCirIsAIntegerType(MlirType type) {
  return unwrap(type).isa<CIRIntegerType>();
}

MlirType mlirCirFloatTypeGet(MlirContext ctx) {
  return wrap(CIRFloatType::get(unwrap(ctx)));
}

bool mlirCirIsAFloatType(MlirType type) {
  return unwrap(type).isa<CIRFloatType>();
}

MlirType mlirCirAtomTypeGet(MlirContext ctx) {
  return wrap(CIRAtomType::get(unwrap(ctx)));
}

bool mlirCirIsAAtomType(MlirType type) {
  return unwrap(type).isa<CIRAtomType>();
}

MlirType mlirCirBoolTypeGet(MlirContext ctx) {
  return wrap(CIRBoolType::get(unwrap(ctx)));
}

bool mlirCirIsABoolType(MlirType type) {
  return unwrap(type).isa<CIRBoolType>();
}

MlirType mlirCirIsizeTypeGet(MlirContext ctx) {
  return wrap(CIRIsizeType::get(unwrap(ctx)));
}

bool mlirCirIsAIsizeType(MlirType type) {
  return unwrap(type).isa<CIRIsizeType>();
}

MlirType mlirCirBigIntTypeGet(MlirContext ctx) {
  return wrap(CIRBigIntType::get(unwrap(ctx)));
}

bool mlirCirIsABigIntType(MlirType type) {
  return unwrap(type).isa<CIRBigIntType>();
}

MlirType mlirCirNilTypeGet(MlirContext ctx) {
  return wrap(CIRNilType::get(unwrap(ctx)));
}

bool mlirCirIsANilType(MlirType type) { return unwrap(type).isa<CIRNilType>(); }

MlirType mlirCirConsTypeGet(MlirContext ctx) {
  return wrap(CIRConsType::get(unwrap(ctx)));
}

bool mlirCirIsAConsType(MlirType type) {
  return unwrap(type).isa<CIRConsType>();
}

MlirType mlirCirMapTypeGet(MlirContext ctx) {
  return wrap(CIRMapType::get(unwrap(ctx)));
}

bool mlirCirIsAMapType(MlirType type) { return unwrap(type).isa<CIRMapType>(); }

MlirType mlirCirFunTypeGet(MlirContext ctx, intptr_t numResults,
                           MlirType const *resultTypes, intptr_t arity,
                           MlirType const *argumentTypes, intptr_t envSize,
                           MlirType const *envTypes) {
  SmallVector<Type, 2> resultStorage;
  SmallVector<Type, 2> argumentStorage;
  SmallVector<Type, 2> envStorage;
  auto context = unwrap(ctx);
  auto calleeType = FunctionType::get(
      context, unwrapList(numResults, resultTypes, resultStorage),
      unwrapList(arity, argumentTypes, argumentStorage));
  return wrap(CIRFunType::get(context, calleeType,
                              unwrapList(envSize, envTypes, envStorage)));
}

bool mlirCirIsAFunType(MlirType type) { return unwrap(type).isa<CIRFunType>(); }

MlirType mlirCirBitsTypeGet(MlirContext ctx) {
  return wrap(CIRBitsType::get(unwrap(ctx)));
}

bool mlirCirIsABitsType(MlirType type) {
  return unwrap(type).isa<CIRBitsType>();
}

MlirType mlirCirBinaryTypeGet(MlirContext ctx) {
  return wrap(CIRBinaryType::get(unwrap(ctx)));
}

bool mlirCirIsABinaryType(MlirType type) {
  return unwrap(type).isa<CIRBinaryType>();
}

MlirType mlirCirBoxTypeGet(MlirType pointee) {
  return wrap(CIRBoxType::get(unwrap(pointee)));
}

bool mlirCirIsABoxType(MlirType type) { return unwrap(type).isa<CIRBoxType>(); }

MlirType mlirCirPidTypeGet(MlirContext ctx) {
  return wrap(CIRPidType::get(unwrap(ctx)));
}

bool mlirCirIsAPidType(MlirType type) { return unwrap(type).isa<CIRPidType>(); }

MlirType mlirCirPortTypeGet(MlirContext ctx) {
  return wrap(CIRPortType::get(unwrap(ctx)));
}

bool mlirCirIsAPortType(MlirType type) {
  return unwrap(type).isa<CIRPortType>();
}

MlirType mlirCirReferenceTypeGet(MlirContext ctx) {
  return wrap(CIRReferenceType::get(unwrap(ctx)));
}

bool mlirCirIsAReferenceType(MlirType type) {
  return unwrap(type).isa<CIRReferenceType>();
}

MlirType mlirCirExceptionTypeGet(MlirContext ctx) {
  return wrap(CIRExceptionType::get(unwrap(ctx)));
}

bool mlirCirIsAExceptionType(MlirType type) {
  return unwrap(type).isa<CIRExceptionType>();
}

MlirType mlirCirTraceTypeGet(MlirContext ctx) {
  return wrap(CIRTraceType::get(unwrap(ctx)));
}

bool mlirCirIsATraceType(MlirType type) {
  return unwrap(type).isa<CIRTraceType>();
}

MlirType mlirCirPtrTypeGet(MlirType pointee) {
  return wrap(PtrType::get(unwrap(pointee)));
}

bool mlirCirIsAPtrType(MlirType type) { return unwrap(type).isa<PtrType>(); }

MlirType mlirCirRecvContextTypeGet(MlirContext ctx) {
  return wrap(CIRRecvContextType::get(unwrap(ctx)));
}

bool mlirCirIsARecvContextType(MlirType type) {
  return unwrap(type).isa<CIRRecvContextType>();
}

MlirType mlirCirBinaryBuilderTypeGet(MlirContext ctx) {
  return wrap(CIRBinaryBuilderType::get(unwrap(ctx)));
}

bool mlirCirIsABinaryBuilderType(MlirType type) {
  return unwrap(type).isa<CIRBinaryBuilderType>();
}

MlirType mlirCirMatchContextTypeGet(MlirContext ctx) {
  return wrap(CIRMatchContextType::get(unwrap(ctx)));
}

bool mlirCirIsAMatchContextType(MlirType type) {
  return unwrap(type).isa<CIRMatchContextType>();
}

MlirType mlirCirProcessTypeGet(MlirContext ctx) {
  return wrap(CIRProcessType::get(unwrap(ctx)));
}

bool mlirCirIsAProcessType(MlirType type) {
  return unwrap(type).isa<CIRProcessType>();
}
