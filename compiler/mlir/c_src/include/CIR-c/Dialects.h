#pragma once

#include "CIR-c/AtomRef.h"
#include "CIR-c/BigIntRef.h"
#include "CIR-c/BinaryEntrySpecifier.h"
#include "CIR-c/Builder.h"
#include "mlir-c/IR.h"
#include "mlir/CAPI/Registration.h"

#ifdef __cplusplus
using ::mlir::cir::AtomRef;
using ::mlir::cir::BigIntRef;
using ::mlir::cir::BinaryEntrySpecifier;

extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(CIR, cir);

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Arithmetic, arith);

MLIR_CAPI_EXPORTED MlirTypeID mlirDialectHandleGetTypeID(MlirDialect dialect);
//===----------------------------------------------------------------------===//
/// Types
//===----------------------------------------------------------------------===//

/// Maps to the Endianness enum in firefly_binary
typedef uint8_t CirEndianness;

/// Creates a cir.none type
MLIR_CAPI_EXPORTED MlirType mlirCirNoneTypeGet(MlirContext ctx);
MLIR_CAPI_EXPORTED bool mlirCirIsANoneType(MlirType type);
/// Creates a cir.term type
MLIR_CAPI_EXPORTED MlirType mlirCirTermTypeGet(MlirContext ctx);
MLIR_CAPI_EXPORTED bool mlirCirIsATermType(MlirType type);
/// Creates a cir.number type
MLIR_CAPI_EXPORTED MlirType mlirCirNumberTypeGet(MlirContext ctx);
MLIR_CAPI_EXPORTED bool mlirCirIsANumberType(MlirType type);
/// Creates a cir.int type
MLIR_CAPI_EXPORTED MlirType mlirCirIntegerTypeGet(MlirContext ctx);
MLIR_CAPI_EXPORTED bool mlirCirIsAIntegerType(MlirType type);
/// Creates a cir.float type
MLIR_CAPI_EXPORTED MlirType mlirCirFloatTypeGet(MlirContext ctx);
MLIR_CAPI_EXPORTED bool mlirCirIsAFloatType(MlirType type);
/// Creates a cir.atom type
MLIR_CAPI_EXPORTED MlirType mlirCirAtomTypeGet(MlirContext ctx);
MLIR_CAPI_EXPORTED bool mlirCirIsAAtomType(MlirType type);
/// Creates a cir.bool type
MLIR_CAPI_EXPORTED MlirType mlirCirBoolTypeGet(MlirContext ctx);
MLIR_CAPI_EXPORTED bool mlirCirIsABoolType(MlirType type);
/// Creates a cir.isize type (machine-width integer)
MLIR_CAPI_EXPORTED MlirType mlirCirIsizeTypeGet(MlirContext ctx);
MLIR_CAPI_EXPORTED bool mlirCirIsAIsizeType(MlirType type);
/// Creates a cir.bigint type (arbitrary-width integer)
MLIR_CAPI_EXPORTED MlirType mlirCirBigIntTypeGet(MlirContext ctx);
MLIR_CAPI_EXPORTED bool mlirCirIsABigIntType(MlirType type);
/// Creates a cir.nil type
MLIR_CAPI_EXPORTED MlirType mlirCirNilTypeGet(MlirContext ctx);
MLIR_CAPI_EXPORTED bool mlirCirIsANilType(MlirType type);
/// Creates a cir.cons type
MLIR_CAPI_EXPORTED MlirType mlirCirConsTypeGet(MlirContext ctx);
MLIR_CAPI_EXPORTED bool mlirCirIsAConsType(MlirType type);
/// Creates a cir.map type
MLIR_CAPI_EXPORTED MlirType mlirCirMapTypeGet(MlirContext ctx);
MLIR_CAPI_EXPORTED bool mlirCirIsAMapType(MlirType type);
/// Creates a cir.fun type (closure)
MLIR_CAPI_EXPORTED MlirType mlirCirFunTypeGet(MlirType resultTy, intptr_t arity,
                                              MlirType const *argumentTypes);
MLIR_CAPI_EXPORTED bool mlirCirIsAFunType(MlirType type);
/// Creates a cir.bits type
MLIR_CAPI_EXPORTED MlirType mlirCirBitsTypeGet(MlirContext ctx);
MLIR_CAPI_EXPORTED bool mlirCirIsABitsType(MlirType type);
/// Creates a cir.binary type
MLIR_CAPI_EXPORTED MlirType mlirCirBinaryTypeGet(MlirContext ctx);
MLIR_CAPI_EXPORTED bool mlirCirIsABinaryType(MlirType type);
/// Creates a cir.box type
MLIR_CAPI_EXPORTED MlirType mlirCirBoxTypeGet(MlirType pointee);
MLIR_CAPI_EXPORTED bool mlirCirIsABoxType(MlirType type);
/// Creates a cir.pid type
MLIR_CAPI_EXPORTED MlirType mlirCirPidTypeGet(MlirContext ctx);
MLIR_CAPI_EXPORTED bool mlirCirIsAPidType(MlirType type);
/// Creates a cir.port type
MLIR_CAPI_EXPORTED MlirType mlirCirPortTypeGet(MlirContext ctx);
MLIR_CAPI_EXPORTED bool mlirCirIsAPortType(MlirType type);
/// Creates a cir.ref type
MLIR_CAPI_EXPORTED MlirType mlirCirReferenceTypeGet(MlirContext ctx);
MLIR_CAPI_EXPORTED bool mlirCirIsAReferenceType(MlirType type);
/// Creates a cir.process type
MLIR_CAPI_EXPORTED MlirType mlirCirProcessTypeGet(MlirContext ctx);
MLIR_CAPI_EXPORTED bool mlirCirIsAProcessType(MlirType type);
/// Creates a cir.exception type
MLIR_CAPI_EXPORTED MlirType mlirCirExceptionTypeGet(MlirContext ctx);
MLIR_CAPI_EXPORTED bool mlirCirIsAExceptionType(MlirType type);
/// Creates a cir.trace type
MLIR_CAPI_EXPORTED MlirType mlirCirTraceTypeGet(MlirContext ctx);
MLIR_CAPI_EXPORTED bool mlirCirIsATraceType(MlirType type);
/// Creates a cir.ptr type
MLIR_CAPI_EXPORTED MlirType mlirCirPtrTypeGet(MlirType pointee);
MLIR_CAPI_EXPORTED bool mlirCirIsAPtrType(MlirType type);
/// Creates a cir.recv_context type
MLIR_CAPI_EXPORTED MlirType mlirCirRecvContextTypeGet(MlirContext ctx);
MLIR_CAPI_EXPORTED bool mlirCirIsARecvContextType(MlirType type);
/// Creates a cir.binary_builder type
MLIR_CAPI_EXPORTED MlirType mlirCirBinaryBuilderTypeGet(MlirContext ctx);
MLIR_CAPI_EXPORTED bool mlirCirIsABinaryBuilderType(MlirType type);
/// Creates a cir.match_context type
MLIR_CAPI_EXPORTED MlirType mlirCirMatchContextTypeGet(MlirContext ctx);
MLIR_CAPI_EXPORTED bool mlirCirIsAMatchContextType(MlirType type);
/// Creates a cir.process type
MLIR_CAPI_EXPORTED MlirType mlirCirProcessTypeGet(MlirContext ctx);
MLIR_CAPI_EXPORTED bool mlirCirIsAProcessType(MlirType type);
//===----------------------------------------------------------------------===//
/// Attributes
//===----------------------------------------------------------------------===//

/// Creates a llvm.linkage attribute
MLIR_CAPI_EXPORTED MlirAttribute mlirLLVMLinkageAttrGet(MlirContext ctx,
                                                        MlirStringRef name);
MLIR_CAPI_EXPORTED bool mlirLLVMLinkageAttrIsA(MlirAttribute attr);

/// Creates a cir.none attribute
MLIR_CAPI_EXPORTED MlirAttribute mlirCirNoneAttrGet(MlirContext ctx);
MLIR_CAPI_EXPORTED bool mlirCirNoneAttrIsA(MlirAttribute attr);

/// Creates a cir.nil attribute
MLIR_CAPI_EXPORTED MlirAttribute mlirCirNilAttrGet(MlirContext ctx);
MLIR_CAPI_EXPORTED bool mlirCirNilAttrIsA(MlirAttribute attr);

/// Creates a cir.bool attribute
MLIR_CAPI_EXPORTED MlirAttribute mlirCirBoolAttrGet(MlirContext ctx,
                                                    bool value);
MLIR_CAPI_EXPORTED bool mlirCirBoolAttrIsA(MlirAttribute attr);
MLIR_CAPI_EXPORTED bool mlirCirBoolAttrValueOf(MlirAttribute attr);

/// Creates a cir.isize attribute
MLIR_CAPI_EXPORTED MlirAttribute mlirCirIsizeAttrGet(MlirContext ctx,
                                                     uint64_t value);
MLIR_CAPI_EXPORTED bool mlirCirIsizeAttrIsA(MlirAttribute attr);
MLIR_CAPI_EXPORTED uint64_t mlirCirIsizeAttrValueOf(MlirAttribute attr);

/// Creates a cir.float attribute
MLIR_CAPI_EXPORTED MlirAttribute mlirCirFloatAttrGet(MlirContext ctx,
                                                     double value);
MLIR_CAPI_EXPORTED bool mlirCirFloatAttrIsA(MlirAttribute attr);
MLIR_CAPI_EXPORTED double mlirCirFloatAttrValueOf(MlirAttribute attr);

/// Creates a cir.atom attribute
MLIR_CAPI_EXPORTED MlirAttribute mlirCirAtomAttrGet(AtomRef atom, MlirType ty);
MLIR_CAPI_EXPORTED bool mlirCirAtomAttrIsA(MlirAttribute attr);
MLIR_CAPI_EXPORTED AtomRef mlirCirAtomAttrValueOf(MlirAttribute attr);

/// Creates a cir.bigint attribute
MLIR_CAPI_EXPORTED MlirAttribute mlirCirBigIntAttrGet(BigIntRef atom,
                                                      MlirType ty);
MLIR_CAPI_EXPORTED bool mlirCirBigIntAttrIsA(MlirAttribute attr);
MLIR_CAPI_EXPORTED BigIntRef mlirCirBigIntAttrValueOf(MlirAttribute attr);

/// Creates a cir.endianness attribute
MLIR_CAPI_EXPORTED MlirAttribute mlirCirEndiannessAttrGet(CirEndianness value,
                                                          MlirContext ctx);
MLIR_CAPI_EXPORTED bool mlirCirEndiannessAttrIsA(MlirAttribute attr);
MLIR_CAPI_EXPORTED CirEndianness
mlirCirEndiannessAttrValueOf(MlirAttribute attr);

MLIR_CAPI_EXPORTED bool mlirCirBinarySpecAttrIsA(MlirAttribute op);

MLIR_CAPI_EXPORTED MlirAttribute
mlirCirBinarySpecAttrGet(BinaryEntrySpecifier spec, MlirContext ctx);

//===----------------------------------------------------------------------===//
/// Operations
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirOperation mlirCirIsNullOp(MlirOpBuilder builder,
                                                 MlirLocation location,
                                                 MlirValue value);

MLIR_CAPI_EXPORTED MlirOperation mlirCirTruncOp(MlirOpBuilder builder,
                                                MlirLocation location,
                                                MlirValue value, MlirType ty);

MLIR_CAPI_EXPORTED MlirOperation mlirCirZExtOp(MlirOpBuilder builder,
                                               MlirLocation location,
                                               MlirValue value, MlirType ty);

MLIR_CAPI_EXPORTED MlirOperation mlirCirCastOp(MlirOpBuilder builder,
                                               MlirLocation location,
                                               MlirValue value, MlirType ty);

MLIR_CAPI_EXPORTED MlirOperation mlirCirConstantOp(MlirOpBuilder builder,
                                                   MlirLocation location,
                                                   MlirAttribute value,
                                                   MlirType ty);

MLIR_CAPI_EXPORTED MlirOperation mlirCirConstantNullOp(MlirOpBuilder builder,
                                                       MlirLocation location,
                                                       MlirType ty);

MLIR_CAPI_EXPORTED MlirOperation mlirCirAndOp(MlirOpBuilder builder,
                                              MlirLocation location,
                                              MlirValue lhs, MlirValue rhs);

MLIR_CAPI_EXPORTED MlirOperation mlirCirAndAlsoOp(MlirOpBuilder builder,
                                                  MlirLocation location,
                                                  MlirValue lhs, MlirValue rhs);

MLIR_CAPI_EXPORTED MlirOperation mlirCirOrOp(MlirOpBuilder builder,
                                             MlirLocation location,
                                             MlirValue lhs, MlirValue rhs);

MLIR_CAPI_EXPORTED MlirOperation mlirCirOrElseOp(MlirOpBuilder builder,
                                                 MlirLocation location,
                                                 MlirValue lhs, MlirValue rhs);

MLIR_CAPI_EXPORTED MlirOperation mlirCirXorOp(MlirOpBuilder builder,
                                              MlirLocation location,
                                              MlirValue lhs, MlirValue rhs);

MLIR_CAPI_EXPORTED MlirOperation mlirCirNotOp(MlirOpBuilder builder,
                                              MlirLocation location,
                                              MlirValue value);

MLIR_CAPI_EXPORTED MlirOperation mlirCirTypeOfOp(MlirOpBuilder builder,
                                                 MlirLocation location,
                                                 MlirValue value);

MLIR_CAPI_EXPORTED MlirOperation mlirCirIsListOp(MlirOpBuilder builder,
                                                 MlirLocation location,
                                                 MlirValue value);

MLIR_CAPI_EXPORTED MlirOperation mlirCirIsNonEmptyListOp(MlirOpBuilder builder,
                                                         MlirLocation location,
                                                         MlirValue value);

MLIR_CAPI_EXPORTED MlirOperation mlirCirIsNumberOp(MlirOpBuilder builder,
                                                   MlirLocation location,
                                                   MlirValue value);

MLIR_CAPI_EXPORTED MlirOperation mlirCirIsFloatOp(MlirOpBuilder builder,
                                                  MlirLocation location,
                                                  MlirValue value);

MLIR_CAPI_EXPORTED MlirOperation mlirCirIsIntegerOp(MlirOpBuilder builder,
                                                    MlirLocation location,
                                                    MlirValue value);

MLIR_CAPI_EXPORTED MlirOperation mlirCirIsIsizeOp(MlirOpBuilder builder,
                                                  MlirLocation location,
                                                  MlirValue value);

MLIR_CAPI_EXPORTED MlirOperation mlirCirIsBigIntOp(MlirOpBuilder builder,
                                                   MlirLocation location,
                                                   MlirValue value);

MLIR_CAPI_EXPORTED MlirOperation mlirCirIsAtomOp(MlirOpBuilder builder,
                                                 MlirLocation location,
                                                 MlirValue value);

MLIR_CAPI_EXPORTED MlirOperation mlirCirIsBoolOp(MlirOpBuilder builder,
                                                 MlirLocation location,
                                                 MlirValue value);

MLIR_CAPI_EXPORTED MlirOperation mlirCirIsTypeOp(MlirOpBuilder builder,
                                                 MlirLocation location,
                                                 MlirValue value, MlirType ty);

MLIR_CAPI_EXPORTED MlirOperation mlirCirIsTaggedTupleOp(MlirOpBuilder builder,
                                                        MlirLocation location,
                                                        MlirValue value,
                                                        AtomRef atom);

MLIR_CAPI_EXPORTED MlirOperation mlirCirMallocOp(MlirOpBuilder builder,
                                                 MlirLocation location,
                                                 MlirValue process,
                                                 MlirType ty);

MLIR_CAPI_EXPORTED MlirOperation mlirCirMakeFunOp(
    MlirOpBuilder builder, MlirLocation location, MlirOperation fun,
    MlirValue process, MlirValue *env, intptr_t arity);

MLIR_CAPI_EXPORTED MlirOperation mlirCirUnpackEnvOp(MlirOpBuilder builder,
                                                    MlirLocation location,
                                                    MlirValue fun,
                                                    MlirAttribute index);

MLIR_CAPI_EXPORTED MlirOperation mlirCirConsOp(MlirOpBuilder builder,
                                               MlirLocation location,
                                               MlirValue process,
                                               MlirValue head, MlirValue value);

MLIR_CAPI_EXPORTED MlirOperation mlirCirHeadOp(MlirOpBuilder builder,
                                               MlirLocation location,
                                               MlirValue cons);

MLIR_CAPI_EXPORTED MlirOperation mlirCirTailOp(MlirOpBuilder builder,
                                               MlirLocation location,
                                               MlirValue cons);

MLIR_CAPI_EXPORTED MlirOperation mlirCirSetElementOp(
    MlirOpBuilder builder, MlirLocation location, MlirValue tuple,
    MlirAttribute index, MlirValue value, bool inPlace);

MLIR_CAPI_EXPORTED MlirOperation mlirCirGetElementOp(MlirOpBuilder builder,
                                                     MlirLocation location,
                                                     MlirValue tuple,
                                                     MlirAttribute index);

MLIR_CAPI_EXPORTED MlirOperation mlirCirRaiseOp(MlirOpBuilder builder,
                                                MlirLocation location,
                                                MlirValue exceptionClass,
                                                MlirValue exceptionReason,
                                                MlirValue exceptionTrace);

MLIR_CAPI_EXPORTED MlirOperation mlirCirExceptionClassOp(MlirOpBuilder builder,
                                                         MlirLocation location,
                                                         MlirValue exception);

MLIR_CAPI_EXPORTED MlirOperation mlirCirExceptionReasonOp(MlirOpBuilder builder,
                                                          MlirLocation location,
                                                          MlirValue exception);

MLIR_CAPI_EXPORTED MlirOperation mlirCirExceptionTraceOp(MlirOpBuilder builder,
                                                         MlirLocation location,
                                                         MlirValue exception);

MLIR_CAPI_EXPORTED MlirOperation mlirCirYieldOp(MlirOpBuilder builder,
                                                MlirLocation location);

MLIR_CAPI_EXPORTED MlirOperation mlirCirRecvStartOp(MlirOpBuilder builder,
                                                    MlirLocation location,
                                                    MlirValue timeout);

MLIR_CAPI_EXPORTED MlirOperation mlirCirRecvNextOp(MlirOpBuilder builder,
                                                   MlirLocation location,
                                                   MlirValue recvContext);

MLIR_CAPI_EXPORTED MlirOperation mlirCirRecvPeekOp(MlirOpBuilder builder,
                                                   MlirLocation location,
                                                   MlirValue recvContext);

MLIR_CAPI_EXPORTED MlirOperation mlirCirRecvPopOp(MlirOpBuilder builder,
                                                  MlirLocation location,
                                                  MlirValue recvContext);

MLIR_CAPI_EXPORTED MlirOperation mlirCirRecvDoneOp(MlirOpBuilder builder,
                                                   MlirLocation location,
                                                   MlirValue recvContext);

MLIR_CAPI_EXPORTED bool mlirCirBinaryMatchStartOpIsA(MlirOperation op);
MLIR_CAPI_EXPORTED bool mlirCirBinaryMatchOpIsA(MlirOperation op);
MLIR_CAPI_EXPORTED bool mlirCirBinaryMatchSkipOpIsA(MlirOperation op);

MLIR_CAPI_EXPORTED MlirOperation mlirCirBinaryTestTailOp(MlirOpBuilder builder,
                                                         MlirLocation location,
                                                         MlirValue matchContext,
                                                         MlirAttribute size);

MLIR_CAPI_EXPORTED MlirOperation mlirCirBinaryPushOp(
    MlirOpBuilder builder, MlirLocation location, MlirValue ctx,
    BinaryEntrySpecifier spec, MlirValue value, MlirValue sizeOpt);

MLIR_CAPI_EXPORTED MlirOperation mlirCirDispatchTableOp(MlirOpBuilder builder,
                                                        MlirLocation location,
                                                        MlirStringRef module);

MLIR_CAPI_EXPORTED void
mlirCirDispatchTableAppendEntry(MlirOperation dispatchTable,
                                MlirOperation dispatchEntry);

MLIR_CAPI_EXPORTED MlirOperation mlirCirCallByOp(MlirOpBuilder bldr,
                                                 MlirLocation location,
                                                 MlirOperation op,
                                                 intptr_t argc,
                                                 MlirValue const *argv);

MLIR_CAPI_EXPORTED MlirOperation mlirCirCallIndirect(MlirOpBuilder builder,
                                                     MlirLocation location,
                                                     MlirValue callee,
                                                     intptr_t argc,
                                                     MlirValue const *argv);

MLIR_CAPI_EXPORTED MlirOperation mlirCirEnterByOp(MlirOpBuilder bldr,
                                                  MlirLocation location,
                                                  MlirOperation op,
                                                  intptr_t argc,
                                                  MlirValue const *argv);

MLIR_CAPI_EXPORTED MlirOperation mlirCirEnterIndirect(MlirOpBuilder builder,
                                                      MlirLocation location,
                                                      MlirValue callee,
                                                      intptr_t argc,
                                                      MlirValue const *argv);

// APIs for func/etc. dialect ops we use

MLIR_CAPI_EXPORTED bool mlirOperationIsAFuncOp(MlirOperation op);

MLIR_CAPI_EXPORTED MlirType mlirFuncOpGetType(MlirOperation op);

MLIR_CAPI_EXPORTED MlirOperation mlirFuncBuildFuncOp(
    MlirOpBuilder bldr, MlirLocation location, MlirStringRef name, MlirType ty,
    size_t numAttrs, MlirNamedAttribute const *attrs, size_t numArgAttrs,
    MlirAttribute const *argAttrs);

/// Convenience functionss for constructing common operations
MLIR_CAPI_EXPORTED MlirOperation mlirFuncCallByOp(MlirOpBuilder bldr,
                                                  MlirLocation location,
                                                  MlirOperation op,
                                                  intptr_t argc,
                                                  MlirValue const *argv);
MLIR_CAPI_EXPORTED MlirOperation mlirFuncCallBySymbol(MlirOpBuilder builder,
                                                      MlirLocation location,
                                                      MlirAttribute calleeAttr,
                                                      intptr_t argc,
                                                      MlirValue const *argv);
MLIR_CAPI_EXPORTED MlirOperation mlirFuncCallByName(
    MlirOpBuilder builder, MlirLocation location, MlirStringRef calleeRef,
    intptr_t resultsc, MlirType const *resultsv, intptr_t argc,
    MlirValue const *argv);
MLIR_CAPI_EXPORTED MlirOperation mlirFuncReturn(MlirOpBuilder builder,
                                                MlirLocation location,
                                                intptr_t resultsc,
                                                MlirValue const *resultsv);
MLIR_CAPI_EXPORTED MlirOperation mlirControlFlowBranch(MlirOpBuilder builder,
                                                       MlirLocation location,
                                                       MlirBlock dest,
                                                       intptr_t argc,
                                                       MlirValue const *argv);
MLIR_CAPI_EXPORTED MlirOperation mlirControlFlowCondBranch(
    MlirOpBuilder builder, MlirLocation location, MlirValue cond,
    MlirBlock trueDest, intptr_t trueArgc, MlirValue const *trueArgv,
    MlirBlock falseDest, intptr_t falseArgc, MlirValue const *falseArgv);

extern "C" {
struct SwitchArm {
  uint32_t value;
  MlirBlock dest;
  MlirValue *operands;
  intptr_t numOperands;
};
}

MLIR_CAPI_EXPORTED MlirOperation mlirControlFlowSwitchOp(
    MlirOpBuilder builder, MlirLocation location, MlirValue value,
    MlirBlock defaultDest, MlirValue *defaultOperands, intptr_t numDefault,
    SwitchArm *arms, intptr_t numArms);

MLIR_CAPI_EXPORTED MlirOperation
mlirScfIfOp(MlirOpBuilder builder, MlirLocation location, MlirType *resultTypes,
            intptr_t numResults, MlirValue cond, bool withElseRegion);

MLIR_CAPI_EXPORTED MlirOperation mlirScfYieldOp(MlirOpBuilder builder,
                                                MlirLocation location,
                                                MlirValue *results,
                                                intptr_t numResults);
MLIR_CAPI_EXPORTED MlirOperation mlirScfForOp(
    MlirOpBuilder builder, MlirLocation location, MlirValue lowerBound,
    MlirValue upperBound, MlirValue step, MlirValue *init, intptr_t numInit);

MLIR_CAPI_EXPORTED MlirOperation mlirScfExecuteRegionOp(MlirOpBuilder builder,
                                                        MlirLocation location);

MLIR_CAPI_EXPORTED bool mlirLLVMFuncOpIsA(MlirOperation op);

MLIR_CAPI_EXPORTED bool mlirLLVMConstantOpIsA(MlirOperation op);

MLIR_CAPI_EXPORTED bool mlirLLVMICmpOpIsA(MlirOperation op);

MLIR_CAPI_EXPORTED MlirAttribute
mlirLLVMICmpPredicateAttrGet(MlirContext ctx, unsigned predicate);

#ifdef __cplusplus
}
#endif
