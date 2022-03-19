#include "CIR-c/Dialects.h"

#include "CIR/Attributes.h"
#include "CIR/Builder.h"
#include "CIR/Dialect.h"
#include "CIR/Ops.h"
#include "CIR/Types.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;
using namespace mlir::cir;

MlirOperation mlirCirCastOp(MlirOpBuilder bldr, MlirLocation location,
                            MlirValue value, MlirType toTy) {
  OpBuilder *builder = unwrap(bldr);
  Operation *op = builder->create<cir::CastOp>(unwrap(location), unwrap(toTy),
                                               unwrap(value));
  return wrap(op);
}

MlirOperation mlirCirConstantOp(MlirOpBuilder bldr, MlirLocation location,
                                MlirAttribute value, MlirType ty) {
  OpBuilder *builder = unwrap(bldr);
  Operation *op = builder->create<cir::ConstantOp>(unwrap(location), unwrap(ty),
                                                   unwrap(value));
  return wrap(op);
}

MlirOperation mlirCirAndOp(MlirOpBuilder bldr, MlirLocation location,
                           MlirValue lhs, MlirValue rhs) {
  OpBuilder *builder = unwrap(bldr);
  Operation *op =
      builder->create<cir::AndOp>(unwrap(location), unwrap(lhs), unwrap(rhs));
  return wrap(op);
}

MlirOperation mlirCirAndAlsoOp(MlirOpBuilder bldr, MlirLocation location,
                               MlirValue lhs, MlirValue rhs) {
  OpBuilder *builder = unwrap(bldr);
  Operation *op = builder->create<cir::AndAlsoOp>(unwrap(location), unwrap(lhs),
                                                  unwrap(rhs));
  return wrap(op);
}

MlirOperation mlirCirOrOp(MlirOpBuilder bldr, MlirLocation location,
                          MlirValue lhs, MlirValue rhs) {
  OpBuilder *builder = unwrap(bldr);
  Operation *op =
      builder->create<cir::OrOp>(unwrap(location), unwrap(lhs), unwrap(rhs));
  return wrap(op);
}

MlirOperation mlirCirOrElseOp(MlirOpBuilder bldr, MlirLocation location,
                              MlirValue lhs, MlirValue rhs) {
  OpBuilder *builder = unwrap(bldr);
  Operation *op = builder->create<cir::OrElseOp>(unwrap(location), unwrap(lhs),
                                                 unwrap(rhs));
  return wrap(op);
}

MlirOperation mlirCirXorOp(MlirOpBuilder bldr, MlirLocation location,
                           MlirValue lhs, MlirValue rhs) {
  OpBuilder *builder = unwrap(bldr);
  Operation *op =
      builder->create<cir::XorOp>(unwrap(location), unwrap(lhs), unwrap(rhs));
  return wrap(op);
}

MlirOperation mlirCirNotOp(MlirOpBuilder bldr, MlirLocation location,
                           MlirValue value) {
  OpBuilder *builder = unwrap(bldr);
  Operation *op = builder->create<cir::NotOp>(unwrap(location), unwrap(value));
  return wrap(op);
}

MlirOperation mlirCirTypeOfImmediateOp(MlirOpBuilder bldr,
                                       MlirLocation location, MlirValue value) {
  OpBuilder *builder = unwrap(bldr);
  Operation *op =
      builder->create<cir::TypeOfImmediateOp>(unwrap(location), unwrap(value));
  return wrap(op);
}

MlirOperation mlirCirTypeOfBoxOp(MlirOpBuilder bldr, MlirLocation location,
                                 MlirValue value) {
  OpBuilder *builder = unwrap(bldr);
  Operation *op =
      builder->create<cir::TypeOfBoxOp>(unwrap(location), unwrap(value));
  return wrap(op);
}

MlirOperation mlirCirTypeOfOp(MlirOpBuilder bldr, MlirLocation location,
                              MlirValue value) {
  OpBuilder *builder = unwrap(bldr);
  Operation *op =
      builder->create<cir::TypeOfOp>(unwrap(location), unwrap(value));
  return wrap(op);
}

MlirOperation mlirCirIsTypeOp(MlirOpBuilder bldr, MlirLocation location,
                              MlirValue value, MlirType ty) {

  OpBuilder *builder = unwrap(bldr);
  Operation *op = builder->create<cir::IsTypeOp>(
      unwrap(location), unwrap(value), TypeAttr::get(unwrap(ty)));
  return wrap(op);
}

MlirOperation mlirCirIsTaggedTupleOp(MlirOpBuilder bldr, MlirLocation location,
                                     MlirValue value, AtomRef atom) {

  OpBuilder *builder = unwrap(bldr);
  auto boolTy = builder->getType<CIRBoolType>();
  Operation *op = builder->create<cir::IsTaggedTupleOp>(
      unwrap(location), unwrap(value),
      AtomAttr::get(builder->getContext(), boolTy, atom));
  return wrap(op);
}

MlirOperation mlirCirMallocOp(MlirOpBuilder bldr, MlirLocation location,
                              MlirType ty) {
  OpBuilder *builder = unwrap(bldr);
  Type pointee = unwrap(ty);
  Type pointer = CIRBoxType::get(pointee);
  Operation *op =
      builder->create<cir::MallocOp>(unwrap(location), pointer, pointee);
  return wrap(op);
}

MlirOperation mlirCirCaptureFunOp(MlirOpBuilder bldr, MlirLocation location,
                                  MlirType funTy, MlirValue *env,
                                  intptr_t arity) {
  OpBuilder *builder = unwrap(bldr);
  CIRFunType calleeTy = unwrap(funTy).cast<CIRFunType>();
  SmallVector<Value, 1> operandStorage;
  ValueRange operands(unwrapList(arity, env, operandStorage));
  Operation *op =
      builder->create<cir::CaptureFunOp>(unwrap(location), calleeTy, operands);
  return wrap(op);
}

MlirOperation mlirCirConsOp(MlirOpBuilder bldr, MlirLocation location,
                            MlirValue head, MlirValue tail) {
  OpBuilder *builder = unwrap(bldr);
  Operation *op = builder->create<cir::ConsOp>(unwrap(location), unwrap(head),
                                               unwrap(tail));
  return wrap(op);
}

MlirOperation mlirCirHeadOp(MlirOpBuilder bldr, MlirLocation location,
                            MlirValue cons) {
  OpBuilder *builder = unwrap(bldr);
  Operation *op = builder->create<cir::HeadOp>(unwrap(location), unwrap(cons));
  return wrap(op);
}

MlirOperation mlirCirTailOp(MlirOpBuilder bldr, MlirLocation location,
                            MlirValue cons) {
  OpBuilder *builder = unwrap(bldr);
  Operation *op = builder->create<cir::TailOp>(unwrap(location), unwrap(cons));
  return wrap(op);
}

MlirOperation mlirCirTupleOp(MlirOpBuilder bldr, MlirLocation location,
                             intptr_t arity) {
  OpBuilder *builder = unwrap(bldr);
  Type termTy = builder->getType<CIRTermType>();
  SmallVector<Type, 2> elementTypes;
  for (auto i = 0; i < arity; i++)
    elementTypes.push_back(termTy);
  Type tupleTy = builder->getTupleType(elementTypes);
  Operation *op = builder->create<cir::TupleOp>(unwrap(location), tupleTy,
                                                (uint32_t)(arity));
  return wrap(op);
}

MlirOperation mlirCirSetElementOp(MlirOpBuilder bldr, MlirLocation location,
                                  MlirValue tuple, MlirValue index,
                                  MlirValue value) {

  OpBuilder *builder = unwrap(bldr);
  Type termTy = builder->getType<CIRTermType>();
  Operation *op = builder->create<cir::SetElementOp>(
      unwrap(location), termTy, unwrap(tuple), unwrap(index), unwrap(value));
  return wrap(op);
}

MlirOperation mlirCirGetElementOp(MlirOpBuilder bldr, MlirLocation location,
                                  MlirValue tuple, MlirValue index) {
  OpBuilder *builder = unwrap(bldr);
  Operation *op = builder->create<cir::GetElementOp>(
      unwrap(location), unwrap(tuple), unwrap(index));
  return wrap(op);
}

MlirOperation mlirCirRaiseOp(MlirOpBuilder bldr, MlirLocation location,
                             MlirValue exceptionClass,
                             MlirValue exceptionReason,
                             MlirValue exceptionTrace) {
  OpBuilder *builder = unwrap(bldr);
  Operation *op = builder->create<cir::RaiseOp>(
      unwrap(location), unwrap(exceptionClass), unwrap(exceptionReason),
      unwrap(exceptionTrace));
  return wrap(op);
}

MlirOperation mlirCirBuildStacktraceOp(MlirOpBuilder bldr,
                                       MlirLocation location) {
  OpBuilder *builder = unwrap(bldr);
  Operation *op = builder->create<cir::BuildStacktraceOp>(unwrap(location));
  return wrap(op);
}

MlirOperation mlirCirExceptionClassOp(MlirOpBuilder bldr, MlirLocation location,
                                      MlirValue exception) {
  OpBuilder *builder = unwrap(bldr);
  Operation *op = builder->create<cir::ExceptionClassOp>(unwrap(location),
                                                         unwrap(exception));
  return wrap(op);
}

MlirOperation mlirCirExceptionReasonOp(MlirOpBuilder bldr,
                                       MlirLocation location,
                                       MlirValue exception) {
  OpBuilder *builder = unwrap(bldr);
  Operation *op = builder->create<cir::ExceptionReasonOp>(unwrap(location),
                                                          unwrap(exception));
  return wrap(op);
}

MlirOperation mlirCirExceptionTraceOp(MlirOpBuilder bldr, MlirLocation location,
                                      MlirValue exception) {
  OpBuilder *builder = unwrap(bldr);
  Operation *op = builder->create<cir::ExceptionTraceOp>(unwrap(location),
                                                         unwrap(exception));
  return wrap(op);
}

MlirOperation mlirCirYieldOp(MlirOpBuilder bldr, MlirLocation location) {
  OpBuilder *builder = unwrap(bldr);
  Operation *op = builder->create<cir::YieldOp>(unwrap(location));
  return wrap(op);
}

MlirOperation mlirCirRecvStartOp(MlirOpBuilder bldr, MlirLocation location,
                                 MlirValue timeout) {
  OpBuilder *builder = unwrap(bldr);
  Operation *op =
      builder->create<cir::RecvStartOp>(unwrap(location), unwrap(timeout));
  return wrap(op);
}

MlirOperation mlirCirRecvNextOp(MlirOpBuilder bldr, MlirLocation location,
                                MlirValue recvContext) {

  OpBuilder *builder = unwrap(bldr);
  Operation *op =
      builder->create<cir::RecvNextOp>(unwrap(location), unwrap(recvContext));
  return wrap(op);
}

MlirOperation mlirCirRecvPeekOp(MlirOpBuilder bldr, MlirLocation location,
                                MlirValue recvContext) {

  OpBuilder *builder = unwrap(bldr);
  Operation *op =
      builder->create<cir::RecvPeekOp>(unwrap(location), unwrap(recvContext));
  return wrap(op);
}

MlirOperation mlirCirRecvPopOp(MlirOpBuilder bldr, MlirLocation location,
                               MlirValue recvContext) {

  OpBuilder *builder = unwrap(bldr);
  Operation *op =
      builder->create<cir::RecvPopOp>(unwrap(location), unwrap(recvContext));
  return wrap(op);
}

MlirOperation mlirCirRecvDoneOp(MlirOpBuilder bldr, MlirLocation location,
                                MlirValue recvContext) {

  OpBuilder *builder = unwrap(bldr);
  Operation *op =
      builder->create<cir::RecvDoneOp>(unwrap(location), unwrap(recvContext));
  return wrap(op);
}

MlirOperation mlirCirBinaryStartOp(MlirOpBuilder bldr, MlirLocation location) {
  OpBuilder *builder = unwrap(bldr);
  Operation *op = builder->create<cir::BinaryStartOp>(unwrap(location));
  return wrap(op);
}

MlirOperation mlirCirBinaryFinishOp(MlirOpBuilder bldr, MlirLocation location,
                                    MlirValue binBuilder) {
  OpBuilder *builder = unwrap(bldr);
  Operation *op = builder->create<cir::BinaryFinishOp>(unwrap(location),
                                                       unwrap(binBuilder));
  return wrap(op);
}

MlirOperation mlirCirBinaryPushIntegerOp(MlirOpBuilder bldr,
                                         MlirLocation location,
                                         MlirValue binBuilder, MlirValue value,
                                         MlirValue bits, bool isSigned,
                                         CirEndianness endianness,
                                         unsigned unit) {
  OpBuilder *builder = unwrap(bldr);
  auto endianAttr = EndiannessAttr::get(
      builder->getContext(), builder->getI8Type(), unwrap(endianness));
  Operation *op = builder->create<cir::BinaryPushIntegerOp>(
      unwrap(location), unwrap(binBuilder), unwrap(value), unwrap(bits),
      isSigned, endianAttr, (uint8_t)unit);
  return wrap(op);
}

MlirOperation mlirCirBinaryPushFloatOp(MlirOpBuilder bldr,
                                       MlirLocation location,
                                       MlirValue binBuilder, MlirValue value,
                                       MlirValue bits, CirEndianness endianness,
                                       unsigned unit) {
  OpBuilder *builder = unwrap(bldr);
  auto endianAttr = EndiannessAttr::get(
      builder->getContext(), builder->getI8Type(), unwrap(endianness));
  Operation *op = builder->create<cir::BinaryPushFloatOp>(
      unwrap(location), unwrap(binBuilder), unwrap(value), unwrap(bits),
      endianAttr, (uint8_t)unit);
  return wrap(op);
}

MlirOperation mlirCirBinaryPushUtf8Op(MlirOpBuilder bldr, MlirLocation location,
                                      MlirValue binBuilder, MlirValue value) {
  OpBuilder *builder = unwrap(bldr);
  Operation *op = builder->create<cir::BinaryPushUtf8Op>(
      unwrap(location), unwrap(binBuilder), unwrap(value));
  return wrap(op);
}

MlirOperation mlirCirBinaryPushUtf16Op(MlirOpBuilder bldr,
                                       MlirLocation location,
                                       MlirValue binBuilder, MlirValue value,
                                       CirEndianness endianness) {
  OpBuilder *builder = unwrap(bldr);
  auto endianAttr = EndiannessAttr::get(
      builder->getContext(), builder->getI8Type(), unwrap(endianness));
  Operation *op = builder->create<cir::BinaryPushUtf16Op>(
      unwrap(location), unwrap(binBuilder), unwrap(value), endianAttr);
  return wrap(op);
}

MlirOperation mlirCirBinaryPushBitsOp(MlirOpBuilder bldr, MlirLocation location,
                                      MlirValue binBuilder, MlirValue value,
                                      MlirValue unit, MlirValue size) {
  OpBuilder *builder = unwrap(bldr);
  Operation *op = builder->create<cir::BinaryPushBitsOp>(
      unwrap(location), unwrap(binBuilder), unwrap(value), unwrap(unit),
      unwrap(size));
  return wrap(op);
}

bool mlirOperationIsAFuncOp(MlirOperation op) {
  return isa<FuncOp>(unwrap(op));
}

MlirType mlirFuncOpGetType(MlirOperation op) {
  auto func = cast<FuncOp>(unwrap(op));
  return wrap(func.getFunctionType());
}

MlirOperation mlirFuncBuildFuncOp(MlirOpBuilder bldr, MlirLocation location,
                                  MlirStringRef name, MlirType ty,
                                  size_t numAttrs,
                                  MlirNamedAttribute const *attrs,
                                  size_t numArgAttrs,
                                  MlirAttribute const *argAttrs) {
  SmallVector<NamedAttribute, 2> attrStorage;
  if (numAttrs > 0) {
    attrStorage.reserve(numAttrs);
    for (size_t i = 0; i < numArgAttrs; ++i) {
      auto namedAttr = *(attrs + i);
      auto name = unwrap(namedAttr.name);
      auto value = unwrap(namedAttr.attribute);
      attrStorage.push_back(NamedAttribute(name, value));
    }
  }
  SmallVector<DictionaryAttr, 2> argAttrStorage;
  if (numArgAttrs > 0) {
    argAttrStorage.reserve(numArgAttrs);
    for (size_t i = 0; i < numArgAttrs; ++i)
      argAttrStorage.push_back(unwrap(*(argAttrs + i)).cast<DictionaryAttr>());
  }
  FunctionType funcTy = unwrap(ty).cast<FunctionType>();
  return wrap(unwrap(bldr)->create<FuncOp>(
      unwrap(location), unwrap(name), funcTy, attrStorage, argAttrStorage));
}

MlirOperation mlirFuncCallByOp(MlirOpBuilder bldr, MlirLocation location,
                               MlirOperation op, intptr_t argc,
                               MlirValue const *argv) {
  auto callee = cast<FuncOp>(unwrap(op));
  SmallVector<Value, 2> argStorage;
  ValueRange args(unwrapList(argc, argv, argStorage));
  return wrap(
      unwrap(bldr)->create<func::CallOp>(unwrap(location), callee, args));
}

MlirOperation mlirFuncCallBySymbol(MlirOpBuilder bldr, MlirLocation location,
                                   MlirAttribute calleeAttr, intptr_t resultsc,
                                   MlirType const *resultv, intptr_t argc,
                                   MlirValue const *argv) {
  auto callee = unwrap(calleeAttr).cast<SymbolRefAttr>();
  SmallVector<Value, 2> argStorage;
  SmallVector<Type, 2> resultStorage;
  ValueRange args(unwrapList(argc, argv, argStorage));
  TypeRange results(unwrapList(resultsc, resultv, resultStorage));
  return wrap(unwrap(bldr)->create<func::CallOp>(unwrap(location), callee,
                                                 results, args));
}

MlirOperation mlirFuncCallByName(MlirOpBuilder bldr, MlirLocation location,
                                 MlirStringRef calleeRef, intptr_t resultsc,
                                 MlirType const *resultsv, intptr_t argc,
                                 MlirValue const *argv) {
  auto callee = unwrap(calleeRef);
  SmallVector<Type, 2> resultStorage;
  SmallVector<Value, 2> argStorage;
  return wrap(unwrap(bldr)->create<func::CallOp>(
      unwrap(location), callee, unwrapList(resultsc, resultsv, resultStorage),
      unwrapList(argc, argv, argStorage)));
}

MlirOperation mlirFuncCallIndirect(MlirOpBuilder bldr, MlirLocation location,
                                   MlirValue callee, intptr_t argc,
                                   MlirValue const *argv) {
  SmallVector<Value, 2> argStorage;
  return wrap(unwrap(bldr)->create<func::CallIndirectOp>(
      unwrap(location), unwrap(callee), unwrapList(argc, argv, argStorage)));
}

MlirOperation mlirFuncReturn(MlirOpBuilder bldr, MlirLocation location,
                             intptr_t resultsc, MlirValue const *resultsv) {
  SmallVector<Value, 2> resultStorage;
  return wrap(unwrap(bldr)->create<func::ReturnOp>(
      unwrap(location), unwrapList(resultsc, resultsv, resultStorage)));
}

MlirOperation mlirControlFlowBranch(MlirOpBuilder bldr, MlirLocation location,
                                    MlirBlock dest, intptr_t argc,
                                    MlirValue const *argv) {
  SmallVector<Value, 2> destOperands;
  return wrap(unwrap(bldr)->create<cf::BranchOp>(
      unwrap(location), unwrap(dest), unwrapList(argc, argv, destOperands)));
}

MlirOperation mlirControlFlowCondBranch(MlirOpBuilder bldr,
                                        MlirLocation location, MlirValue cond,
                                        MlirBlock trueDest, intptr_t trueArgc,
                                        MlirValue const *trueArgv,
                                        MlirBlock falseDest, intptr_t falseArgc,
                                        MlirValue const *falseArgv) {
  SmallVector<Value, 2> trueOperands;
  SmallVector<Value, 2> falseOperands;
  return wrap(unwrap(bldr)->create<cf::CondBranchOp>(
      unwrap(location), unwrap(cond), unwrap(trueDest),
      unwrapList(trueArgc, trueArgv, trueOperands), unwrap(falseDest),
      unwrapList(falseArgc, falseArgv, falseOperands)));
}

MlirOperation mlirControlFlowSwitchOp(MlirOpBuilder bldr, MlirLocation location,
                                      MlirValue value, MlirBlock defaultDest,
                                      MlirValue *defaultOperands,
                                      intptr_t numDefault, SwitchArm *armsPtr,
                                      intptr_t numArms) {

  OpBuilder *builder = unwrap(bldr);

  OperationState state(unwrap(location), cf::SwitchOp::getOperationName());
  state.addOperands(unwrap(value));

  SmallVector<Value, 2> defaultStorage;
  state.addOperands(unwrapList(numDefault, defaultOperands, defaultStorage));

  SmallVector<Value, 2> operandStorage;
  SmallVector<Block *> caseDestinations;
  SmallVector<int32_t> caseValues;
  SmallVector<int32_t> rangeSegments;

  ArrayRef<SwitchArm> arms(armsPtr, numArms);
  int32_t numCaseOperands = 0;
  for (auto arm : arms) {
    state.addOperands(
        unwrapList(arm.numOperands, arm.operands, operandStorage));
    rangeSegments.push_back(arm.numOperands);
    caseValues.push_back(arm.value);
    caseDestinations.push_back(unwrap(arm.dest));
    numCaseOperands += arm.numOperands;
    operandStorage.clear();
  }
  state.addAttribute("case_operand_segments",
                     builder->getI32TensorAttr(rangeSegments));
  state.addAttribute(
      "operand_segment_sizes",
      builder->getI32VectorAttr(
          {1, static_cast<int32_t>(numDefault), numCaseOperands}));
  state.addAttribute("case_values", builder->getI32TensorAttr(caseValues));
  state.addSuccessors(unwrap(defaultDest));
  state.addSuccessors(caseDestinations);

  Operation *op = builder->create(state);
  auto result = dyn_cast<cf::SwitchOp>(op);
  assert(result && "builder didn't return the right type");
  return wrap(op);
}

MlirOperation mlirScfIfOp(MlirOpBuilder bldr, MlirLocation location,
                          MlirType *resultTypes, intptr_t numResults,
                          MlirValue cond, bool withElseRegion) {
  SmallVector<Type> resultStorage;
  OpBuilder *builder = unwrap(bldr);
  Operation *op = builder->create<scf::IfOp>(
      unwrap(location), unwrapList(numResults, resultTypes, resultStorage),
      unwrap(cond), withElseRegion);
  return wrap(op);
}

MlirOperation mlirScfYieldOp(MlirOpBuilder bldr, MlirLocation location,
                             MlirValue *results, intptr_t numResults) {
  SmallVector<Value> resultStorage;

  OpBuilder *builder = unwrap(bldr);
  Operation *op = builder->create<scf::YieldOp>(
      unwrap(location), unwrapList(numResults, results, resultStorage));
  return wrap(op);
}

MlirOperation mlirScfForOp(MlirOpBuilder bldr, MlirLocation location,
                           MlirValue lowerBound, MlirValue upperBound,
                           MlirValue step, MlirValue *init, intptr_t numInit) {

  SmallVector<Value> initStorage;

  OpBuilder *builder = unwrap(bldr);
  Operation *op = builder->create<scf::ForOp>(
      unwrap(location), unwrap(lowerBound), unwrap(upperBound), unwrap(step),
      unwrapList(numInit, init, initStorage));
  return wrap(op);
}

MlirOperation mlirScfExecuteRegionOp(MlirOpBuilder bldr, MlirLocation location,
                                     MlirType *results, intptr_t numResults) {
  OpBuilder *builder = unwrap(bldr);
  SmallVector<Type> resultStorage;
  Operation *op = builder->create<scf::ExecuteRegionOp>(
      unwrap(location), unwrapList(numResults, results, resultStorage));
  return wrap(op);
}
