#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/Twine.h"

#include "CIR/Builder.h"

using namespace mlir;
using namespace mlir::cir;

using ::llvm::ArrayRef;
using ::llvm::SmallVector;
using ::llvm::StringRef;
using ::llvm::Twine;

extern "C" MlirOperation mlirOperationGetParentModule(MlirOperation op) {
  auto mod = unwrap(op)->getParentOfType<ModuleOp>();
  if (mod)
    return wrap(mod.getOperation());
  return MlirOperation{nullptr};
}

extern "C" MlirBuilder mlirBuilderNewFromContext(MlirContext context) {
  return wrap(new mlir::Builder(unwrap(context)));
}

extern "C" MlirBuilder mlirBuilderNewFromModule(MlirModule module) {
  return wrap(new mlir::Builder(unwrap(module)));
}

extern "C" MlirContext mlirBuilderGetContext(MlirBuilder builder) {
  return wrap(unwrap(builder)->getContext());
}

extern "C" MlirLocation mlirBuilderGetUnknownLoc(MlirBuilder builder) {
  return wrap(unwrap(builder)->getUnknownLoc());
}

extern "C" MlirLocation mlirBuilderGetFileLineColLoc(MlirBuilder builder,
                                                     MlirStringRef filename,
                                                     unsigned line,
                                                     unsigned column) {
  auto file = unwrap(builder)->getStringAttr(unwrap(filename));
  Location loc = FileLineColLoc::get(file, line, column);
  return wrap(loc);
}

extern "C" MlirLocation mlirBuilderGetFusedLoc(MlirBuilder builder,
                                               size_t numLocs,
                                               MlirLocation *locs) {
  SmallVector<Location, 1> locStorage;
  auto locations = unwrapList(numLocs, locs, locStorage);
  return wrap(unwrap(builder)->getFusedLoc(locations));
}

extern "C" MlirType mlirBuilderGetF64Type(MlirBuilder builder) {
  return wrap((Type)unwrap(builder)->getF64Type());
}

extern "C" MlirType mlirBuilderGetIndexType(MlirBuilder builder) {
  return wrap((Type)unwrap(builder)->getIndexType());
}

extern "C" MlirType mlirBuilderGetI1Type(MlirBuilder builder) {
  return wrap((Type)unwrap(builder)->getI1Type());
}

extern "C" MlirType mlirBuilderGetI32Type(MlirBuilder builder) {
  return wrap((Type)unwrap(builder)->getI32Type());
}

extern "C" MlirType mlirBuilderGetI64Type(MlirBuilder builder) {
  return wrap((Type)unwrap(builder)->getI64Type());
}

extern "C" MlirType mlirBuilderGetIntegerType(MlirBuilder builder,
                                              unsigned width) {
  return wrap((Type)unwrap(builder)->getIntegerType(width));
}

extern "C" MlirType mlirBuilderGetSignedIntegerType(MlirBuilder builder,
                                                    unsigned width) {
  return wrap((Type)unwrap(builder)->getIntegerType(width, /*isSigned=*/true));
}

extern "C" MlirType mlirBuilderGetUnsignedIntegerType(MlirBuilder builder,
                                                      unsigned width) {
  return wrap((Type)unwrap(builder)->getIntegerType(width, /*isSigned=*/false));
}

extern "C" MlirType mlirBuilderGetFunctionType(MlirBuilder builder,
                                               intptr_t numInputs,
                                               MlirType *inputs,
                                               intptr_t numResults,
                                               MlirType *results) {
  SmallVector<Type, 1> inStorage;
  SmallVector<Type, 1> outStorage;
  auto ins = unwrapList(numInputs, inputs, inStorage);
  auto outs = unwrapList(numResults, results, outStorage);
  return wrap((Type)unwrap(builder)->getFunctionType(ins, outs));
}

extern "C" MlirType mlirBuilderGetNoneType(MlirBuilder builder) {
  return wrap((Type)unwrap(builder)->getNoneType());
}

extern "C" MlirNamedAttribute mlirBuilderGetNamedAttr(MlirBuilder builder,
                                                      MlirStringRef name,
                                                      MlirAttribute val) {
  NamedAttribute attr =
      unwrap(builder)->getNamedAttr(unwrap(name), unwrap(val));
  return MlirNamedAttribute{wrap(attr.getName()), wrap(attr.getValue())};
}

extern "C" MlirAttribute mlirBuilderGetUnitAttr(MlirBuilder builder) {
  return wrap((Attribute)unwrap(builder)->getUnitAttr());
}

extern "C" MlirAttribute mlirBuilderGetBoolAttr(MlirBuilder builder,
                                                int value) {
  return wrap((Attribute)unwrap(builder)->getBoolAttr(value));
}

extern "C" MlirAttribute mlirBuilderGetIntegerAttr(MlirBuilder builder,
                                                   MlirType ty, int64_t value) {
  return wrap((Attribute)unwrap(builder)->getIntegerAttr(unwrap(ty), value));
}

extern "C" MlirAttribute mlirBuilderGetI8Attr(MlirBuilder builder,
                                              int8_t value) {
  return wrap((Attribute)unwrap(builder)->getI8IntegerAttr(value));
}

extern "C" MlirAttribute mlirBuilderGetI16Attr(MlirBuilder builder,
                                               int16_t value) {
  return wrap((Attribute)unwrap(builder)->getI16IntegerAttr(value));
}

extern "C" MlirAttribute mlirBuilderGetI32Attr(MlirBuilder builder,
                                               int32_t value) {
  return wrap((Attribute)unwrap(builder)->getI32IntegerAttr(value));
}

extern "C" MlirAttribute mlirBuilderGetI64Attr(MlirBuilder builder,
                                               int64_t value) {
  return wrap((Attribute)unwrap(builder)->getI64IntegerAttr(value));
}

extern "C" MlirAttribute mlirBuilderGetIndexAttr(MlirBuilder builder,
                                                 int64_t value) {
  return wrap((Attribute)unwrap(builder)->getIndexAttr(value));
}

extern "C" MlirAttribute mlirBuilderGetFloatAttr(MlirBuilder builder,
                                                 MlirType ty, double value) {
  return wrap((Attribute)unwrap(builder)->getFloatAttr(unwrap(ty), value));
}

extern "C" MlirAttribute mlirBuilderGetStringAttr(MlirBuilder builder,
                                                  MlirStringRef bytes) {
  return wrap((Attribute)unwrap(builder)->getStringAttr(unwrap(bytes)));
}

extern "C" MlirAttribute mlirBuilderGetArrayAttr(MlirBuilder builder,
                                                 intptr_t numElements,
                                                 MlirAttribute *elements) {
  SmallVector<Attribute, 1> storage;
  auto els = unwrapList(numElements, elements, storage);
  return wrap((Attribute)unwrap(builder)->getArrayAttr(els));
}

extern "C" MlirAttribute
mlirBuilderGetFlatSymbolRefAttrByOperation(MlirBuilder builder,
                                           MlirOperation op) {
  FlatSymbolRefAttr sym = SymbolRefAttr::get(unwrap(op));
  return wrap((Attribute)sym);
}

extern "C" MlirAttribute
mlirBuilderGetFlatSymbolRefAttrByName(MlirBuilder builder,
                                      MlirStringRef symbol) {
  FlatSymbolRefAttr sym =
      SymbolRefAttr::get(unwrap(builder)->getContext(), unwrap(symbol));
  return wrap((Attribute)sym);
}

extern "C" MlirAttribute mlirBuilderGetSymbolRefAttr(MlirBuilder builder,
                                                     MlirStringRef value,
                                                     intptr_t numNested,
                                                     MlirAttribute *nested) {
  SmallVector<FlatSymbolRefAttr, 1> storage;
  if (numNested > 0) {
    storage.reserve(numNested);
    for (auto i = 0; i < numNested; ++i) {
      auto attr = unwrap(*(nested + i));
      if (!attr.isa<FlatSymbolRefAttr>())
        llvm::report_fatal_error(
            "invalid nested symbol, expected FlatSymbolRefAttr");
      storage.push_back(attr.cast<FlatSymbolRefAttr>());
    }
  }
  auto attr =
      SymbolRefAttr::get(unwrap(builder)->getContext(), unwrap(value), storage);
  return wrap((Attribute)attr);
}

extern "C" MlirOpBuilder mlirOpBuilderNewFromContext(MlirContext context) {
  return wrap(new mlir::OpBuilder(unwrap(context)));
}

extern "C" MlirModule mlirOpBuilderCreateModule(MlirOpBuilder builder,
                                                MlirLocation loc,
                                                MlirStringRef name) {
  auto moduleOp =
      unwrap(builder)->create<mlir::ModuleOp>(unwrap(loc), unwrap(name));
  return MlirModule{moduleOp.getOperation()};
}

extern "C" MlirOpBuilder mlirOpBuilderAtBlockBegin(MlirBlock block) {
  auto *blk = unwrap(block);
  return wrap(new mlir::OpBuilder(blk, blk->begin()));
}

extern "C" MlirOpBuilder mlirOpBuilderAtBlockEnd(MlirBlock block) {
  auto *blk = unwrap(block);
  return wrap(new mlir::OpBuilder(blk, blk->end()));
}

extern "C" MlirOpBuilder mlirOpBuilderAtBlockTerminator(MlirBlock block) {
  auto *blk = unwrap(block);
  auto *terminator = blk->getTerminator();
  if (terminator == nullptr)
    return MlirOpBuilder{nullptr};
  return wrap(new mlir::OpBuilder(blk, Block::iterator(terminator)));
}

extern "C" MlirInsertPoint
mlirOpBuilderSaveInsertionPoint(MlirOpBuilder builder) {
  auto *opBuilder = unwrap(builder);
  auto *iblock = opBuilder->getInsertionBlock();
  auto ipoint = opBuilder->getInsertionPoint();
  return wrap(new cir::InsertPoint(iblock, ipoint));
}

extern "C" void mlirOpBuilderRestoreInsertionPoint(MlirOpBuilder builder,
                                                   MlirInsertPoint ip) {
  auto insertPoint = unwrap(ip);
  auto *block = insertPoint->block;
  auto point = insertPoint->point;
  unwrap(builder)->setInsertionPoint(block, point);
  delete insertPoint;
}

extern "C" void mlirOpBuilderSetInsertionPoint(MlirOpBuilder builder,
                                               MlirOperation op) {
  unwrap(builder)->setInsertionPoint(unwrap(op));
}

extern "C" void mlirOpBuilderSetInsertionPointAfter(MlirOpBuilder builder,
                                                    MlirOperation op) {
  unwrap(builder)->setInsertionPointAfter(unwrap(op));
}

extern "C" void mlirOpBuilderSetInsertionPointAfterValue(MlirOpBuilder builder,
                                                         MlirValue val) {
  unwrap(builder)->setInsertionPointAfterValue(unwrap(val));
}

extern "C" void mlirOpBuilderSetInsertionPointToStart(MlirOpBuilder builder,
                                                      MlirBlock block) {
  unwrap(builder)->setInsertionPointToStart(unwrap(block));
}

extern "C" void mlirOpBuilderSetInsertionPointToEnd(MlirOpBuilder builder,
                                                    MlirBlock block) {
  unwrap(builder)->setInsertionPointToEnd(unwrap(block));
}

extern "C" MlirBlock mlirOpBuilderGetInsertionBlock(MlirOpBuilder builder) {
  return wrap(unwrap(builder)->getInsertionBlock());
}

extern "C" MlirBlock mlirOpBuilderCreateBlock(MlirOpBuilder builder,
                                              MlirRegion parent,
                                              intptr_t numArgs, MlirType *args,
                                              MlirLocation *locs) {
  SmallVector<Type, 1> storage;
  SmallVector<Location, 1> locStorage;
  auto argTypes = unwrapList(numArgs, args, storage);
  auto locations = unwrapList(numArgs, locs, locStorage);
  return wrap(unwrap(builder)->createBlock(unwrap(parent), Region::iterator(),
                                           argTypes, locations));
}

extern "C" MlirBlock mlirOpBuilderCreateBlockBefore(MlirOpBuilder builder,
                                                    MlirBlock before,
                                                    intptr_t numArgs,
                                                    MlirType *args,
                                                    MlirLocation *locs) {
  SmallVector<Type, 1> storage;
  SmallVector<Location, 1> locStorage;
  auto argTypes = unwrapList(numArgs, args, storage);
  auto locations = unwrapList(numArgs, locs, locStorage);
  return wrap(
      unwrap(builder)->createBlock(unwrap(before), argTypes, locations));
}

extern "C" MlirOperation mlirOpBuilderInsertOperation(MlirOpBuilder builder,
                                                      MlirOperation op) {
  return wrap(unwrap(builder)->insert(unwrap(op)));
}

extern "C" MlirBlock mlirBlockSplitBefore(MlirBlock blk, MlirOperation op) {
  Block *block = unwrap(blk);
  return wrap(block->splitBlock(unwrap(op)));
}

extern "C" MlirOperation
mlirOpBuilderCreateOperation(MlirOpBuilder builder,
                             const MlirOperationState *opState) {
  assert(opState && "invalid operation state");

  OperationState state(unwrap(opState->location), unwrap(opState->name));
  if (!state.name.isRegistered()) {
    Twine reason = Twine("Building op `") + state.name.getStringRef().str() +
                   "` but it isn't registered in this MLIRContext";
    llvm::report_fatal_error(reason);
  }

  SmallVector<Type, 4> resultStorage;
  SmallVector<Value, 8> operandStorage;
  SmallVector<Block *, 2> successorStorage;
  state.addTypes(
      unwrapList(opState->nResults, opState->results, resultStorage));
  state.addOperands(
      unwrapList(opState->nOperands, opState->operands, operandStorage));
  state.addSuccessors(
      unwrapList(opState->nSuccessors, opState->successors, successorStorage));

  state.attributes.reserve(opState->nAttributes);
  for (intptr_t i = 0; i < opState->nAttributes; ++i)
    state.addAttribute(unwrap(opState->attributes[i].name),
                       unwrap(opState->attributes[i].attribute));

  for (intptr_t i = 0; i < opState->nRegions; ++i)
    state.addRegion(std::unique_ptr<Region>(unwrap(opState->regions[i])));

  MlirOperation result = wrap(unwrap(builder)->create(state));
  free(opState->results);
  free(opState->operands);
  free(opState->successors);
  free(opState->regions);
  free(opState->attributes);
  return result;
}

extern "C" MlirOperation mlirOpBuilderCloneOperation(MlirOpBuilder builder,
                                                     MlirOperation op) {
  auto *operation = unwrap(op);
  return wrap(unwrap(builder)->clone(*operation));
}

extern "C" void mlirOpBuilderDestroy(MlirOpBuilder builder) {
  delete unwrap(builder);
}
