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

MlirOperation mlirOperationGetParentModule(MlirOperation op) {
  auto mod = unwrap(op)->getParentOfType<ModuleOp>();
  if (mod)
    return wrap(mod.getOperation());
  return MlirOperation{nullptr};
}

MlirBuilder mlirBuilderNewFromContext(MlirContext context) {
  return wrap(new mlir::Builder(unwrap(context)));
}

MlirBuilder mlirBuilderNewFromModule(MlirModule module) {
  return wrap(new mlir::Builder(unwrap(module)));
}

MlirContext mlirBuilderGetContext(MlirBuilder builder) {
  return wrap(unwrap(builder)->getContext());
}

MlirLocation mlirBuilderGetUnknownLoc(MlirBuilder builder) {
  return wrap(unwrap(builder)->getUnknownLoc());
}

MlirLocation mlirBuilderGetFileLineColLoc(MlirBuilder builder,
                                          MlirStringRef filename, unsigned line,
                                          unsigned column) {
  auto file = unwrap(builder)->getStringAttr(unwrap(filename));
  Location loc = FileLineColLoc::get(file, line, column);
  return wrap(loc);
}

MlirLocation mlirBuilderGetFusedLoc(MlirBuilder builder, size_t numLocs,
                                    MlirLocation *locs) {
  SmallVector<Location, 1> locStorage;
  auto locations = unwrapList(numLocs, locs, locStorage);
  return wrap(unwrap(builder)->getFusedLoc(locations));
}

MlirType mlirBuilderGetF64Type(MlirBuilder builder) {
  return wrap((Type)unwrap(builder)->getF64Type());
}

MlirType mlirBuilderGetIndexType(MlirBuilder builder) {
  return wrap((Type)unwrap(builder)->getIndexType());
}

MlirType mlirBuilderGetI1Type(MlirBuilder builder) {
  return wrap((Type)unwrap(builder)->getI1Type());
}

MlirType mlirBuilderGetI32Type(MlirBuilder builder) {
  return wrap((Type)unwrap(builder)->getI32Type());
}

MlirType mlirBuilderGetI64Type(MlirBuilder builder) {
  return wrap((Type)unwrap(builder)->getI64Type());
}

MlirType mlirBuilderGetIntegerType(MlirBuilder builder, unsigned width) {
  return wrap((Type)unwrap(builder)->getIntegerType(width));
}

MlirType mlirBuilderGetSignedIntegerType(MlirBuilder builder, unsigned width) {
  return wrap((Type)unwrap(builder)->getIntegerType(width, /*isSigned=*/true));
}

MlirType mlirBuilderGetUnsignedIntegerType(MlirBuilder builder,
                                           unsigned width) {
  return wrap((Type)unwrap(builder)->getIntegerType(width, /*isSigned=*/false));
}

MlirType mlirBuilderGetFunctionType(MlirBuilder builder, intptr_t numInputs,
                                    MlirType *inputs, intptr_t numResults,
                                    MlirType *results) {
  SmallVector<Type, 1> inStorage;
  SmallVector<Type, 1> outStorage;
  auto ins = unwrapList(numInputs, inputs, inStorage);
  auto outs = unwrapList(numResults, results, outStorage);
  return wrap((Type)unwrap(builder)->getFunctionType(ins, outs));
}

MlirType mlirBuilderGetNoneType(MlirBuilder builder) {
  return wrap((Type)unwrap(builder)->getNoneType());
}

MlirAttribute mlirBuilderGetUnitAttr(MlirBuilder builder) {
  return wrap((Attribute)unwrap(builder)->getUnitAttr());
}

MlirAttribute mlirBuilderGetBoolAttr(MlirBuilder builder, int value) {
  return wrap((Attribute)unwrap(builder)->getBoolAttr(value));
}

MlirAttribute mlirBuilderGetIntegerAttr(MlirBuilder builder, MlirType ty,
                                        int64_t value) {
  return wrap((Attribute)unwrap(builder)->getIntegerAttr(unwrap(ty), value));
}

MlirAttribute mlirBuilderGetI8Attr(MlirBuilder builder, int8_t value) {
  return wrap((Attribute)unwrap(builder)->getI8IntegerAttr(value));
}

MlirAttribute mlirBuilderGetI16Attr(MlirBuilder builder, int16_t value) {
  return wrap((Attribute)unwrap(builder)->getI16IntegerAttr(value));
}

MlirAttribute mlirBuilderGetI32Attr(MlirBuilder builder, int32_t value) {
  return wrap((Attribute)unwrap(builder)->getI32IntegerAttr(value));
}

MlirAttribute mlirBuilderGetI64Attr(MlirBuilder builder, int64_t value) {
  return wrap((Attribute)unwrap(builder)->getI64IntegerAttr(value));
}

MlirAttribute mlirBuilderGetIndexAttr(MlirBuilder builder, int64_t value) {
  return wrap((Attribute)unwrap(builder)->getIndexAttr(value));
}

MlirAttribute mlirBuilderGetFloatAttr(MlirBuilder builder, MlirType ty,
                                      double value) {
  return wrap((Attribute)unwrap(builder)->getFloatAttr(unwrap(ty), value));
}

MlirAttribute mlirBuilderGetStringAttr(MlirBuilder builder,
                                       MlirStringRef bytes) {
  return wrap((Attribute)unwrap(builder)->getStringAttr(unwrap(bytes)));
}

MlirAttribute mlirBuilderGetArrayAttr(MlirBuilder builder, intptr_t numElements,
                                      MlirAttribute *elements) {
  SmallVector<Attribute, 1> storage;
  auto els = unwrapList(numElements, elements, storage);
  return wrap((Attribute)unwrap(builder)->getArrayAttr(els));
}

MlirAttribute mlirBuilderGetFlatSymbolRefAttrByOperation(MlirBuilder builder,
                                                         MlirOperation op) {
  FlatSymbolRefAttr sym = SymbolRefAttr::get(unwrap(op));
  return wrap((Attribute)sym);
}

MlirAttribute mlirBuilderGetFlatSymbolRefAttrByName(MlirBuilder builder,
                                                    MlirStringRef symbol) {
  FlatSymbolRefAttr sym =
      SymbolRefAttr::get(unwrap(builder)->getContext(), unwrap(symbol));
  return wrap((Attribute)sym);
}

MlirAttribute mlirBuilderGetSymbolRefAttr(MlirBuilder builder,
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

MlirOpBuilder mlirOpBuilderNewFromContext(MlirContext context) {
  return wrap(new mlir::OpBuilder(unwrap(context)));
}

MlirModule mlirOpBuilderCreateModule(MlirOpBuilder builder, MlirLocation loc,
                                     MlirStringRef name) {
  auto moduleOp =
      unwrap(builder)->create<mlir::ModuleOp>(unwrap(loc), unwrap(name));
  return MlirModule{moduleOp.getOperation()};
}

MlirOpBuilder mlirOpBuilderAtBlockBegin(MlirBlock block) {
  auto *blk = unwrap(block);
  return wrap(new mlir::OpBuilder(blk, blk->begin()));
}

MlirOpBuilder mlirOpBuilderAtBlockEnd(MlirBlock block) {
  auto *blk = unwrap(block);
  return wrap(new mlir::OpBuilder(blk, blk->end()));
}

MlirOpBuilder mlirOpBuilderAtBlockTerminator(MlirBlock block) {
  auto *blk = unwrap(block);
  auto *terminator = blk->getTerminator();
  if (terminator == nullptr)
    return MlirOpBuilder{nullptr};
  return wrap(new mlir::OpBuilder(blk, Block::iterator(terminator)));
}

MlirInsertPoint mlirOpBuilderSaveInsertionPoint(MlirOpBuilder builder) {
  auto *opBuilder = unwrap(builder);
  auto *iblock = opBuilder->getInsertionBlock();
  auto ipoint = opBuilder->getInsertionPoint();
  return wrap(new cir::InsertPoint(iblock, ipoint));
}

void mlirOpBuilderRestoreInsertionPoint(MlirOpBuilder builder,
                                        MlirInsertPoint ip) {
  auto insertPoint = unwrap(ip);
  auto *block = insertPoint->block;
  auto point = insertPoint->point;
  unwrap(builder)->setInsertionPoint(block, point);
  delete insertPoint;
}

void mlirOpBuilderSetInsertionPoint(MlirOpBuilder builder, MlirOperation op) {
  unwrap(builder)->setInsertionPoint(unwrap(op));
}

void mlirOpBuilderSetInsertionPointAfter(MlirOpBuilder builder,
                                         MlirOperation op) {
  unwrap(builder)->setInsertionPointAfter(unwrap(op));
}

void mlirOpBuilderSetInsertionPointAfterValue(MlirOpBuilder builder,
                                              MlirValue val) {
  unwrap(builder)->setInsertionPointAfterValue(unwrap(val));
}

void mlirOpBuilderSetInsertionPointToStart(MlirOpBuilder builder,
                                           MlirBlock block) {
  unwrap(builder)->setInsertionPointToStart(unwrap(block));
}

void mlirOpBuilderSetInsertionPointToEnd(MlirOpBuilder builder,
                                         MlirBlock block) {
  unwrap(builder)->setInsertionPointToEnd(unwrap(block));
}

MlirBlock mlirOpBuilderGetInsertionBlock(MlirOpBuilder builder) {
  return wrap(unwrap(builder)->getInsertionBlock());
}

MlirBlock mlirOpBuilderCreateBlock(MlirOpBuilder builder, MlirRegion parent,
                                   intptr_t numArgs, MlirType *args,
                                   MlirLocation *locs) {
  SmallVector<Type, 1> storage;
  SmallVector<Location, 1> locStorage;
  auto argTypes = unwrapList(numArgs, args, storage);
  auto locations = unwrapList(numArgs, locs, locStorage);
  return wrap(unwrap(builder)->createBlock(unwrap(parent), Region::iterator(),
                                           argTypes, locations));
}

MlirBlock mlirOpBuilderCreateBlockBefore(MlirOpBuilder builder,
                                         MlirBlock before, intptr_t numArgs,
                                         MlirType *args, MlirLocation *locs) {
  SmallVector<Type, 1> storage;
  SmallVector<Location, 1> locStorage;
  auto argTypes = unwrapList(numArgs, args, storage);
  auto locations = unwrapList(numArgs, locs, locStorage);
  return wrap(
      unwrap(builder)->createBlock(unwrap(before), argTypes, locations));
}

MlirOperation mlirOpBuilderInsertOperation(MlirOpBuilder builder,
                                           MlirOperation op) {
  return wrap(unwrap(builder)->insert(unwrap(op)));
}

MlirBlock mlirBlockSplitBefore(MlirBlock blk, MlirOperation op) {
  Block *block = unwrap(blk);
  return wrap(block->splitBlock(unwrap(op)));
}

MlirOperation mlirOpBuilderCreateOperation(MlirOpBuilder builder,
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

MlirOperation mlirOpBuilderCloneOperation(MlirOpBuilder builder,
                                          MlirOperation op) {
  auto *operation = unwrap(op);
  return wrap(unwrap(builder)->clone(*operation));
}

void mlirOpBuilderDestroy(MlirOpBuilder builder) { delete unwrap(builder); }
