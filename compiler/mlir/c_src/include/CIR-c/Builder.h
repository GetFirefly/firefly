#pragma once

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#define DEFINE_C_API_STRUCT(name, storage)                                     \
  struct name {                                                                \
    storage *ptr;                                                              \
  };                                                                           \
  typedef struct name name

#ifdef __cplusplus
extern "C" {
#endif

DEFINE_C_API_STRUCT(MlirBuilder, void);
DEFINE_C_API_STRUCT(MlirOpBuilder, void);
DEFINE_C_API_STRUCT(MlirInsertPoint, void);

MLIR_CAPI_EXPORTED MlirOperation mlirOperationGetParentModule(MlirOperation op);
MLIR_CAPI_EXPORTED MlirBuilder mlirBuilderNewFromContext(MlirContext context);
MLIR_CAPI_EXPORTED MlirBuilder mlirBuilderNewFromModule(MlirModule module);
MLIR_CAPI_EXPORTED MlirContext mlirBuilderGetContext(MlirBuilder builder);
MLIR_CAPI_EXPORTED MlirLocation mlirBuilderGetUnknownLoc(MlirBuilder builder);
MLIR_CAPI_EXPORTED MlirLocation
mlirBuilderGetFileLineColLoc(MlirBuilder builder, MlirStringRef filename,
                             unsigned line, unsigned column);
MLIR_CAPI_EXPORTED MlirLocation mlirBuilderGetFusedLoc(MlirBuilder builder,
                                                       size_t numLocs,
                                                       MlirLocation *locs);
MLIR_CAPI_EXPORTED MlirType mlirBuilderGetF64Type(MlirBuilder builder);
MLIR_CAPI_EXPORTED MlirType mlirBuilderGetIndexType(MlirBuilder builder);
MLIR_CAPI_EXPORTED MlirType mlirBuilderGetI1Type(MlirBuilder builder);
MLIR_CAPI_EXPORTED MlirType mlirBuilderGetI32Type(MlirBuilder builder);
MLIR_CAPI_EXPORTED MlirType mlirBuilderGetI64Type(MlirBuilder builder);
MLIR_CAPI_EXPORTED MlirType mlirBuilderGetIntegerType(MlirBuilder builder,
                                                      unsigned width);
MLIR_CAPI_EXPORTED MlirType mlirBuilderGetSignedIntegerType(MlirBuilder builder,
                                                            unsigned width);
MLIR_CAPI_EXPORTED MlirType
mlirBuilderGetUnsignedIntegerType(MlirBuilder builder, unsigned width);
MLIR_CAPI_EXPORTED MlirType mlirBuilderGetFunctionType(MlirBuilder builder,
                                                       intptr_t numInputs,
                                                       MlirType *inputs,
                                                       intptr_t numResults,
                                                       MlirType *results);
MLIR_CAPI_EXPORTED MlirType mlirBuilderGetNoneType(MlirBuilder builder);
MLIR_CAPI_EXPORTED MlirAttribute mlirBuilderGetUnitAttr(MlirBuilder builder);
MLIR_CAPI_EXPORTED MlirAttribute mlirBuilderGetBoolAttr(MlirBuilder builder,
                                                        int value);
MLIR_CAPI_EXPORTED MlirAttribute mlirBuilderGetIntegerAttr(MlirBuilder builder,
                                                           MlirType ty,
                                                           int64_t value);
MLIR_CAPI_EXPORTED MlirAttribute mlirBuilderGetI8Attr(MlirBuilder builder,
                                                      int8_t value);
MLIR_CAPI_EXPORTED MlirAttribute mlirBuilderGetI16Attr(MlirBuilder builder,
                                                       int16_t value);
MLIR_CAPI_EXPORTED MlirAttribute mlirBuilderGetI32Attr(MlirBuilder builder,
                                                       int32_t value);
MLIR_CAPI_EXPORTED MlirAttribute mlirBuilderGetI64Attr(MlirBuilder builder,
                                                       int64_t value);
MLIR_CAPI_EXPORTED MlirAttribute mlirBuilderGetIndexAttr(MlirBuilder builder,
                                                         int64_t value);
MLIR_CAPI_EXPORTED MlirAttribute mlirBuilderGetFloatAttr(MlirBuilder builder,
                                                         MlirType ty,
                                                         double value);
MLIR_CAPI_EXPORTED MlirAttribute mlirBuilderGetStringAttr(MlirBuilder builder,
                                                          MlirStringRef bytes);
MLIR_CAPI_EXPORTED MlirAttribute mlirBuilderGetArrayAttr(
    MlirBuilder builder, intptr_t numElements, MlirAttribute *elements);
MLIR_CAPI_EXPORTED MlirAttribute mlirBuilderGetFlatSymbolRefAttrByOperation(
    MlirBuilder builder, MlirOperation op);
MLIR_CAPI_EXPORTED MlirAttribute mlirBuilderGetFlatSymbolRefAttrByName(
    MlirBuilder builder, MlirStringRef symbol);
MLIR_CAPI_EXPORTED MlirAttribute
mlirBuilderGetSymbolRefAttr(MlirBuilder builder, MlirStringRef value,
                            intptr_t numNested, MlirAttribute *nested);
MLIR_CAPI_EXPORTED MlirOpBuilder
mlirOpBuilderNewFromContext(MlirContext context);
MLIR_CAPI_EXPORTED MlirModule mlirOpBuilderCreateModule(MlirOpBuilder builder,
                                                        MlirLocation loc,
                                                        MlirStringRef name);
MLIR_CAPI_EXPORTED MlirOpBuilder mlirOpBuilderAtBlockBegin(MlirBlock block);
MLIR_CAPI_EXPORTED MlirOpBuilder mlirOpBuilderAtBlockEnd(MlirBlock block);
MLIR_CAPI_EXPORTED MlirOpBuilder
mlirOpBuilderAtBlockTerminator(MlirBlock block);
MLIR_CAPI_EXPORTED MlirInsertPoint
mlirOpBuilderSaveInsertionPoint(MlirOpBuilder builder);
MLIR_CAPI_EXPORTED void
mlirOpBuilderRestoreInsertionPoint(MlirOpBuilder builder, MlirInsertPoint ip);
MLIR_CAPI_EXPORTED void mlirOpBuilderSetInsertionPoint(MlirOpBuilder builder,
                                                       MlirOperation op);
MLIR_CAPI_EXPORTED void
mlirOpBuilderSetInsertionPointAfter(MlirOpBuilder builder, MlirOperation op);
MLIR_CAPI_EXPORTED void
mlirOpBuilderSetInsertionPointAfterValue(MlirOpBuilder builder, MlirValue val);
MLIR_CAPI_EXPORTED void
mlirOpBuilderSetInsertionPointToStart(MlirOpBuilder builder, MlirBlock block);
MLIR_CAPI_EXPORTED void
mlirOpBuilderSetInsertionPointToEnd(MlirOpBuilder builder, MlirBlock block);
MLIR_CAPI_EXPORTED MlirBlock
mlirOpBuilderGetInsertionBlock(MlirOpBuilder builder);

MLIR_CAPI_EXPORTED MlirBlock mlirOpBuilderCreateBlock(MlirOpBuilder builder,
                                                      MlirRegion parent,
                                                      intptr_t numArgs,
                                                      MlirType *args,
                                                      MlirLocation *locs);
MLIR_CAPI_EXPORTED MlirBlock mlirOpBuilderCreateBlockBefore(
    MlirOpBuilder builder, MlirBlock before, intptr_t numArgs, MlirType *args,
    MlirLocation *locs);
MLIR_CAPI_EXPORTED MlirOperation
mlirOpBuilderInsertOperation(MlirOpBuilder builder, MlirOperation op);
MLIR_CAPI_EXPORTED MlirBlock mlirBlockSplitBefore(MlirBlock blk,
                                                  MlirOperation op);
MLIR_CAPI_EXPORTED MlirOperation mlirOpBuilderCreateOperation(
    MlirOpBuilder builder, const MlirOperationState *opState);
MLIR_CAPI_EXPORTED MlirOperation
mlirOpBuilderCloneOperation(MlirOpBuilder builder, MlirOperation op);
MLIR_CAPI_EXPORTED void mlirOpBuilderDestroy(MlirOpBuilder builder);

#ifdef __cplusplus
} // extern "C"
#endif

#undef DEFINE_C_API_STRUCT
