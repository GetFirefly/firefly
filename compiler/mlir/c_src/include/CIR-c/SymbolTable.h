#pragma once

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

enum MlirVisibility {
  MlirVisibilityPublic,
  MlirVisibilityPrivate,
  MlirVisibilityNested,
};
typedef enum MlirVisibility MlirVisibility;

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED MlirOperation
mlirSymbolTableGetNearestSymbolTable(MlirOperation from);
MLIR_CAPI_EXPORTED MlirOperation mlirSymbolTableLookupNearestSymbolFrom(
    MlirOperation from, MlirStringRef symbol);
MLIR_CAPI_EXPORTED MlirOperation mlirSymbolTableLookupIn(MlirOperation in,
                                                         MlirStringRef symbol);
MLIR_CAPI_EXPORTED int mlirSymbolTableSymbolKnownUseEmpty(MlirStringRef symbol,
                                                          MlirOperation from);
MLIR_CAPI_EXPORTED MlirVisibility
mlirSymbolTableGetSymbolVisibility(MlirOperation symbol);
MLIR_CAPI_EXPORTED void
mlirSymbolTableSetSymbolVisibility(MlirOperation symbol,
                                   MlirVisibility visibility);
MLIR_CAPI_EXPORTED MlirStringRef
mlirSymbolTableGetSymbolName(MlirOperation symbol);
MLIR_CAPI_EXPORTED void mlirSymbolTableSetSymbolName(MlirOperation symbol,
                                                     MlirStringRef name);
MLIR_CAPI_EXPORTED MlirStringRef mlirSymbolTableGetSymbolAttrName();
MLIR_CAPI_EXPORTED MlirStringRef mlirSymbolTableGetVisibilityAttrName();

#ifdef __cplusplus
}
#endif
