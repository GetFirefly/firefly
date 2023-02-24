#pragma once

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED MlirStringRef mlirOperationGetDialectName(MlirOperation op);

#ifdef __cplusplus
}
#endif
