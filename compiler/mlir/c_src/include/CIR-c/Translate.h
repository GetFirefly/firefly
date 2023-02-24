#pragma once

#include "mlir-c/IR.h"
#include "llvm-c/Types.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED LLVMModuleRef mlirTranslateModuleToLLVMIR(
    MlirModule module, LLVMContextRef llvmContext, MlirStringRef name);

#ifdef __cplusplus
}
#endif
