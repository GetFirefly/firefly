#pragma once

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "llvm-c/Types.h"

#if defined(_WIN32)
#include "windows.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED MlirStringRef mlirModuleGetName(MlirModule module);
MLIR_CAPI_EXPORTED MlirModule mlirModuleClone(MlirModule module);
MLIR_CAPI_EXPORTED bool mlirOperationIsAModule(MlirOperation op);
MLIR_CAPI_EXPORTED MlirModule mlirParseFile(MlirContext context,
                                            MlirStringRef path);

#if defined(_WIN32)
MLIR_CAPI_EXPORTED bool LLVMEmitToFileDescriptor(LLVMModuleRef m, HANDLE handle,
                                                 char **errorMessage);
MLIR_CAPI_EXPORTED bool LLVMEmitBitcodeToFileDescriptor(LLVMModuleRef m,
                                                        HANDLE handle,
                                                        char **errorMessage);
MLIR_CAPI_EXPORTED bool MLIREmitToFileDescriptor(MlirModule m, HANDLE handle,
                                                 char **errorMessage);
#else
MLIR_CAPI_EXPORTED bool LLVMEmitToFileDescriptor(LLVMModuleRef m, int fd,
                                                 char **errorMessage);
MLIR_CAPI_EXPORTED bool LLVMEmitBitcodeToFileDescriptor(LLVMModuleRef m, int fd,
                                                        char **errorMessage);
MLIR_CAPI_EXPORTED bool MLIREmitToFileDescriptor(MlirModule m, int fd,
                                                 char **errorMessage);
#endif

MLIR_CAPI_EXPORTED LLVMMemoryBufferRef MLIREmitToMemoryBuffer(MlirModule m);

#ifdef __cplusplus
}
#endif
