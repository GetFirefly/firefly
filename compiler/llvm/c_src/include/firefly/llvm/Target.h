#pragma once

#include "firefly/llvm/CAPI.h"

#include "llvm-c/Core.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Support/CodeGen.h"

#if defined(_WIN32)
#include "windows.h"
#endif

namespace llvm {
class TargetMachine;
class DataLayout;
} // namespace llvm

extern "C" {
DEFINE_C_API_STRUCT(LLVMTargetMachineRef, void);

struct TargetMachineConfig {
  MlirStringRef triple;
  MlirStringRef cpu;
  MlirStringRef abi;
  const MlirStringRef *features;
  unsigned featuresLen;
  bool relaxELFRelocations;
  bool positionIndependentCode;
  bool dataSections;
  bool functionSections;
  bool emitStackSizeSection;
  bool preserveAsmComments;
  bool enableThreading;
  llvm::CodeModel::Model *codeModel;
  llvm::Reloc::Model *relocModel;
  llvm::CodeGenOpt::Level optLevel;
};
}

DEFINE_C_API_PTR_METHODS(LLVMTargetMachineRef, llvm::TargetMachine);

extern "C" {
MLIR_CAPI_EXPORTED bool LLVMFireflyHasFeature(LLVMTargetMachineRef tm,
                                              const char *feature);
MLIR_CAPI_EXPORTED void PrintTargetCPUs(LLVMTargetMachineRef tm);
MLIR_CAPI_EXPORTED void PrintTargetFeatures(LLVMTargetMachineRef tm);
MLIR_CAPI_EXPORTED LLVMTargetMachineRef
LLVMFireflyCreateTargetMachine(TargetMachineConfig *conf, char **error);

#if defined(_WIN32)
MLIR_CAPI_EXPORTED bool LLVMTargetMachineEmitToFileDescriptor(
    LLVMTargetMachineRef t, LLVMModuleRef m, HANDLE handle,
    llvm::CodeGenFileType codegen, char **errorMessage);
#else
MLIR_CAPI_EXPORTED bool
LLVMTargetMachineEmitToFileDescriptor(LLVMTargetMachineRef t, LLVMModuleRef m,
                                      int fd, llvm::CodeGenFileType codegen,
                                      char **errorMessage);
#endif

MLIR_CAPI_EXPORTED void LLVM_InitializeAllTargetInfos(void);

MLIR_CAPI_EXPORTED void LLVM_InitializeAllTargets(void);

MLIR_CAPI_EXPORTED void LLVM_InitializeAllTargetMCs(void);

MLIR_CAPI_EXPORTED void LLVM_InitializeAllAsmPrinters(void);

MLIR_CAPI_EXPORTED void LLVM_InitializeAllAsmParsers(void);

MLIR_CAPI_EXPORTED void LLVM_InitializeAllDisassemblers(void);

/* These functions return true on failure. */

MLIR_CAPI_EXPORTED LLVMBool LLVM_InitializeNativeTarget(void);

MLIR_CAPI_EXPORTED LLVMBool LLVM_InitializeNativeAsmParser(void);

MLIR_CAPI_EXPORTED LLVMBool LLVM_InitializeNativeAsmPrinter(void);

MLIR_CAPI_EXPORTED LLVMBool LLVM_InitializeNativeDisassembler(void);

MLIR_CAPI_EXPORTED LLVMTargetDataRef
LLVMCreateTargetDataLayout(LLVMTargetMachineRef tm);

MLIR_CAPI_EXPORTED void LLVMDisposeTargetMachine(LLVMTargetMachineRef tm);

MLIR_CAPI_EXPORTED char *LLVMGetHostCPUName(void);

MLIR_CAPI_EXPORTED char *LLVMGetTargetMachineTriple(LLVMTargetMachineRef tm);
}
