#pragma once

#include "mlir-c/Support.h"
#include "llvm-c/Types.h"
#include "llvm/IR/DiagnosticInfo.h"

namespace llvm {
enum DiagnosticKind;
}

namespace firefly {
enum class DiagnosticKind {
  InlineAsm,
  ResourceLimit,
  StackSize,
  Linker,
  Lowering,
  DebugMetadataVersion,
  DebugMetadataInvalid,
  ISelFallback,
  SampleProfile,
  OptimizationRemark,
  OptimizationRemarkMissed,
  OptimizationRemarkAnalysis,
  OptimizationRemarkAnalysisFPCommute,
  OptimizationRemarkAnalysisAliasing,
  OptimizationFailure,
  MachineOptimizationRemark,
  MachineOptimizationRemarkAnalysis,
  MachineOptimizationRemarkMissed,
  MIRParser,
  PGOProfile,
  Unsupported,
  SrcMgr,
  DontCall,
  Other,
};

static DiagnosticKind toDiagnosticKind(llvm::DiagnosticKind Kind);

} // namespace firefly

extern "C" {

MLIR_CAPI_EXPORTED firefly::DiagnosticKind
LLVMFireflyGetDiagInfoKind(LLVMDiagnosticInfoRef di);

MLIR_CAPI_EXPORTED bool
LLVMFireflyOptimizationDiagnosticIsVerbose(LLVMDiagnosticInfoRef d);

MLIR_CAPI_EXPORTED MlirStringRef
LLVMFireflyOptimizationDiagnosticPassName(LLVMDiagnosticInfoRef d);

MLIR_CAPI_EXPORTED MlirStringRef
LLVMFireflyOptimizationDiagnosticRemarkName(LLVMDiagnosticInfoRef d);

MLIR_CAPI_EXPORTED const char *
LLVMFireflyOptimizationDiagnosticMessage(LLVMDiagnosticInfoRef d);

MLIR_CAPI_EXPORTED LLVMValueRef
LLVMFireflyOptimizationDiagnosticCodeRegion(LLVMDiagnosticInfoRef d);

MLIR_CAPI_EXPORTED LLVMValueRef
LLVMFireflyDiagnosticWithLocFunction(LLVMDiagnosticInfoRef d);

MLIR_CAPI_EXPORTED bool
LLVMFireflyDiagnosticWithLocSourceLoc(LLVMDiagnosticInfoRef d,
                                      MlirStringRef *relativePath,
                                      unsigned *line, unsigned *col);

MLIR_CAPI_EXPORTED LLVMValueRef
LLVMFireflyISelFallbackDiagnosticFunction(LLVMDiagnosticInfoRef di);

MLIR_CAPI_EXPORTED const char *
LLVMFireflyUnsupportedDiagnosticMessage(LLVMDiagnosticInfoRef di);
}
