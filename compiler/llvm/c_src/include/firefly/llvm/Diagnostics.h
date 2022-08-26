#ifndef FIREFLY_DIAGNOSTICS_H
#define FIREFLY_DIAGNOSTICS_H

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

#endif
