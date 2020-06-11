#ifndef LUMEN_DIAGNOSTICS_H
#define LUMEN_DIAGNOSTICS_H

#include "llvm/IR/DiagnosticInfo.h"

namespace llvm {
enum DiagnosticKind;
}

namespace lumen {
enum class DiagnosticKind {
  InlineAsm,
  ResourceLimit,
  StackSize,
  Linker,
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
  MisExpect,
  Unsupported,
  Other,
};

static DiagnosticKind toDiagnosticKind(llvm::DiagnosticKind Kind);

}  // namespace lumen

#endif
