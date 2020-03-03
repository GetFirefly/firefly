#ifndef LUMEN_DIAGNOSTICS_H
#define LUMEN_DIAGNOSTICS_H

#include "llvm/IR/DiagnosticInfo.h"

namespace llvm {
enum DiagnosticKind;
}

namespace lumen {
enum class DiagnosticKind {
  Other,
  InlineAsm,
  StackSize,
  DebugMetadataVersion,
  SampleProfile,
  OptimizationRemark,
  OptimizationRemarkMissed,
  OptimizationRemarkAnalysis,
  OptimizationRemarkAnalysisFPCommute,
  OptimizationRemarkAnalysisAliasing,
  OptimizationRemarkOther,
  OptimizationFailure,
  PGOProfile,
  Linker,
};

static DiagnosticKind toDiagnosticKind(llvm::DiagnosticKind Kind);

}  // namespace lumen

#endif
