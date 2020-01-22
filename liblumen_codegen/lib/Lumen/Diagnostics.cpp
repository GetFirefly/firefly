#include "lumen/Diagnostics.h"
#include "lumen/Lumen.h"
#include "lumen/Support/RustString.h"

#include "llvm-c/Core.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/Support/CBindingWrapping.h"

using namespace llvm;

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(SMDiagnostic, LLVMSMDiagnosticRef)

enum class LLVMLumenDiagnosticKind {
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

static LLVMLumenDiagnosticKind toRust(DiagnosticKind Kind) {
  switch (Kind) {
  case DK_InlineAsm:
    return LLVMLumenDiagnosticKind::InlineAsm;
  case DK_StackSize:
    return LLVMLumenDiagnosticKind::StackSize;
  case DK_DebugMetadataVersion:
    return LLVMLumenDiagnosticKind::DebugMetadataVersion;
  case DK_SampleProfile:
    return LLVMLumenDiagnosticKind::SampleProfile;
  case DK_OptimizationRemark:
    return LLVMLumenDiagnosticKind::OptimizationRemark;
  case DK_OptimizationRemarkMissed:
    return LLVMLumenDiagnosticKind::OptimizationRemarkMissed;
  case DK_OptimizationRemarkAnalysis:
    return LLVMLumenDiagnosticKind::OptimizationRemarkAnalysis;
  case DK_OptimizationRemarkAnalysisFPCommute:
    return LLVMLumenDiagnosticKind::OptimizationRemarkAnalysisFPCommute;
  case DK_OptimizationRemarkAnalysisAliasing:
    return LLVMLumenDiagnosticKind::OptimizationRemarkAnalysisAliasing;
  case DK_PGOProfile:
    return LLVMLumenDiagnosticKind::PGOProfile;
  case DK_Linker:
    return LLVMLumenDiagnosticKind::Linker;
  default:
    return (Kind >= DK_FirstRemark && Kind <= DK_LastRemark)
               ? LLVMLumenDiagnosticKind::OptimizationRemarkOther
               : LLVMLumenDiagnosticKind::Other;
  }
}

extern "C" LLVMLumenDiagnosticKind
LLVMLumenGetDiagInfoKind(LLVMDiagnosticInfoRef DI) {
  return toRust((DiagnosticKind)unwrap(DI)->getKind());
}

extern "C" void LLVMLumenWriteSMDiagnosticToString(LLVMSMDiagnosticRef D,
                                                   RustStringRef Str) {
  RawRustStringOstream OS(Str);
  unwrap(D)->print("", OS);
}

extern "C" void LLVMLumenWriteDiagnosticInfoToString(LLVMDiagnosticInfoRef DI,
                                                     RustStringRef Str) {
  RawRustStringOstream OS(Str);
  DiagnosticPrinterRawOStream DP(OS);
  unwrap(DI)->print(DP);
}

extern "C" void LLVMLumenUnpackOptimizationDiagnostic(
    LLVMDiagnosticInfoRef DI, RustStringRef PassNameOut,
    LLVMValueRef *FunctionOut, unsigned *Line, unsigned *Column,
    RustStringRef FilenameOut, RustStringRef MessageOut) {
  // Undefined to call this not on an optimization diagnostic!
  llvm::DiagnosticInfoOptimizationBase *Opt =
      static_cast<llvm::DiagnosticInfoOptimizationBase *>(unwrap(DI));

  RawRustStringOstream PassNameOS(PassNameOut);
  PassNameOS << Opt->getPassName();
  *FunctionOut = wrap(&Opt->getFunction());

  RawRustStringOstream FilenameOS(FilenameOut);
  DiagnosticLocation loc = Opt->getLocation();
  if (loc.isValid()) {
    *Line = loc.getLine();
    *Column = loc.getColumn();
    FilenameOS << loc.getAbsolutePath();
  }

  RawRustStringOstream MessageOS(MessageOut);
  MessageOS << Opt->getMsg();
}
