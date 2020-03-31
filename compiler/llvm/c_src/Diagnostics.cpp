#include "lumen/llvm/Diagnostics.h"
#include "lumen/llvm/RustString.h"

#include "llvm-c/Core.h"
#include "llvm-c/Types.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/CBindingWrapping.h"

using namespace lumen;

typedef struct LLVMOpaqueSMDiagnostic *LLVMSMDiagnosticRef;

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(llvm::SMDiagnostic, LLVMSMDiagnosticRef);
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(llvm::DiagnosticInfo, LLVMDiagnosticInfoRef);

DiagnosticKind lumen::toDiagnosticKind(llvm::DiagnosticKind Kind) {
  switch (Kind) {
    case llvm::DK_InlineAsm:
      return DiagnosticKind::InlineAsm;
    case llvm::DK_StackSize:
      return DiagnosticKind::StackSize;
    case llvm::DK_DebugMetadataVersion:
      return DiagnosticKind::DebugMetadataVersion;
    case llvm::DK_SampleProfile:
      return DiagnosticKind::SampleProfile;
    case llvm::DK_OptimizationRemark:
      return DiagnosticKind::OptimizationRemark;
    case llvm::DK_OptimizationRemarkMissed:
      return DiagnosticKind::OptimizationRemarkMissed;
    case llvm::DK_OptimizationRemarkAnalysis:
      return DiagnosticKind::OptimizationRemarkAnalysis;
    case llvm::DK_OptimizationRemarkAnalysisFPCommute:
      return DiagnosticKind::OptimizationRemarkAnalysisFPCommute;
    case llvm::DK_OptimizationRemarkAnalysisAliasing:
      return DiagnosticKind::OptimizationRemarkAnalysisAliasing;
    case llvm::DK_PGOProfile:
      return DiagnosticKind::PGOProfile;
    case llvm::DK_Linker:
      return DiagnosticKind::Linker;
    default:
      return (Kind >= llvm::DK_FirstRemark && Kind <= llvm::DK_LastRemark)
                 ? DiagnosticKind::OptimizationRemarkOther
                 : DiagnosticKind::Other;
  }
}

extern "C" DiagnosticKind LLVMLumenGetDiagInfoKind(LLVMDiagnosticInfoRef DI) {
  llvm::DiagnosticInfo *info = unwrap(DI);
  return toDiagnosticKind((llvm::DiagnosticKind)info->getKind());
}

extern "C" void LLVMLumenWriteSMDiagnosticToString(LLVMSMDiagnosticRef D,
                                                   RustStringRef Str) {
  RawRustStringOstream OS(Str);
  unwrap(D)->print("", OS);
}

extern "C" void LLVMLumenWriteDiagnosticInfoToString(LLVMDiagnosticInfoRef DI,
                                                     RustStringRef Str) {
  RawRustStringOstream OS(Str);
  llvm::DiagnosticPrinterRawOStream DP(OS);
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
  llvm::DiagnosticLocation loc = Opt->getLocation();
  if (loc.isValid()) {
    *Line = loc.getLine();
    *Column = loc.getColumn();
    FilenameOS << loc.getAbsolutePath();
  }

  RawRustStringOstream MessageOS(MessageOut);
  MessageOS << Opt->getMsg();
}
