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

DiagnosticKind lumen::toDiagnosticKind(llvm::DiagnosticKind kind) {
  switch (kind) {
    case llvm::DK_InlineAsm:
      return DiagnosticKind::InlineAsm;
    case llvm::DK_ResourceLimit:
      return DiagnosticKind::ResourceLimit;
    case llvm::DK_StackSize:
      return DiagnosticKind::StackSize;
    case llvm::DK_Linker:
      return DiagnosticKind::Linker;
    case llvm::DK_DebugMetadataVersion:
      return DiagnosticKind::DebugMetadataVersion;
    case llvm::DK_DebugMetadataInvalid:
      return DiagnosticKind::DebugMetadataInvalid;
    case llvm::DK_ISelFallback:
      return DiagnosticKind::ISelFallback;
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
    case llvm::DK_MachineOptimizationRemark:
      return DiagnosticKind::MachineOptimizationRemark;
    case llvm::DK_MachineOptimizationRemarkMissed:
      return DiagnosticKind::MachineOptimizationRemarkMissed;
    case llvm::DK_MachineOptimizationRemarkAnalysis:
      return DiagnosticKind::MachineOptimizationRemarkAnalysis;
    case llvm::DK_MIRParser:
      return DiagnosticKind::MIRParser;
    case llvm::DK_PGOProfile:
      return DiagnosticKind::PGOProfile;
    case llvm::DK_MisExpect:
      return DiagnosticKind::MisExpect;
    case llvm::DK_Unsupported:
      return DiagnosticKind::Unsupported;
    default:
      return DiagnosticKind::Other;
  }
}

extern "C" DiagnosticKind 
LLVMLumenGetDiagInfoKind(LLVMDiagnosticInfoRef di) {
  llvm::DiagnosticInfo *info = unwrap(di);
  return toDiagnosticKind((llvm::DiagnosticKind)info->getKind());
}

extern "C" void 
LLVMLumenWriteSMDiagnosticToString(LLVMSMDiagnosticRef d, RustStringRef str) {
  RawRustStringOstream out(str);
  unwrap(d)->print("", out);
}

extern "C" void 
LLVMLumenWriteDiagnosticInfoToString(LLVMDiagnosticInfoRef di, RustStringRef str) {
  RawRustStringOstream out(str);
  llvm::DiagnosticPrinterRawOStream printer(out);
  unwrap(di)->print(printer);
}

extern "C" bool 
LLVMLumenIsVerboseOptimizationDiagnostic(LLVMDiagnosticInfoRef di) {
  // Undefined to call this not on an optimization diagnostic!
  llvm::DiagnosticInfoOptimizationBase *opt =
      static_cast<llvm::DiagnosticInfoOptimizationBase *>(unwrap(di));

  return opt->isVerbose();
}

extern "C" void 
LLVMLumenUnpackOptimizationDiagnostic(LLVMDiagnosticInfoRef di, RustStringRef passNameOut,
                                      LLVMValueRef *functionOut, 
                                      unsigned *line, unsigned *column,
                                      RustStringRef filenameOut, 
                                      RustStringRef messageOut) {
  // Undefined to call this not on an optimization diagnostic!
  llvm::DiagnosticInfoOptimizationBase *opt =
      static_cast<llvm::DiagnosticInfoOptimizationBase *>(unwrap(di));

  RawRustStringOstream passNameOS(passNameOut);
  passNameOS << opt->getPassName();
  *functionOut = wrap(&opt->getFunction());

  RawRustStringOstream filenameOS(filenameOut);
  llvm::DiagnosticLocation loc = opt->getLocation();
  if (loc.isValid()) {
    *line = loc.getLine();
    *column = loc.getColumn();
    filenameOS << loc.getAbsolutePath();
  }

  RawRustStringOstream messageOS(messageOut);
  messageOS << opt->getMsg();
}

extern "C" void 
LLVMLumenUnpackISelFallbackDiagnostic(LLVMDiagnosticInfoRef di, LLVMValueRef *functionOut) {
  llvm::DiagnosticInfoISelFallback *opt =
      static_cast<llvm::DiagnosticInfoISelFallback *>(unwrap(di));

  *functionOut = wrap(&opt->getFunction());
}
