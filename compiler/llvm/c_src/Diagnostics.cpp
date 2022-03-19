#include "lumen/llvm/Diagnostics.h"

#include "mlir-c/Support.h"
#include "mlir/CAPI/Support.h"
#include "llvm-c/Core.h"
#include "llvm-c/Types.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/CBindingWrapping.h"
#include "llvm/Support/Casting.h"

using namespace llvm;

lumen::DiagnosticKind lumen::toDiagnosticKind(llvm::DiagnosticKind kind) {
  switch (kind) {
  case llvm::DK_InlineAsm:
    return lumen::DiagnosticKind::InlineAsm;
  case llvm::DK_ResourceLimit:
    return lumen::DiagnosticKind::ResourceLimit;
  case llvm::DK_StackSize:
    return lumen::DiagnosticKind::StackSize;
  case llvm::DK_Linker:
    return lumen::DiagnosticKind::Linker;
  case llvm::DK_Lowering:
    return lumen::DiagnosticKind::Lowering;
  case llvm::DK_DebugMetadataVersion:
    return lumen::DiagnosticKind::DebugMetadataVersion;
  case llvm::DK_DebugMetadataInvalid:
    return lumen::DiagnosticKind::DebugMetadataInvalid;
  case llvm::DK_ISelFallback:
    return lumen::DiagnosticKind::ISelFallback;
  case llvm::DK_SampleProfile:
    return lumen::DiagnosticKind::SampleProfile;
  case llvm::DK_OptimizationRemark:
    return lumen::DiagnosticKind::OptimizationRemark;
  case llvm::DK_OptimizationRemarkMissed:
    return lumen::DiagnosticKind::OptimizationRemarkMissed;
  case llvm::DK_OptimizationRemarkAnalysis:
    return lumen::DiagnosticKind::OptimizationRemarkAnalysis;
  case llvm::DK_OptimizationRemarkAnalysisFPCommute:
    return lumen::DiagnosticKind::OptimizationRemarkAnalysisFPCommute;
  case llvm::DK_OptimizationRemarkAnalysisAliasing:
    return lumen::DiagnosticKind::OptimizationRemarkAnalysisAliasing;
  case llvm::DK_MachineOptimizationRemark:
    return lumen::DiagnosticKind::MachineOptimizationRemark;
  case llvm::DK_MachineOptimizationRemarkMissed:
    return lumen::DiagnosticKind::MachineOptimizationRemarkMissed;
  case llvm::DK_MachineOptimizationRemarkAnalysis:
    return lumen::DiagnosticKind::MachineOptimizationRemarkAnalysis;
  case llvm::DK_MIRParser:
    return lumen::DiagnosticKind::MIRParser;
  case llvm::DK_PGOProfile:
    return lumen::DiagnosticKind::PGOProfile;
  case llvm::DK_Unsupported:
    return lumen::DiagnosticKind::Unsupported;
  case llvm::DK_SrcMgr:
    return lumen::DiagnosticKind::SrcMgr;
  case llvm::DK_DontCall:
    return lumen::DiagnosticKind::DontCall;
  default:
    return lumen::DiagnosticKind::Other;
  }
}

extern "C" lumen::DiagnosticKind
LLVMLumenGetDiagInfoKind(LLVMDiagnosticInfoRef di) {
  llvm::DiagnosticInfo *info = unwrap(di);
  return lumen::toDiagnosticKind((llvm::DiagnosticKind)info->getKind());
}

extern "C" bool
LLVMLumenOptimizationDiagnosticIsVerbose(LLVMDiagnosticInfoRef d) {
  llvm::DiagnosticInfoOptimizationBase *opt =
      static_cast<llvm::DiagnosticInfoOptimizationBase *>(unwrap(d));
  return opt->isVerbose();
}

extern "C" MlirStringRef
LLVMLumenOptimizationDiagnosticPassName(LLVMDiagnosticInfoRef d) {
  llvm::DiagnosticInfoOptimizationBase *opt =
      static_cast<llvm::DiagnosticInfoOptimizationBase *>(unwrap(d));
  return wrap(opt->getPassName());
}

extern "C" MlirStringRef
LLVMLumenOptimizationDiagnosticRemarkName(LLVMDiagnosticInfoRef d) {
  llvm::DiagnosticInfoOptimizationBase *opt =
      static_cast<llvm::DiagnosticInfoOptimizationBase *>(unwrap(d));
  return wrap(opt->getRemarkName());
}

extern "C" const char *
LLVMLumenOptimizationDiagnosticMessage(LLVMDiagnosticInfoRef d) {
  llvm::DiagnosticInfoOptimizationBase *opt =
      static_cast<llvm::DiagnosticInfoOptimizationBase *>(unwrap(d));
  auto msg = opt->getMsg();
  return strdup(msg.c_str());
}

extern "C" LLVMValueRef
LLVMLumenOptimizationDiagnosticCodeRegion(LLVMDiagnosticInfoRef d) {
  llvm::DiagnosticInfoOptimizationBase *opt =
      static_cast<llvm::DiagnosticInfoOptimizationBase *>(unwrap(d));
  if (auto irOpt = dyn_cast_or_null<llvm::DiagnosticInfoIROptimization>(opt))
    return wrap(irOpt->getCodeRegion());
  else
    return nullptr;
}

extern "C" LLVMValueRef
LLVMLumenDiagnosticWithLocFunction(LLVMDiagnosticInfoRef d) {
  llvm::DiagnosticInfoWithLocationBase *opt =
      static_cast<llvm::DiagnosticInfoWithLocationBase *>(unwrap(d));
  return wrap(&opt->getFunction());
}

extern "C" bool LLVMLumenDiagnosticWithLocSourceLoc(LLVMDiagnosticInfoRef d,
                                                    MlirStringRef *relativePath,
                                                    unsigned *line,
                                                    unsigned *col) {
  llvm::DiagnosticInfoWithLocationBase *opt =
      static_cast<llvm::DiagnosticInfoWithLocationBase *>(unwrap(d));
  if (opt->isLocationAvailable()) {
    auto loc = opt->getLocation();
    *relativePath = wrap(loc.getRelativePath());
    *line = loc.getLine();
    *col = loc.getColumn();
    return true;
  } else {
    return false;
  }
}

extern "C" LLVMValueRef
LLVMLumenISelFallbackDiagnosticFunction(LLVMDiagnosticInfoRef di) {
  llvm::DiagnosticInfoISelFallback *opt =
      static_cast<llvm::DiagnosticInfoISelFallback *>(unwrap(di));

  return wrap(&opt->getFunction());
}

extern "C" const char *
LLVMLumenUnsupportedDiagnosticMessage(LLVMDiagnosticInfoRef di) {
  llvm::DiagnosticInfoUnsupported *opt =
      static_cast<llvm::DiagnosticInfoUnsupported *>(unwrap(di));

  auto msg = opt->getMessage().str();
  return strdup(msg.c_str());
}
