#include "firefly/llvm/Diagnostics.h"

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

firefly::DiagnosticKind firefly::toDiagnosticKind(llvm::DiagnosticKind kind) {
  switch (kind) {
  case llvm::DK_InlineAsm:
    return firefly::DiagnosticKind::InlineAsm;
  case llvm::DK_ResourceLimit:
    return firefly::DiagnosticKind::ResourceLimit;
  case llvm::DK_StackSize:
    return firefly::DiagnosticKind::StackSize;
  case llvm::DK_Linker:
    return firefly::DiagnosticKind::Linker;
  case llvm::DK_Lowering:
    return firefly::DiagnosticKind::Lowering;
  case llvm::DK_DebugMetadataVersion:
    return firefly::DiagnosticKind::DebugMetadataVersion;
  case llvm::DK_DebugMetadataInvalid:
    return firefly::DiagnosticKind::DebugMetadataInvalid;
  case llvm::DK_ISelFallback:
    return firefly::DiagnosticKind::ISelFallback;
  case llvm::DK_SampleProfile:
    return firefly::DiagnosticKind::SampleProfile;
  case llvm::DK_OptimizationRemark:
    return firefly::DiagnosticKind::OptimizationRemark;
  case llvm::DK_OptimizationRemarkMissed:
    return firefly::DiagnosticKind::OptimizationRemarkMissed;
  case llvm::DK_OptimizationRemarkAnalysis:
    return firefly::DiagnosticKind::OptimizationRemarkAnalysis;
  case llvm::DK_OptimizationRemarkAnalysisFPCommute:
    return firefly::DiagnosticKind::OptimizationRemarkAnalysisFPCommute;
  case llvm::DK_OptimizationRemarkAnalysisAliasing:
    return firefly::DiagnosticKind::OptimizationRemarkAnalysisAliasing;
  case llvm::DK_MachineOptimizationRemark:
    return firefly::DiagnosticKind::MachineOptimizationRemark;
  case llvm::DK_MachineOptimizationRemarkMissed:
    return firefly::DiagnosticKind::MachineOptimizationRemarkMissed;
  case llvm::DK_MachineOptimizationRemarkAnalysis:
    return firefly::DiagnosticKind::MachineOptimizationRemarkAnalysis;
  case llvm::DK_MIRParser:
    return firefly::DiagnosticKind::MIRParser;
  case llvm::DK_PGOProfile:
    return firefly::DiagnosticKind::PGOProfile;
  case llvm::DK_Unsupported:
    return firefly::DiagnosticKind::Unsupported;
  case llvm::DK_SrcMgr:
    return firefly::DiagnosticKind::SrcMgr;
  case llvm::DK_DontCall:
    return firefly::DiagnosticKind::DontCall;
  default:
    return firefly::DiagnosticKind::Other;
  }
}

firefly::DiagnosticKind LLVMFireflyGetDiagInfoKind(LLVMDiagnosticInfoRef di) {
  llvm::DiagnosticInfo *info = unwrap(di);
  return firefly::toDiagnosticKind((llvm::DiagnosticKind)info->getKind());
}

bool LLVMFireflyOptimizationDiagnosticIsVerbose(LLVMDiagnosticInfoRef d) {
  llvm::DiagnosticInfoOptimizationBase *opt =
      static_cast<llvm::DiagnosticInfoOptimizationBase *>(unwrap(d));
  return opt->isVerbose();
}

MlirStringRef
LLVMFireflyOptimizationDiagnosticPassName(LLVMDiagnosticInfoRef d) {
  llvm::DiagnosticInfoOptimizationBase *opt =
      static_cast<llvm::DiagnosticInfoOptimizationBase *>(unwrap(d));
  return wrap(opt->getPassName());
}

MlirStringRef
LLVMFireflyOptimizationDiagnosticRemarkName(LLVMDiagnosticInfoRef d) {
  llvm::DiagnosticInfoOptimizationBase *opt =
      static_cast<llvm::DiagnosticInfoOptimizationBase *>(unwrap(d));
  return wrap(opt->getRemarkName());
}

const char *LLVMFireflyOptimizationDiagnosticMessage(LLVMDiagnosticInfoRef d) {
  llvm::DiagnosticInfoOptimizationBase *opt =
      static_cast<llvm::DiagnosticInfoOptimizationBase *>(unwrap(d));
  auto msg = opt->getMsg();
  return strdup(msg.c_str());
}

LLVMValueRef
LLVMFireflyOptimizationDiagnosticCodeRegion(LLVMDiagnosticInfoRef d) {
  llvm::DiagnosticInfoOptimizationBase *opt =
      static_cast<llvm::DiagnosticInfoOptimizationBase *>(unwrap(d));
  if (auto irOpt = dyn_cast_or_null<llvm::DiagnosticInfoIROptimization>(opt))
    return wrap(irOpt->getCodeRegion());
  else
    return nullptr;
}

LLVMValueRef LLVMFireflyDiagnosticWithLocFunction(LLVMDiagnosticInfoRef d) {
  llvm::DiagnosticInfoWithLocationBase *opt =
      static_cast<llvm::DiagnosticInfoWithLocationBase *>(unwrap(d));
  return wrap(&opt->getFunction());
}

bool LLVMFireflyDiagnosticWithLocSourceLoc(LLVMDiagnosticInfoRef d,
                                           MlirStringRef *relativePath,
                                           unsigned *line, unsigned *col) {
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

LLVMValueRef
LLVMFireflyISelFallbackDiagnosticFunction(LLVMDiagnosticInfoRef di) {
  llvm::DiagnosticInfoISelFallback *opt =
      static_cast<llvm::DiagnosticInfoISelFallback *>(unwrap(di));

  return wrap(&opt->getFunction());
}

const char *LLVMFireflyUnsupportedDiagnosticMessage(LLVMDiagnosticInfoRef di) {
  llvm::DiagnosticInfoUnsupported *opt =
      static_cast<llvm::DiagnosticInfoUnsupported *>(unwrap(di));

  auto msg = opt->getMessage().str();
  return strdup(msg.c_str());
}
