#ifndef EIR_LOWER_H
#define EIR_LOWER_H

#include "eir/Context.h"
#include "lumen/Target.h"

#include "llvm-c/Core.h"

namespace eir {
enum class TargetDialect {
  Unknown,
  TargetNone,
  TargetEIR,
  TargetStandard,
  TargetLLVM,
};
} // namespace eir

extern "C" {
MLIRModuleRef MLIRLowerModule(MLIRContextRef context, MLIRModuleRef mod,
                              eir::TargetDialect dialect,
                              LLVMLumenCodeGenOptLevel opt);

LLVMModuleRef MLIRLowerToLLVMIR(MLIRModuleRef m, LLVMLumenCodeGenOptLevel opt,
                                LLVMLumenCodeGenSizeLevel size,
                                LLVMTargetMachineRef tm);
} // extern "C"

#endif // EIR_LOWER_H
