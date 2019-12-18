#ifndef LUMEN_TARGET_UTILS_H
#define LUMEN_TARGET_UTILS_H

#include "llvm-c/TargetMachine.h"
#include "llvm/Support/CBindingWrapping.h"
#include "llvm/Target/TargetMachine.h"

namespace L = llvm;

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(L::TargetMachine, LLVMTargetMachineRef);

enum class LLVMLumenCodeModel {
  Other,
  Small,
  Kernel,
  Medium,
  Large,
  None,
};

static llvm::CodeModel::Model fromRust(LLVMLumenCodeModel Model) {
  switch (Model) {
  case LLVMLumenCodeModel::Small:
    return llvm::CodeModel::Small;
  case LLVMLumenCodeModel::Kernel:
    return llvm::CodeModel::Kernel;
  case LLVMLumenCodeModel::Medium:
    return llvm::CodeModel::Medium;
  case LLVMLumenCodeModel::Large:
    return llvm::CodeModel::Large;
  default:
    llvm::report_fatal_error("Bad CodeModel.");
  }
}

enum class LLVMLumenCodeGenOptLevel {
  Other,
  None,
  Less,
  Default,
  Aggressive,
};

static llvm::CodeGenOpt::Level fromRust(LLVMLumenCodeGenOptLevel Level) {
  switch (Level) {
  case LLVMLumenCodeGenOptLevel::None:
    return llvm::CodeGenOpt::None;
  case LLVMLumenCodeGenOptLevel::Less:
    return llvm::CodeGenOpt::Less;
  case LLVMLumenCodeGenOptLevel::Default:
    return llvm::CodeGenOpt::Default;
  case LLVMLumenCodeGenOptLevel::Aggressive:
    return llvm::CodeGenOpt::Aggressive;
  default:
    llvm::report_fatal_error("Bad CodeGenOptLevel.");
  }
}

enum class LLVMLumenCodeGenSizeLevel {
  Other,
  None,
  Less,
  Aggressive,
};

static unsigned fromRust(LLVMLumenCodeGenSizeLevel Level) {
  switch (Level) {
  case LLVMLumenCodeGenSizeLevel::None:
    return 0;
  case LLVMLumenCodeGenSizeLevel::Less:
    return 1;
  case LLVMLumenCodeGenSizeLevel::Aggressive:
    return 2;
  default:
    llvm::report_fatal_error("Bad CodeGenSizeLevel.");
  }
}

enum class LLVMLumenRelocMode {
  Default,
  Static,
  PIC,
  DynamicNoPic,
  ROPI,
  RWPI,
  ROPIRWPI,
};

static llvm::Optional<llvm::Reloc::Model>
fromRust(LLVMLumenRelocMode LumenReloc) {
  switch (LumenReloc) {
  case LLVMLumenRelocMode::Default:
    return llvm::None;
  case LLVMLumenRelocMode::Static:
    return llvm::Reloc::Static;
  case LLVMLumenRelocMode::PIC:
    return llvm::Reloc::PIC_;
  case LLVMLumenRelocMode::DynamicNoPic:
    return llvm::Reloc::DynamicNoPIC;
  case LLVMLumenRelocMode::ROPI:
    return llvm::Reloc::ROPI;
  case LLVMLumenRelocMode::RWPI:
    return llvm::Reloc::RWPI;
  case LLVMLumenRelocMode::ROPIRWPI:
    return llvm::Reloc::ROPI_RWPI;
  }
  llvm::report_fatal_error("Bad RelocModel.");
}

extern "C" {
/**
 * Print all target processors for the currently selected target
 */
void PrintTargetCPUs(LLVMTargetMachineRef TM);

/**
 * Print all target features for the currently selected target
 */
void PrintTargetFeatures(LLVMTargetMachineRef TM);

LLVMTargetMachineRef LLVMLumenCreateTargetMachine(
    const char *TripleStr, const char *CPU, const char *Feature,
    const char *Abi, LLVMLumenCodeModel LumenCM, LLVMLumenRelocMode LumenReloc,
    LLVMLumenCodeGenOptLevel LumenOptLevel, bool PositionIndependentExecutable,
    bool FunctionSections, bool DataSections, bool TrapUnreachable,
    bool Singlethread, bool AsmComments, bool EmitStackSizeSection,
    bool RelaxElfRelocations);

void LLVMLumenDisposeTargetMachine(LLVMTargetMachineRef TM);

bool LLVMTargetMachineEmitToFileDescriptor(LLVMTargetMachineRef t,
                                           LLVMModuleRef m,
#if defined(_WIN32)
                                           HANDLE handle,
#else
                                           int fd,
#endif
                                           LLVMCodeGenFileType codegen,
                                           char **errorMessage);
} // end extern

#endif
