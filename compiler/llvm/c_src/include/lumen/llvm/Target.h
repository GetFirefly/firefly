#ifndef LUMEN_TARGET_H
#define LUMEN_TARGET_H

#include "llvm/ADT/Triple.h"
#include "llvm/Support/CBindingWrapping.h"
#include "llvm-c/TargetMachine.h"
#include "llvm/Target/TargetMachine.h"

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(::llvm::TargetMachine, LLVMTargetMachineRef);

namespace lumen {

using CodeGenOptLevel = ::llvm::CodeGenOpt::Level;

enum class CodeModel {
  Other,
  Small,
  Kernel,
  Medium,
  Large,
  None,
};

enum class OptLevel {
  Other,
  None,
  Less,
  Default,
  Aggressive,
};

enum class SizeLevel {
  Other,
  None,
  Less,
  Aggressive,
};

enum class RelocMode {
  Default,
  Static,
  PIC,
  DynamicNoPic,
  ROPI,
  RWPI,
  ROPIRWPI,
};

llvm::CodeModel::Model toLLVM(CodeModel cm);

llvm::CodeGenOpt::Level toLLVM(OptLevel level);

unsigned toLLVM(SizeLevel level);

llvm::Reloc::Model toLLVM(RelocMode mode);
}  // namespace lumen

#endif
