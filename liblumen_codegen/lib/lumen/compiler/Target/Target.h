#ifndef LUMEN_TARGET_H
#define LUMEN_TARGET_H

#include "llvm/ADT/Triple.h"
#include "llvm/Target/TargetMachine.h"

namespace lumen {

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

llvm::CodeModel::Model toLLVM(CodeModel cm) {
  switch (cm) {
  case CodeModel::Small:
    return llvm::CodeModel::Small;
  case CodeModel::Kernel:
    return llvm::CodeModel::Kernel;
  case CodeModel::Medium:
    return llvm::CodeModel::Medium;
  case CodeModel::Large:
    return llvm::CodeModel::Large;
  default:
    llvm::report_fatal_error("invalid llvm code model");
  }
}

llvm::CodeGenOpt::Level toLLVM(OptLevel level) {
  switch (level) {
  case OptLevel::None:
    return llvm::CodeGenOpt::None;
  case OptLevel::Less:
    return llvm::CodeGenOpt::Less;
  case OptLevel::Default:
    return llvm::CodeGenOpt::Default;
  case OptLevel::Aggressive:
    return llvm::CodeGenOpt::Aggressive;
  default:
    llvm::report_fatal_error("invalid llvm optimization level");
  }
}

unsigned toLLVM(SizeLevel level) {
  switch (level) {
  case SizeLevel::None:
    return 0;
  case SizeLevel::Less:
    return 1;
  case SizeLevel::Aggressive:
    return 2;
  default:
    llvm::report_fatal_error("invalid llvm code size level");
  }
}

llvm::Reloc::Model toLLVM(RelocMode mode) {
  switch (mode) {
  case RelocMode::Default:
    return llvm::Reloc::Static;
  case RelocMode::Static:
    return llvm::Reloc::Static;
  case RelocMode::PIC:
    return llvm::Reloc::PIC_;
  case RelocMode::DynamicNoPic:
    return llvm::Reloc::DynamicNoPIC;
  case RelocMode::ROPI:
    return llvm::Reloc::ROPI;
  case RelocMode::RWPI:
    return llvm::Reloc::RWPI;
  case RelocMode::ROPIRWPI:
    return llvm::Reloc::ROPI_RWPI;
  default:
    llvm::report_fatal_error("invalid llvm reloc mode");
  }
}

} // namespace lumen


#endif
