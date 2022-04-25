#ifndef LUMEN_TARGET_H
#define LUMEN_TARGET_H

#include "mlir-c/Support.h"
#include "mlir/CAPI/Support.h"
#include "llvm-c/TargetMachine.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/CBindingWrapping.h"
#include "llvm/Target/TargetMachine.h"

#include <stdlib.h>

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

enum class RelocModel {
  Static,
  Pic,
  Pie,
  DynamicNoPic,
  Ropi,
  Rwpi,
  RopiRwpi,
};

struct TargetMachineConfig {
  MlirStringRef triple;
  MlirStringRef cpu;
  MlirStringRef abi;
  const MlirStringRef *features;
  unsigned featuresLen;
  bool relaxELFRelocations;
  bool positionIndependentCode;
  bool dataSections;
  bool functionSections;
  bool emitStackSizeSection;
  bool preserveAsmComments;
  bool enableThreading;
  CodeModel codeModel;
  RelocModel relocModel;
  OptLevel optLevel;
  SizeLevel sizeLevel;
};

llvm::Optional<llvm::CodeModel::Model> toLLVM(CodeModel cm);

llvm::CodeGenOpt::Level toLLVM(OptLevel level);

unsigned toLLVM(SizeLevel level);

llvm::Reloc::Model toLLVM(RelocModel model);
} // namespace lumen

#endif
