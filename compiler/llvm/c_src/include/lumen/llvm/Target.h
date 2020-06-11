#ifndef LUMEN_TARGET_H
#define LUMEN_TARGET_H

#include "llvm-c/TargetMachine.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/ADT/Triple.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/CBindingWrapping.h"

DEFINE_STDCXX_CONVERSION_FUNCTIONS(llvm::TargetMachine, LLVMTargetMachineRef);

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
  Default,
  Static,
  PIC,
  DynamicNoPic,
  ROPI,
  RWPI,
  ROPIRWPI,
};

struct TargetFeature {
    const char *name;
    unsigned nameLen;
};

struct TargetMachineConfig {
    const char *triple;
    unsigned tripleLen;
    const char *cpu;
    unsigned cpuLen;
    const char *abi;
    unsigned abiLen;
    const TargetFeature *features;
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
}  // namespace lumen

#endif
