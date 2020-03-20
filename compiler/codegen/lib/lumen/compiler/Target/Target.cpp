#include "lumen/compiler/Target/Target.h"

#include "lumen/compiler/Support/ErrorHandling.h"
// On Windows we have a custom output stream type that
// can wrap the raw file handle we get from Rust
#if defined(_WIN32)
#include "lumen/compiler/Support/raw_win32_handle_ostream.h"
#endif

#include "llvm-c/Core.h"
#include "llvm-c/Target.h"
#include "llvm-c/TargetMachine.h"
#include "llvm/ADT/Optional.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CBindingWrapping.h"
#include "llvm/Support/TargetRegistry.h"

using ::llvm::Optional;

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(llvm::Module, LLVMModuleRef);
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(llvm::TargetMachine, LLVMTargetMachineRef);

using ::llvm::ArrayRef;
using ::llvm::TargetMachine;
using ::llvm::Triple;

/// getLongestEntryLength - Return the length of the longest entry in the table.
template <typename KV>
static size_t getLongestEntryLength(ArrayRef<KV> table) {
  size_t maxLen = 0;
  for (auto &entry : table) maxLen = std::max(maxLen, std::strlen(entry.Key));
  return maxLen;
}

extern "C" void PrintTargetCPUs(LLVMTargetMachineRef tm) {
  const TargetMachine *target = unwrap(tm);
  auto *mcInfo = target->getMCSubtargetInfo();
  auto hostArch = Triple(llvm::sys::getProcessTriple()).getArch();
  auto targetArch = target->getTargetTriple().getArch();
  auto cpuTable = mcInfo->getSubtargetSubTypes();
  unsigned maxLen = getLongestEntryLength(cpuTable);

  printf("Available CPUs for this target:\n");
  if (hostArch == targetArch) {
    auto host = llvm::sys::getHostCPUName();
    printf("    %-*s - Select the CPU of the current host (currently %.*s).\n",
           maxLen, "native", (int)host.size(), host.data());
  }
  for (auto &cpu : cpuTable) printf("    %-*s\n", maxLen, cpu.Key);
  printf("\n");
}

extern "C" void PrintTargetFeatures(LLVMTargetMachineRef tm) {
  const TargetMachine *target = unwrap(tm);
  auto *mcInfo = target->getMCSubtargetInfo();
  auto featureTable = mcInfo->getSubtargetFeatures();
  unsigned maxLen = getLongestEntryLength(featureTable);

  printf("Available features for this target:\n");
  for (auto &feature : featureTable)
    printf("    %-*s - %s.\n", maxLen, feature.Key, feature.Desc);
  printf("\n");

  printf(
      "Use +feature to enable a feature, or -feature to disable it.\n"
      "For example, lumen -C -target-cpu=mycpu -C "
      "target-feature=+feature1,-feature2\n\n");
}

extern "C" LLVMTargetMachineRef LLVMLumenCreateTargetMachine(
    const char *TripleStr, const char *CPU, const char *Feature,
    const char *Abi, lumen::CodeModel LumenCM, lumen::RelocMode LumenReloc,
    lumen::OptLevel LumenOptLevel, bool PositionIndependentExecutable,
    bool FunctionSections, bool DataSections, bool TrapUnreachable,
    bool Singlethread, bool AsmComments, bool EmitStackSizeSection,
    bool RelaxElfRelocations) {
  auto OptLevel = toLLVM(LumenOptLevel);
  auto RM = toLLVM(LumenReloc);

  std::string Error;
  Triple Trip(Triple::normalize(TripleStr));
  const llvm::Target *TheTarget =
      llvm::TargetRegistry::lookupTarget(Trip.getTriple(), Error);
  if (TheTarget == nullptr) {
    LLVMLumenSetLastError(Error.c_str());
    return nullptr;
  }

  llvm::TargetOptions Options;

  Options.FloatABIType = llvm::FloatABI::Default;
  Options.DataSections = DataSections;
  Options.FunctionSections = FunctionSections;
  Options.MCOptions.AsmVerbose = AsmComments;
  Options.MCOptions.PreserveAsmComments = AsmComments;

  if (TrapUnreachable) {
    // Tell LLVM to codegen `unreachable` into an explicit trap instruction.
    // This limits the extent of possible undefined behavior in some cases, as
    // it prevents control flow from "falling through" into whatever code
    // happens to be laid out next in memory.
    Options.TrapUnreachable = true;
  }

  if (Singlethread) {
    Options.ThreadModel = llvm::ThreadModel::Single;
  }

  Options.EmitStackSizeSection = EmitStackSizeSection;

  Optional<llvm::CodeModel::Model> CM;
  if (LumenCM != lumen::CodeModel::None) CM = toLLVM(LumenCM);
  return wrap(TheTarget->createTargetMachine(Trip.getTriple(), CPU, Feature,
                                             Options, RM, CM, OptLevel));
}

extern "C" void LLVMLumenDisposeTargetMachine(LLVMTargetMachineRef tm) {
  delete unwrap(tm);
}

#if defined(_WIN32)
extern "C" bool LLVMTargetMachineEmitToFileDescriptor(
    LLVMTargetMachineRef t, LLVMModuleRef m, HANDLE handle,
    LLVMCodeGenFileType codegen, char **errorMessage) {
  raw_win32_handle_ostream stream(handle, /*shouldClose=*/false,
                                  /*unbuffered=*/false);
#else
extern "C" bool LLVMTargetMachineEmitToFileDescriptor(
    LLVMTargetMachineRef t, LLVMModuleRef m, int fd,
    LLVMCodeGenFileType codegen, char **errorMessage) {
  llvm::raw_fd_ostream stream(fd, /*shouldClose=*/false, /*unbuffered=*/false);
#endif
  TargetMachine *tm = unwrap(t);
  llvm::Module *mod = unwrap(m);
  mod->setDataLayout(tm->createDataLayout());

  llvm::CodeGenFileType ft;
  switch (codegen) {
    case LLVMCodeGenFileType::LLVMAssemblyFile:
      ft = llvm::CodeGenFileType::CGFT_AssemblyFile;
      break;
    default:
      ft = llvm::CodeGenFileType::CGFT_ObjectFile;
      break;
  }

  llvm::legacy::PassManager pass;
  std::string error;
  if (tm->addPassesToEmitFile(pass, stream, nullptr, ft)) {
    error = "TargetMachine can't emit a file of this type";
    *errorMessage = strdup(error.c_str());
    return true;
  }

  pass.run(*mod);
  stream.flush();

  return false;
}

namespace lumen {
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

}  // namespace lumen

extern "C" void LLVM_InitializeAllTargetInfos(void) {
  LLVMInitializeAllTargetInfos();
}

extern "C" void LLVM_InitializeAllTargets(void) { LLVMInitializeAllTargets(); }

extern "C" void LLVM_InitializeAllTargetMCs(void) {
  LLVMInitializeAllTargetMCs();
}

extern "C" void LLVM_InitializeAllAsmPrinters(void) {
  LLVMInitializeAllAsmPrinters();
}

extern "C" void LLVM_InitializeAllAsmParsers(void) {
  LLVMInitializeAllAsmParsers();
}

extern "C" void LLVM_InitializeAllDisassemblers(void) {
  LLVMInitializeAllDisassemblers();
}

/* These functions return true on failure. */
extern "C" LLVMBool LLVM_InitializeNativeTarget(void) {
  return LLVMInitializeNativeTarget();
}

extern "C" LLVMBool LLVM_InitializeNativeAsmParser(void) {
  return LLVMInitializeNativeAsmParser();
}

extern "C" LLVMBool LLVM_InitializeNativeAsmPrinter(void) {
  return LLVMInitializeNativeAsmPrinter();
}

extern "C" LLVMBool LLVM_InitializeNativeDisassembler(void) {
  return LLVMInitializeNativeDisassembler();
}
