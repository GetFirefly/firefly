#include "lumen/llvm/Target.h"
// On Windows we have a custom output stream type that
// can wrap the raw file handle we get from Rust
#if defined(_WIN32)
#include "lumen/llvm/raw_win32_handle_ostream.h"
#endif

#include "llvm-c/Core.h"
#include "llvm-c/Target.h"
#include "llvm/ADT/Optional.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CBindingWrapping.h"
#include "llvm/Support/Host.h"

using namespace llvm;

using ::lumen::TargetMachineConfig;

DEFINE_STDCXX_CONVERSION_FUNCTIONS(TargetMachine, LLVMTargetMachineRef);

/// getLongestEntryLength - Return the length of the longest entry in the table.
template <typename KV> static size_t getLongestEntryLength(ArrayRef<KV> table) {
  size_t maxLen = 0;
  for (auto &entry : table)
    maxLen = std::max(maxLen, std::strlen(entry.Key));
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
  for (auto &cpu : cpuTable)
    printf("    %-*s\n", maxLen, cpu.Key);
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

  printf("Use +feature to enable a feature, or -feature to disable it.\n"
         "For example, lumen -C -target-cpu=mycpu -C "
         "target-feature=+feature1,-feature2\n\n");
}

extern "C" LLVMTargetMachineRef
LLVMLumenCreateTargetMachine(TargetMachineConfig *conf, char *error) {
  TargetMachineConfig config = *conf;
  StringRef targetTriple = unwrap(config.triple);
  StringRef cpu = unwrap(config.cpu);
  StringRef abi = unwrap(config.abi);
  Triple triple(Triple::normalize(targetTriple));

  std::string err;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(triple.getTriple(), err);
  if (target == nullptr) {
    error = strdup(err.c_str());
    return nullptr;
  }
  error = nullptr;

  SubtargetFeatures features;
  features.getDefaultSubtargetFeatures(triple);

  unsigned featuresLen = config.featuresLen;
  if (featuresLen > 0) {
    ArrayRef<MlirStringRef> targetFeatures(config.features, featuresLen);
    for (auto &feature : targetFeatures) {
      features.AddFeature(unwrap(feature));
    }
  }

  auto rm = config.relocModel;
  llvm::Reloc::Model relocModel;
  if (rm != lumen::RelocModel::Default)
    relocModel = toLLVM(rm);
  else
    relocModel = config.positionIndependentCode ? llvm::Reloc::PIC_
                                                : llvm::Reloc::Static;

  auto codeModel = toLLVM(config.codeModel);

  llvm::TargetOptions options;
  options.DebuggerTuning = llvm::DebuggerKind::LLDB;
  options.FloatABIType = llvm::FloatABI::Default;
  options.DataSections = config.dataSections;
  options.FunctionSections = config.functionSections;
  options.GuaranteedTailCallOpt = true;
  options.MCOptions.AsmVerbose = config.preserveAsmComments;
  options.MCOptions.PreserveAsmComments = config.preserveAsmComments;
  options.MCOptions.ABIName = abi.str();
  options.EmitStackSizeSection = config.emitStackSizeSection;
  // Tell LLVM to codegen `unreachable` into an explicit trap instruction.
  // This limits the extent of possible undefined behavior in some cases, as
  // it prevents control flow from "falling through" into whatever code
  // happens to be laid out next in memory.
  options.TrapUnreachable = true;
  options.EmitCallSiteInfo = true;

  if (!config.enableThreading) {
    options.ThreadModel = llvm::ThreadModel::Single;
  }

  // Always enable wasm exceptions, regardless of features provided;
  // we do this because our codegen depends on exceptions being lowered
  // correctly
  switch (triple.getArch()) {
  case Triple::ArchType::wasm32:
  case Triple::ArchType::wasm64:
    options.ExceptionModel = llvm::ExceptionHandling::Wasm;
    break;
  default:
    break;
  }

  auto optLevel = toLLVM(config.optLevel);
  auto *targetMachine =
      target->createTargetMachine(triple.getTriple(), cpu, features.getString(),
                                  options, relocModel, codeModel, optLevel);

  if (optLevel == llvm::CodeGenOpt::Level::None) {
    targetMachine->setFastISel(true);
    targetMachine->setO0WantsFastISel(true);
    targetMachine->setGlobalISel(false);
  } else {
    targetMachine->setFastISel(false);
    targetMachine->setO0WantsFastISel(false);
    targetMachine->setGlobalISel(false);
  }

  return wrap(targetMachine);
}

#if defined(_WIN32)
extern "C" bool LLVMTargetMachineEmitToFileDescriptor(
    LLVMTargetMachineRef t, LLVMModuleRef m, HANDLE handle,
    LLVMCodeGenFileType codegen, char **errorMessage) {
  raw_win32_handle_ostream stream(handle, /*shouldClose=*/false,
                                  /*unbuffered=*/false);
#else
extern "C" bool
LLVMTargetMachineEmitToFileDescriptor(LLVMTargetMachineRef t, LLVMModuleRef m,
                                      int fd, LLVMCodeGenFileType codegen,
                                      char **errorMessage) {
  llvm::raw_fd_ostream stream(fd, /*shouldClose=*/false, /*unbuffered=*/false,
                              llvm::raw_ostream::OStreamKind::OK_FDStream);
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

Optional<llvm::CodeModel::Model> lumen::toLLVM(lumen::CodeModel cm) {
  switch (cm) {
  case lumen::CodeModel::Small:
    return llvm::CodeModel::Small;
  case lumen::CodeModel::Kernel:
    return llvm::CodeModel::Kernel;
  case lumen::CodeModel::Medium:
    return llvm::CodeModel::Medium;
  case lumen::CodeModel::Large:
    return llvm::CodeModel::Large;
  case lumen::CodeModel::None:
    return llvm::None;
  default:
    llvm::report_fatal_error("invalid llvm code model");
  }
}

llvm::CodeGenOpt::Level lumen::toLLVM(lumen::OptLevel level) {
  switch (level) {
  case lumen::OptLevel::None:
    return llvm::CodeGenOpt::None;
  case lumen::OptLevel::Less:
    return llvm::CodeGenOpt::Less;
  case lumen::OptLevel::Default:
    return llvm::CodeGenOpt::Default;
  case lumen::OptLevel::Aggressive:
    return llvm::CodeGenOpt::Aggressive;
  default:
    llvm::report_fatal_error("invalid llvm optimization level");
  }
}

unsigned lumen::toLLVM(lumen::SizeLevel level) {
  switch (level) {
  case lumen::SizeLevel::None:
    return 0;
  case lumen::SizeLevel::Less:
    return 1;
  case lumen::SizeLevel::Aggressive:
    return 2;
  default:
    llvm::report_fatal_error("invalid llvm code size level");
  }
}

llvm::Reloc::Model lumen::toLLVM(lumen::RelocModel model) {
  switch (model) {
  case lumen::RelocModel::Default:
    return llvm::Reloc::Static;
  case lumen::RelocModel::Static:
    return llvm::Reloc::Static;
  case lumen::RelocModel::PIC:
    return llvm::Reloc::PIC_;
  case lumen::RelocModel::DynamicNoPic:
    return llvm::Reloc::DynamicNoPIC;
  case lumen::RelocModel::ROPI:
    return llvm::Reloc::ROPI;
  case lumen::RelocModel::RWPI:
    return llvm::Reloc::RWPI;
  case lumen::RelocModel::ROPIRWPI:
    return llvm::Reloc::ROPI_RWPI;
  default:
    llvm::report_fatal_error("invalid llvm reloc model");
  }
}

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
