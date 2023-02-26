#include "firefly/llvm/Target.h"
// On Windows we have a custom output stream type that
// can wrap the raw file handle we get from Rust
#if defined(_WIN32)
#include "firefly/llvm/raw_win32_handle_ostream.h"
#endif

#include "mlir/CAPI/Support.h"
#include "llvm-c/Core.h"
#include "llvm-c/Target.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
//#include "llvm/Support/CBindingWrapping.h"
#include "llvm/Support/Host.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

/// getLongestEntryLength - Return the length of the longest entry in the table.
template <typename KV> static size_t getLongestEntryLength(ArrayRef<KV> table) {
  size_t maxLen = 0;
  for (auto &entry : table)
    maxLen = std::max(maxLen, std::strlen(entry.Key));
  return maxLen;
}

bool LLVMFireflyHasFeature(LLVMTargetMachineRef tm, const char *feature) {
  llvm::TargetMachine *target = unwrap(tm);
  const llvm::MCSubtargetInfo *mcInfo = target->getMCSubtargetInfo();
  return mcInfo->checkFeatures(std::string("+") + feature);
}

#ifdef FIREFLY_LLVM

void PrintTargetCPUs(LLVMTargetMachineRef tm) {
  const llvm::TargetMachine *target = unwrap(tm);
  auto *mcInfo = target->getMCSubtargetInfo();
  auto hostArch = Triple(llvm::sys::getProcessTriple()).getArch();
  auto targetArch = target->getTargetTriple().getArch();
  auto cpuTable = mcInfo->getCPUTable();
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

void PrintTargetFeatures(LLVMTargetMachineRef tm) {
  const llvm::TargetMachine *target = unwrap(tm);
  const llvm::MCSubtargetInfo *mcInfo = target->getMCSubtargetInfo();
  const ArrayRef<SubtargetFeatureKV> featTable = mcInfo->getFeatureTable();

  unsigned maxLen = getLongestEntryLength(featTable);

  printf("Available features for this target:\n");
  for (auto &feature : featureTable)
    printf("    %-*s - %s.\n", maxLen, feature.Key, feature.Desc);
  printf("\n");

  printf("Use +feature to enable a feature, or -feature to disable it.\n"
         "For example, firefly -C -target-cpu=mycpu -C "
         "target-feature=+feature1,-feature2\n\n");
}

#else

void PrintTargetCPUs(LLVMTargetMachineRef _tm) {
  printf("Target feature help is not supported by this LLVM version.\n\n");
}

void PrintTargetFeatures(LLVMTargetMachineRef _tm) {
  printf("Target cpu help is not supported by this LLVM version.\n\n");
}

#endif // define FIREFLY_LLVM

LLVMTargetMachineRef LLVMFireflyCreateTargetMachine(TargetMachineConfig *conf,
                                                    char **error) {
  TargetMachineConfig config = *conf;
  StringRef targetTriple = unwrap(config.triple);
  StringRef cpu = unwrap(config.cpu);
  StringRef abi = unwrap(config.abi);
  Triple triple(Triple::normalize(targetTriple));

  std::string err;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(triple.getTriple(), err);
  if (target == nullptr) {
    *error = strdup(err.c_str());
    return wrap((llvm::TargetMachine *)nullptr);
  }
  *error = nullptr;

  SubtargetFeatures features;
  features.getDefaultSubtargetFeatures(triple);

  unsigned featuresLen = config.featuresLen;
  if (featuresLen > 0) {
    ArrayRef<MlirStringRef> targetFeatures(config.features, featuresLen);
    for (auto &feature : targetFeatures) {
      features.AddFeature(unwrap(feature));
    }
  }

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
  Optional<llvm::CodeModel::Model> codeModel;
  if (config.codeModel)
    codeModel = *config.codeModel;
  Optional<llvm::Reloc::Model> relocModel;
  if (config.relocModel)
    relocModel = *config.relocModel;

  auto *targetMachine = target->createTargetMachine(
      triple.getTriple(), cpu, features.getString(), options, relocModel,
      codeModel, config.optLevel);

  if (config.optLevel == llvm::CodeGenOpt::Level::None) {
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
bool LLVMTargetMachineEmitToFileDescriptor(LLVMTargetMachineRef t,
                                           LLVMModuleRef m, HANDLE handle,
                                           llvm::CodeGenFileType codegen,
                                           char **errorMessage) {
  raw_win32_handle_ostream stream(handle, /*shouldClose=*/false,
                                  /*unbuffered=*/false);
#else
bool LLVMTargetMachineEmitToFileDescriptor(LLVMTargetMachineRef t,
                                           LLVMModuleRef m, int fd,
                                           llvm::CodeGenFileType ft,
                                           char **errorMessage) {
  llvm::raw_fd_ostream stream(fd, /*shouldClose=*/false, /*unbuffered=*/false,
                              llvm::raw_ostream::OStreamKind::OK_FDStream);
#endif
  TargetMachine *tm = unwrap(t);
  llvm::Module *mod = llvm::unwrap(m);
  mod->setDataLayout(tm->createDataLayout());

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

void LLVM_InitializeAllTargetInfos(void) {
  LLVMInitializeX86TargetInfo();
  LLVMInitializeAArch64TargetInfo();
  LLVMInitializeWebAssemblyTargetInfo();
}

void LLVM_InitializeAllTargets(void) {
  LLVMInitializeX86Target();
  LLVMInitializeAArch64Target();
  LLVMInitializeWebAssemblyTarget();
}

void LLVM_InitializeAllTargetMCs(void) {
  LLVMInitializeX86TargetMC();
  LLVMInitializeAArch64TargetMC();
  LLVMInitializeWebAssemblyTargetMC();
}

void LLVM_InitializeAllAsmPrinters(void) {
  LLVMInitializeX86AsmPrinter();
  LLVMInitializeAArch64AsmPrinter();
  LLVMInitializeWebAssemblyAsmPrinter();
}

void LLVM_InitializeAllAsmParsers(void) {
  LLVMInitializeX86AsmParser();
  LLVMInitializeAArch64AsmParser();
  LLVMInitializeWebAssemblyAsmParser();
}

void LLVM_InitializeAllDisassemblers(void) {
  LLVMInitializeX86Disassembler();
  LLVMInitializeAArch64Disassembler();
  LLVMInitializeWebAssemblyDisassembler();
}

/* These functions return true on failure. */
LLVMBool LLVM_InitializeNativeTarget(void) {
  return LLVMInitializeNativeTarget();
}

LLVMBool LLVM_InitializeNativeAsmParser(void) {
  return LLVMInitializeNativeAsmParser();
}

LLVMBool LLVM_InitializeNativeAsmPrinter(void) {
  return LLVMInitializeNativeAsmPrinter();
}

LLVMBool LLVM_InitializeNativeDisassembler(void) {
  return LLVMInitializeNativeDisassembler();
}
