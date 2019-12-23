#include "lumen/Target.h"
#include "lumen/Lumen.h"

// On Windows we have a custom output stream type that
// can wrap the raw file handle we get from Rust
#if defined(_WIN32)
#include "lumen/Support/raw_win32_handle_ostream.h"
#endif

#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"

#include <cstdlib>
#include <stdio.h>

using namespace llvm;

/// getLongestEntryLength - Return the length of the longest entry in the table.
template <typename KV>
static size_t getLongestEntryLength(ArrayRef<KV> Table) {
  size_t MaxLen = 0;
  for (auto &I : Table)
    MaxLen = std::max(MaxLen, std::strlen(I.Key));
  return MaxLen;
}

extern "C" void PrintTargetCPUs(LLVMTargetMachineRef TM) {
  const TargetMachine *Target = unwrap(TM);
  const MCSubtargetInfo *MCInfo = Target->getMCSubtargetInfo();
  const Triple::ArchType HostArch = Triple(sys::getProcessTriple()).getArch();
  const Triple::ArchType TargetArch = Target->getTargetTriple().getArch();
  const ArrayRef<SubtargetSubTypeKV> CPUTable = MCInfo->getSubtargetSubTypes();
  unsigned MaxCPULen = getLongestEntryLength(CPUTable);

  printf("Available CPUs for this target:\n");
  if (HostArch == TargetArch) {
    const StringRef HostCPU = sys::getHostCPUName();
    printf("    %-*s - Select the CPU of the current host (currently %.*s).\n",
           MaxCPULen, "native", (int)HostCPU.size(), HostCPU.data());
  }
  for (auto &CPU : CPUTable)
    printf("    %-*s\n", MaxCPULen, CPU.Key);
  printf("\n");
}

extern "C" void PrintTargetFeatures(LLVMTargetMachineRef TM) {
  const TargetMachine *Target = unwrap(TM);
  const MCSubtargetInfo *MCInfo = Target->getMCSubtargetInfo();
  const ArrayRef<SubtargetFeatureKV> FeatTable = MCInfo->getSubtargetFeatures();
  unsigned MaxFeatLen = getLongestEntryLength(FeatTable);

  printf("Available features for this target:\n");
  for (auto &Feature : FeatTable)
    printf("    %-*s - %s.\n", MaxFeatLen, Feature.Key, Feature.Desc);
  printf("\n");

  printf("Use +feature to enable a feature, or -feature to disable it.\n"
         "For example, rustc -C -target-cpu=mycpu -C "
         "target-feature=+feature1,-feature2\n\n");
}

extern "C" LLVMTargetMachineRef LLVMLumenCreateTargetMachine(
    const char *TripleStr, const char *CPU, const char *Feature,
    const char *Abi, LLVMLumenCodeModel LumenCM, LLVMLumenRelocMode LumenReloc,
    LLVMLumenCodeGenOptLevel LumenOptLevel, bool PositionIndependentExecutable,
    bool FunctionSections, bool DataSections, bool TrapUnreachable,
    bool Singlethread, bool AsmComments, bool EmitStackSizeSection,
    bool RelaxElfRelocations) {

  auto OptLevel = fromRust(LumenOptLevel);
  auto RM = fromRust(LumenReloc);

  std::string Error;
  Triple Trip(Triple::normalize(TripleStr));
  const llvm::Target *TheTarget =
      TargetRegistry::lookupTarget(Trip.getTriple(), Error);
  if (TheTarget == nullptr) {
    LLVMLumenSetLastError(Error.c_str());
    return nullptr;
  }

  TargetOptions Options;

  Options.FloatABIType = FloatABI::Default;
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
    Options.ThreadModel = ThreadModel::Single;
  }

  Options.EmitStackSizeSection = EmitStackSizeSection;

  Optional<CodeModel::Model> CM;
  if (LumenCM != LLVMLumenCodeModel::None)
    CM = fromRust(LumenCM);
  return wrap(TheTarget->createTargetMachine(Trip.getTriple(), CPU, Feature, Options, RM, CM, OptLevel));
}

extern "C" void LLVMLumenDisposeTargetMachine(LLVMTargetMachineRef TM) {
  delete unwrap(TM);
}

#if defined(_WIN32)
bool LLVMTargetMachineEmitToFileDescriptor(LLVMTargetMachineRef t,
                                           LLVMModuleRef m, HANDLE handle,
                                           LLVMCodeGenFileType codegen,
                                           char **errorMessage) {
  raw_win32_handle_ostream stream(handle, /*shouldClose=*/false,
                                  /*unbuffered=*/false);
#else
bool LLVMTargetMachineEmitToFileDescriptor(LLVMTargetMachineRef t,
                                           LLVMModuleRef m, int fd,
                                           LLVMCodeGenFileType codegen,
                                           char **errorMessage) {
  raw_fd_ostream stream(fd, /*shouldClose=*/false, /*unbuffered=*/false);
#endif
  TargetMachine *tm = unwrap(t);
  Module *mod = unwrap(m);
  mod->setDataLayout(tm->createDataLayout());

  CodeGenFileType ft;
  switch (codegen) {
  case LLVMCodeGenFileType::LLVMAssemblyFile:
    ft = CodeGenFileType::CGFT_AssemblyFile;
    break;
  default:
    ft = CodeGenFileType::CGFT_ObjectFile;
    break;
  }

  legacy::PassManager pass;
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
