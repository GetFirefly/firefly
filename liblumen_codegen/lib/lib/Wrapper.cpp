#include "eir/Dialect.h"
#include "lumen/Lumen.h"

#include "llvm/ADT/Optional.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/Pass/PassManager.h"

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace llvm;
using namespace llvm::sys;

static LLVM_THREAD_LOCAL char *LastError;

extern "C" char *LLVMLumenGetLastError(void) {
  char *Ret = LastError;
  LastError = nullptr;
  return Ret;
}

extern "C" void LLVMLumenSetLastError(const char *Err) {
  free((void *)LastError);
  LastError = strdup(Err);
}

static void FatalErrorHandler(void *UserData, const std::string &Reason,
                              bool GenCrashDiag) {
  std::cerr << "LLVM ERROR: " << Reason << std::endl;

  sys::RunInterruptHandlers();

  exit(101);
}

extern "C" void LLVMLumenInstallFatalErrorHandler() {
  install_fatal_error_handler(FatalErrorHandler);
}

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(Twine, LLVMTwineRef)

extern "C" void LLVMLumenWriteTwineToString(LLVMTwineRef T, RustStringRef Str) {
  RawRustStringOstream OS(Str);
  unwrap(T)->print(OS);
}

extern "C" void LLVMLumenWriteTypeToString(LLVMTypeRef Ty, RustStringRef Str) {
  RawRustStringOstream OS(Str);
  unwrap<llvm::Type>(Ty)->print(OS);
}

extern "C" void LLVMLumenWriteValueToString(LLVMValueRef V, RustStringRef Str) {
  RawRustStringOstream OS(Str);
  if (!V) {
    OS << "(null)";
  } else {
    OS << "(";
    unwrap<llvm::Value>(V)->getType()->print(OS);
    OS << ":";
    unwrap<llvm::Value>(V)->print(OS);
    OS << ")";
  }
}

extern "C" void LLVMLumenSetLLVMOptions(int Argc, char **Argv) {
  // Initializing the command-line options more than once is not allowed.
  // So check if they've already been initialized.
  static bool Initialized = false;
  if (Initialized)
    return;
  Initialized = true;

  M::registerPassManagerCLOptions();
  cl::ParseCommandLineOptions(Argc, Argv);

  // Register the EIR dialect with MLIR
  M::registerDialect<M::StandardOpsDialect>();
  M::registerDialect<M::loop::LoopOpsDialect>();
  M::registerDialect<eir::EirDialect>();
}

extern "C" uint32_t LLVMLumenVersionMajor() { return LLVM_VERSION_MAJOR; }
extern "C" uint32_t LLVMLumenVersionMinor() { return LLVM_VERSION_MINOR; }
