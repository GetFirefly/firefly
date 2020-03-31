#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>

static LLVM_THREAD_LOCAL char *LastError;

extern "C" char *LLVMLumenGetLastError(void) {
  char *ret = LastError;
  LastError = nullptr;
  return ret;
}

extern "C" void LLVMLumenSetLastError(const char *err) {
  free((void *)LastError);
  LastError = strdup(err);
}

static void FatalErrorHandler(void *userData, const std::string &Reason,
                              bool genCrashDiag) {
  std::cerr << "LLVM ERROR: " << Reason << std::endl;

  llvm::sys::RunInterruptHandlers();

  exit(101);
}

extern "C" void LLVMLumenInstallFatalErrorHandler() {
  llvm::install_fatal_error_handler(FatalErrorHandler);
}
