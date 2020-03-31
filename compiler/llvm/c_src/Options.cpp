#include "llvm/Support/CommandLine.h"

extern "C"
void LLVMLumenSetLLVMOptions(int argc, char **argv) {
  // Initializing the command-line options more than once is not allowed.
  // So check if they've already been initialized.
  static bool initialized = false;
  if (initialized) return;
  initialized = true;

  llvm::cl::ParseCommandLineOptions(argc, argv);
}
