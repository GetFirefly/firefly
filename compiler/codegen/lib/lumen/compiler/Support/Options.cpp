#include "lumen/compiler/Support/Options.h"

#include "llvm/Support/CommandLine.h"
#include "lumen/compiler/Dialect/EIR/IR/EIRDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/PassManager.h"

using namespace lumen;

extern "C" uint32_t LLVMLumenVersionMajor() { return LLVM_VERSION_MAJOR; }
extern "C" uint32_t LLVMLumenVersionMinor() { return LLVM_VERSION_MINOR; }

void LLVMLumenSetLLVMOptions(int argc, char **argv) {
  // Initializing the command-line options more than once is not allowed.
  // So check if they've already been initialized.
  static bool initialized = false;
  if (initialized) return;
  initialized = true;

  mlir::registerPassManagerCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv);

  // Register the EIR dialect with MLIR
  mlir::registerDialect<mlir::LLVM::LLVMDialect>();
  mlir::registerDialect<eir::EirDialect>();
}
