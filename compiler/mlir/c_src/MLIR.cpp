#include "lumen/mlir/MLIR.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Pass/PassManager.h"

extern "C" {
  struct ContextOptions {
    bool printOpOnDiagnostic;
    bool printStackTraceOnDiagnostic;
    bool enableMultithreading;
  };
}

extern "C"
MLIRContextRef MLIRCreateContext(ContextOptions *opts) {
  auto *ctx = new mlir::MLIRContext(/*loadAllDialects=*/false);
  ctx->printOpOnDiagnostic(opts->printOpOnDiagnostic);
  ctx->printStackTraceOnDiagnostic(opts->printStackTraceOnDiagnostic);
  ctx->enableMultithreading(opts->enableMultithreading);
  ctx->allowUnregisteredDialects(false);
  
  return wrap(ctx);
}

extern "C"
void MLIRLumenInit() {
  // Initializing the command-line options more than once is not allowed.
  // So check if they've already been initialized.
  static bool initialized = false;
  if (initialized) return;
  initialized = true;

  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  mlir::registerAsmPrinterCLOptions();
}
