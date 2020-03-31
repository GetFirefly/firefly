#include "lumen/mlir/MLIR.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"

extern "C"
MLIRContextRef MLIRCreateContext() { return wrap(new mlir::MLIRContext()); }

extern "C"
void MLIRLumenInit() {
  // Initializing the command-line options more than once is not allowed.
  // So check if they've already been initialized.
  static bool initialized = false;
  if (initialized) return;
  initialized = true;

  mlir::registerPassManagerCLOptions();
}
