#include "lumen/compiler/Support/MLIR.h"

#include "mlir/IR/MLIRContext.h"

MLIRContextRef MLIRCreateContext() { return wrap(new mlir::MLIRContext()); }
