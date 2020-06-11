#ifndef LUMEN_SUPPORT_MLIR_H
#define LUMEN_SUPPORT_MLIR_H

#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/CBindingWrapping.h"

namespace mlir {
class ModuleOp;
class PassManager;
}  // namespace mlir

typedef struct MLIROpaqueContext *MLIRContextRef;
typedef struct MLIROpaqueModuleOp *MLIRModuleRef;
typedef struct MLIROpaquePassManager *MLIRPassManagerRef;

DEFINE_STDCXX_CONVERSION_FUNCTIONS(mlir::MLIRContext, MLIRContextRef);
DEFINE_STDCXX_CONVERSION_FUNCTIONS(mlir::ModuleOp, MLIRModuleRef);
DEFINE_STDCXX_CONVERSION_FUNCTIONS(mlir::PassManager, MLIRPassManagerRef);

#endif
