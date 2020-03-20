#ifndef LUMEN_SUPPORT_MLIR_H
#define LUMEN_SUPPORT_MLIR_H

#include "llvm/Support/CBindingWrapping.h"

namespace mlir {
class MLIRContext;
class ModuleOp;
}  // namespace mlir

using ::mlir::MLIRContext;

typedef struct MLIROpaqueContext *MLIRContextRef;
typedef struct MLIROpaqueModuleOp *MLIRModuleRef;

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(mlir::MLIRContext, MLIRContextRef);
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(mlir::ModuleOp, MLIRModuleRef);

// Specialized opaque context conversions.
inline MLIRContext **unwrap(MLIRContextRef *Tys) {
  return reinterpret_cast<MLIRContext **>(Tys);
}
inline MLIRContextRef *wrap(const MLIRContext **Tys) {
  return reinterpret_cast<MLIRContextRef *>(const_cast<MLIRContext **>(Tys));
}

extern "C" {
MLIRContextRef MLIRCreateContext();
}

#endif
