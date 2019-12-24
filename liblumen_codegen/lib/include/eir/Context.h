#ifndef EIR_CONTEXT_H
#define EIR_CONTEXT_H

#include "lumen/RustString.h"

#include "llvm-c/Types.h"
#include "llvm/Support/CBindingWrapping.h"

namespace mlir {
class MLIRContext;
class ModuleOp;
class DiagnosticEngine;
class Diagnostic;
} // namespace mlir

namespace M = mlir;

typedef struct MLIROpaqueContext *MLIRContextRef;
typedef struct MLIROpaqueModuleOp *MLIRModuleRef;
typedef struct MLIROpaqueDiagnosticEngine *MLIRDiagnosticEngineRef;
typedef struct MLIROpaqueDiagnostic *MLIRDiagnosticRef;

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(M::MLIRContext, MLIRContextRef);
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(M::ModuleOp, MLIRModuleRef);
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(M::DiagnosticEngine,
                                   MLIRDiagnosticEngineRef);
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(M::Diagnostic, MLIRDiagnosticRef);

// Specialized opaque context conversions.
inline M::MLIRContext **unwrap(MLIRContextRef *Tys) {
  return reinterpret_cast<M::MLIRContext **>(Tys);
}
inline MLIRContextRef *wrap(const M::MLIRContext **Tys) {
  return reinterpret_cast<MLIRContextRef *>(const_cast<M::MLIRContext **>(Tys));
}

typedef int (*RustDiagnosticCallback)(MLIRDiagnosticRef, void *);

extern "C" {
MLIRContextRef MLIRCreateContext();

MLIRDiagnosticEngineRef MLIRGetDiagnosticEngine(MLIRContextRef ctx);

void MLIRRegisterDiagnosticHandler(MLIRContextRef ctx, void *handler,
                                   RustDiagnosticCallback callback);

void MLIRWriteDiagnosticInfoToString(MLIRDiagnosticRef D, RustStringRef Str);

bool MLIREmitToFileDescriptor(MLIRModuleRef m,
#if defined(_WIN32)
                              HANDLE handle,
#else
                              int fd,
#endif
                              char **errorMessage);

LLVMMemoryBufferRef MLIREmitToMemoryBuffer(MLIRModuleRef m);
}

#endif // end EIR_CONTEXT_H
