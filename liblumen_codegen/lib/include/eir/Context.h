// -*- mode: C++ -*-
#ifndef EIR_CONTEXT_H
#define EIR_CONTEXT_H

#include "lumen/RustString.h"
#include "llvm/Support/CBindingWrapping.h"

namespace mlir {
class MLIRContext;
class DiagnosticEngine;
class Diagnostic;
} // namespace mlir

namespace M = mlir;

typedef struct MLIROpaqueContext *MLIRContextRef;
typedef struct MLIROpaqueDiagnosticEngine *MLIRDiagnosticEngineRef;
typedef struct MLIROpaqueDiagnostic *MLIRDiagnosticRef;

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(M::MLIRContext, MLIRContextRef);
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

extern "C" MLIRContextRef MLIRCreateContext();

extern "C" MLIRDiagnosticEngineRef MLIRGetDiagnosticEngine(MLIRContextRef ctx);

extern "C" void MLIRRegisterDiagnosticHandler(MLIRContextRef ctx, void *handler,
                                              RustDiagnosticCallback callback);

extern "C" void MLIRWriteDiagnosticInfoToString(MLIRDiagnosticRef D,
                                                RustStringRef Str);
#endif // end EIR_CONTEXT_H
