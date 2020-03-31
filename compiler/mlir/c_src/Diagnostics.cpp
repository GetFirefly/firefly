#include "lumen/mlir/MLIR.h"
#include "lumen/llvm/RustString.h"

#include "llvm-c/Core.h"
#include "llvm-c/Types.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/Function.h"

#include "mlir/IR/Diagnostics.h"

#include "llvm/Support/CBindingWrapping.h"

typedef struct MLIROpaqueDiagnosticEngine *MLIRDiagnosticEngineRef;
typedef struct MLIROpaqueDiagnostic *MLIRDiagnosticRef;

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(mlir::DiagnosticEngine,
                                   MLIRDiagnosticEngineRef);
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(mlir::Diagnostic, MLIRDiagnosticRef);

typedef int (*RustDiagnosticCallback)(MLIRDiagnosticRef, void *);

extern "C" MLIRDiagnosticEngineRef MLIRGetDiagnosticEngine(MLIRContextRef ctx) {
  mlir::DiagnosticEngine &engine = unwrap(ctx)->getDiagEngine();
  return wrap(&engine);
}

extern "C" void MLIRRegisterDiagnosticHandler(MLIRContextRef ctx, void *handler,
                                              RustDiagnosticCallback callback) {
  mlir::DiagnosticEngine &engine = unwrap(ctx)->getDiagEngine();
  mlir::DiagnosticEngine::HandlerID id = engine.registerHandler(
      [=](mlir::Diagnostic &diag) -> mlir::LogicalResult {
        // Handle the reported diagnostic.
        // Return success to signal that the diagnostic has either been fully
        // processed, or failure if the diagnostic should be propagated to the
        // previous handlers.
        bool should_propogate_diagnostic = callback(wrap(&diag), handler);
        return mlir::failure(should_propogate_diagnostic);
      });
}

extern "C" void MLIRWriteDiagnosticInfoToString(MLIRDiagnosticRef d,
                                                RustStringRef str) {
  RawRustStringOstream OS(str);
  unwrap(d)->print(OS);
}
