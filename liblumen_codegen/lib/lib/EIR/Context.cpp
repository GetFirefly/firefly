#include "eir/Context.h"
#include "mlir/IR/Diagnostics.h"

static M::MLIRContext &globalContext() {
  static thread_local M::MLIRContext context;
  return context;
}

extern "C" MLIRContextRef MLIRCreateContext() {
  return wrap(new M::MLIRContext());
}

extern "C" MLIRDiagnosticEngineRef MLIRGetDiagnosticEngine(MLIRContextRef ctx) {
  M::DiagnosticEngine &engine = unwrap(ctx)->getDiagEngine();
  return wrap(&engine);
}

extern "C" void MLIRRegisterDiagnosticHandler(MLIRContextRef ctx, void *handler,
                                              RustDiagnosticCallback callback) {
  M::DiagnosticEngine &engine = unwrap(ctx)->getDiagEngine();
  M::DiagnosticEngine::HandlerID id =
      engine.registerHandler([=](M::Diagnostic &diag) -> M::LogicalResult {
        // Handle the reported diagnostic.
        // Return success to signal that the diagnostic has either been fully
        // processed, or failure if the diagnostic should be propagated to the
        // previous handlers.
        bool should_propogate_diagnostic = callback(wrap(&diag), handler);
        return M::failure(should_propogate_diagnostic);
      });
}

extern "C" void MLIRWriteDiagnosticInfoToString(MLIRDiagnosticRef D,
                                                RustStringRef Str) {
  RawRustStringOstream OS(Str);
  unwrap(D)->print(OS);
}
