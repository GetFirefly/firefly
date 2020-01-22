#include "eir/Context.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Module.h"

#include "llvm-c/Core.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
// On Windows we have a custom output stream type that
// can wrap the raw file handle we get from Rust
#if defined(_WIN32)
#include "lumen/Support/raw_win32_handle_ostream.h"
#endif

namespace M = mlir;
namespace L = llvm;

static M::MLIRContext &globalContext() {
  static thread_local M::MLIRContext context;
  return context;
}

MLIRContextRef MLIRCreateContext() { return wrap(new M::MLIRContext()); }

MLIRDiagnosticEngineRef MLIRGetDiagnosticEngine(MLIRContextRef ctx) {
  M::DiagnosticEngine &engine = unwrap(ctx)->getDiagEngine();
  return wrap(&engine);
}

void MLIRRegisterDiagnosticHandler(MLIRContextRef ctx, void *handler,
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

void MLIRWriteDiagnosticInfoToString(MLIRDiagnosticRef D, RustStringRef Str) {
  RawRustStringOstream OS(Str);
  unwrap(D)->print(OS);
}

#if defined(_WIN32)
bool MLIREmitToFileDescriptor(MLIRModuleRef m, HANDLE handle,
                              char **errorMessage) {
  L::raw_win32_handle_ostream stream(handle, /*shouldClose=*/false,
                                     /*unbuffered=*/false);
#else
bool MLIREmitToFileDescriptor(MLIRModuleRef m, int fd, char **errorMessage) {
  L::raw_fd_ostream stream(fd, /*shouldClose=*/false, /*unbuffered=*/false);
#endif
  M::ModuleOp *mod = unwrap(m);
  mod->print(stream);
  if (stream.has_error()) {
    std::error_code error = stream.error();
    *errorMessage = strdup(error.message().c_str());
    return true;
  }
  stream.flush();
  return false;
}

LLVMMemoryBufferRef MLIREmitToMemoryBuffer(MLIRModuleRef m) {
  M::ModuleOp *mod = unwrap(m);
  L::SmallString<0> codeString;
  L::raw_svector_ostream oStream(codeString);
  mod->print(oStream);
  L::StringRef data = oStream.str();
  return LLVMCreateMemoryBufferWithMemoryRangeCopy(data.data(), data.size(),
                                                   "");
}
