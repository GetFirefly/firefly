#include "lumen/mlir/MLIR.h"
#include "lumen/llvm/RustString.h"

#include "llvm-c/Core.h"
#include "llvm-c/Types.h"
#include "llvm/ADT/Optional.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/Function.h"

#include "mlir/IR/Diagnostics.h"

#include "llvm/Support/CBindingWrapping.h"

using llvm::Optional;
using mlir::Diagnostic;
using mlir::Location;

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

extern "C" void MLIRRegisterDiagnosticHandler(MLIRContextRef ctx, void *handlerState,
                                              RustDiagnosticCallback callback) {
  mlir::DiagnosticEngine &engine = unwrap(ctx)->getDiagEngine();
  mlir::DiagnosticEngine::HandlerID id = engine.registerHandler(
      [=](mlir::Diagnostic &diag) -> mlir::LogicalResult {
        // Handle the reported diagnostic.
        // Return success to signal that the diagnostic has either been fully
        // processed, or failure if the diagnostic should be propagated to the
        // previous handlers.
        bool should_propogate_diagnostic = callback(wrap(&diag), handlerState);
        return mlir::failure(should_propogate_diagnostic);
      });
}

namespace LumenDiagnostics {
  enum Severity {
    Note = 0,
    Warning,
    Error,
    Remark
  };
}

extern "C" {
  struct LumenSourceLoc {
    unsigned line;
    unsigned column;
    unsigned filenameLen;
    const char *filename;
  };
}

extern "C" LumenDiagnostics::Severity MLIRGetDiagnosticSeverity(MLIRDiagnosticRef d) {
  Diagnostic *diagnostic = unwrap(d);

  switch (diagnostic->getSeverity()) {
    case mlir::DiagnosticSeverity::Note:
      return LumenDiagnostics::Severity::Note;
    case mlir::DiagnosticSeverity::Warning:
      return LumenDiagnostics::Severity::Warning;
    case mlir::DiagnosticSeverity::Error:
      return LumenDiagnostics::Severity::Error;
    case mlir::DiagnosticSeverity::Remark:
      return LumenDiagnostics::Severity::Remark;
    default:
      llvm_unreachable("unknown diagnostic severity");
  }
}

/// Return a processable FileLineColLoc from the given location.
static Optional<mlir::FileLineColLoc> getFileLineColLoc(mlir::Location loc) {
  if (auto nameLoc = loc.dyn_cast<mlir::NameLoc>())
    return getFileLineColLoc(loc.cast<mlir::NameLoc>().getChildLoc());
  if (auto fileLoc = loc.dyn_cast<mlir::FileLineColLoc>())
    return fileLoc;
  if (auto callLoc = loc.dyn_cast<mlir::CallSiteLoc>())
    return getFileLineColLoc(loc.cast<mlir::CallSiteLoc>().getCallee());
  if (auto fusedLoc = loc.dyn_cast<mlir::FusedLoc>()) {
    for (auto subLoc : loc.cast<mlir::FusedLoc>().getLocations()) {
      if (auto callLoc = getFileLineColLoc(subLoc)) {
        return callLoc;
      }
    }
    return llvm::None;
  }
  return llvm::None;
}

extern "C" bool MLIRGetDiagnosticLocation(MLIRDiagnosticRef d, LumenSourceLoc *out) {
  Diagnostic *diagnostic = unwrap(d);
  Location loc = diagnostic->getLocation();
  Optional<mlir::FileLineColLoc> fileLocOpt = getFileLineColLoc(loc);

  if (fileLocOpt) {
    mlir::FileLineColLoc fileLoc = fileLocOpt.getValue();
    auto fn = fileLoc.getFilename();
    auto filename = fn.data();
    auto filenameLen = (unsigned)fn.size();
    auto line = fileLoc.getLine();
    auto column = fileLoc.getColumn();
    *out = LumenSourceLoc{line, column, filenameLen, filename};
    return true;
  }

  return false;
}

typedef void (*RustDiagnosticNoteCallback)(MLIRDiagnosticRef, void *);

extern "C" void MLIRForEachDiagnosticNote(MLIRDiagnosticRef d, RustDiagnosticNoteCallback callback, void *ifd) {
  Diagnostic *diagnostic = unwrap(d);
  for (auto &note : diagnostic->getNotes()) {
    callback(wrap(&note), ifd);
  }
}

extern "C" void MLIRWriteDiagnosticToString(MLIRDiagnosticRef d,
                                            RustStringRef str) {
  RawRustStringOstream OS(str);
  unwrap(d)->print(OS);
}
