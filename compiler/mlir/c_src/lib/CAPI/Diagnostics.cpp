#include "mlir/CAPI/Diagnostics.h"
#include "mlir/CAPI/IR.h"
#include "mlir/IR/Diagnostics.h"

#include "llvm/Support/CBindingWrapping.h"

#include "CIR-c/Diagnostics.h"

#include <optional>

using mlir::Location;

/// Return a processable FileLineColLoc from the given location.
static std::optional<mlir::FileLineColLoc>
getFileLineColLoc(mlir::Location loc) {
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
    return std::nullopt;
  }
  return std::nullopt;
}

FileLineColLoc mlirDiagnosticGetFileLineCol(MlirDiagnostic d) {
  Location loc = unwrap(mlirDiagnosticGetLocation(d));
  std::optional<mlir::FileLineColLoc> fileLocOpt = getFileLineColLoc(loc);

  if (fileLocOpt) {
    mlir::FileLineColLoc fileLoc = *fileLocOpt;
    auto fn = fileLoc.getFilename();
    auto filename = fn.data();
    auto filenameLen = (unsigned)fn.size();
    auto line = fileLoc.getLine();
    auto column = fileLoc.getColumn();
    return FileLineColLoc{line, column, filenameLen, filename};
  }

  return FileLineColLoc{0, 0, 0, nullptr};
}

void mlirContextSetPrintOpOnDiagnostic(MlirContext context, bool enable) {
  return unwrap(context)->printOpOnDiagnostic(enable);
}

void mlirContextSetPrintStackTraceOnDiagnostic(MlirContext context,
                                               bool enable) {
  return unwrap(context)->printStackTraceOnDiagnostic(enable);
}
