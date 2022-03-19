#include "llvm/ADT/Optional.h"

#include "mlir/CAPI/Diagnostics.h"
#include "mlir/CAPI/IR.h"
#include "mlir/IR/Diagnostics.h"

#include "llvm/Support/CBindingWrapping.h"

using llvm::Optional;
using mlir::Location;

extern "C" {
struct FileLineColLoc {
  unsigned line;
  unsigned column;
  unsigned filenameLen;
  const char *filename;
};
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

extern "C" FileLineColLoc mlirDiagnosticGetFileLineCol(MlirDiagnostic d) {
  Location loc = unwrap(mlirDiagnosticGetLocation(d));
  Optional<mlir::FileLineColLoc> fileLocOpt = getFileLineColLoc(loc);

  if (fileLocOpt.hasValue()) {
    mlir::FileLineColLoc fileLoc = fileLocOpt.getValue();
    auto fn = fileLoc.getFilename();
    auto filename = fn.data();
    auto filenameLen = (unsigned)fn.size();
    auto line = fileLoc.getLine();
    auto column = fileLoc.getColumn();
    return FileLineColLoc{line, column, filenameLen, filename};
  }

  return FileLineColLoc{0, 0, 0, nullptr};
}
