#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

using ::llvm::MemoryBuffer;
using ::llvm::SourceMgr;
using ::llvm::StringRef;
using ::mlir::OwningOpRef;

extern "C" MlirModule mlirParseFile(MlirContext context, MlirStringRef path) {
  // Parse the input mlir.
  OwningOpRef<mlir::ModuleOp> owned =
      mlir::parseSourceFile<mlir::ModuleOp>(unwrap(path), unwrap(context));
  if (!owned) {
    return MlirModule{nullptr};
  }

  return MlirModule{owned.release().getOperation()};
}
