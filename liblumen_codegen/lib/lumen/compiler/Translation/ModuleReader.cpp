#include <memory>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SourceMgr.h"
#include "lumen/compiler/Support/MLIR.h"
#include "lumen/compiler/Support/MemoryBuffer.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Parser.h"

using ::llvm::MemoryBuffer;
using ::llvm::SourceMgr;
using ::llvm::StringRef;
using ::mlir::MLIRContext;

extern "C" MLIRModuleRef MLIRParseFile(MLIRContextRef context,
                                       const char *filename) {
  MLIRContext *ctx = unwrap(context);
  assert(ctx != nullptr && "invalid MLIRContext pointer");
  StringRef inputFilePath(filename);

  // Parse the input mlir.
  auto mod = mlir::parseSourceFile(inputFilePath, ctx);
  if (!mod) {
    return nullptr;
  }

  // We're doing our own memory management, so extract the module from
  // its owning reference
  return wrap(new mlir::ModuleOp(mod.release()));
}

extern "C" MLIRModuleRef MLIRParseBuffer(MLIRContextRef context,
                                         LLVMMemoryBufferRef buffer) {
  MLIRContext *ctx = unwrap(context);
  assert(ctx != nullptr && "invalid MLIRContext pointer");
  SourceMgr sourceMgr;
  auto buffer_ptr = std::unique_ptr<MemoryBuffer>(unwrap(buffer));
  sourceMgr.AddNewSourceBuffer(std::move(buffer_ptr), llvm::SMLoc());

  // Parse the input mlir.
  auto mod = mlir::parseSourceFile(sourceMgr, ctx);
  if (!mod) return nullptr;

  // We're doing our own memory management, so extract the module from
  // its owning reference
  return wrap(new mlir::ModuleOp(mod.release()));
}
