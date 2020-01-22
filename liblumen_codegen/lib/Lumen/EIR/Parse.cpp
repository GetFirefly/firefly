#include "eir/Parse.h"
#include "eir/Context.h"
#include "lumen/LLVM.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Parser.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

#include <memory>

namespace M = mlir;
namespace L = llvm;

MLIRModuleRef MLIRParseFile(MLIRContextRef context, const char *filename) {
  M::MLIRContext *ctx = unwrap(context);
  assert(ctx != nullptr && "invalid MLIRContext pointer");
  L::StringRef inputFilePath(filename);

  // Parse the input mlir.
  auto mod = M::parseSourceFile(inputFilePath, ctx);
  if (!mod) {
    return nullptr;
  }

  // We're doing our own memory management, so extract the module from
  // its owning reference
  return wrap(new M::ModuleOp(mod.release()));
}

MLIRModuleRef MLIRParseBuffer(MLIRContextRef context,
                              LLVMMemoryBufferRef buffer) {
  M::MLIRContext *ctx = unwrap(context);
  assert(ctx != nullptr && "invalid MLIRContext pointer");
  L::SourceMgr sourceMgr;
  auto buffer_ptr = std::unique_ptr<L::MemoryBuffer>(unwrap(buffer));
  sourceMgr.AddNewSourceBuffer(std::move(buffer_ptr), L::SMLoc());

  // Parse the input mlir.
  auto mod = M::parseSourceFile(sourceMgr, ctx);
  if (!mod) {
    return nullptr;
  }

  // We're doing our own memory management, so extract the module from
  // its owning reference
  return wrap(new M::ModuleOp(mod.release()));
}
