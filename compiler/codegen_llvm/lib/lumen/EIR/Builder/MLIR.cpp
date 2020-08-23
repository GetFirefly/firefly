#include "lumen/mlir/MLIR.h"

#include "lumen/EIR/IR/EIRDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"

using ::llvm::unwrap;
using ::mlir::MLIRContext;

extern "C" void MLIRRegisterDialects(MLIRContextRef context) {
  MLIRContext *ctx = unwrap(context);

  // Register the LLVM and EIR dialects with MLIR
  ctx->getOrLoadDialect<mlir::StandardOpsDialect>();
  ctx->getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  ctx->getOrLoadDialect<lumen::eir::eirDialect>();
}
