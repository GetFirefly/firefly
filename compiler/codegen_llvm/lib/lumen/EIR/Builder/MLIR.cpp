#include "lumen/mlir/MLIR.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"

#include "lumen/EIR/IR/EIRDialect.h"

using ::llvm::unwrap;
using ::mlir::MLIRContext;

extern "C" void MLIRRegisterDialects(MLIRContextRef context) {
    MLIRContext *ctx = unwrap(context);

    // Register the LLVM and EIR dialects with MLIR
    ctx->loadDialect<mlir::StandardOpsDialect, mlir::LLVM::LLVMDialect,
                     lumen::eir::eirDialect>();
    assert(ctx->getLoadedDialects().size() >= 3 && "failed to load dialects!");
}
