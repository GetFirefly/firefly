#ifndef LUMEN_COMPILER_DIALECT_EIR_CONVERSION_EIRTOLLVM_H_
#define LUMEN_COMPILER_DIALECT_EIR_CONVERSION_EIRTOLLVM_H_

#include <memory>

#include "lumen/compiler/Dialect/EIR/IR/EIROps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace llvm {
class TargetMachine;
}  // namespace llvm

namespace lumen {
namespace eir {
std::unique_ptr<mlir::OpPassBase<mlir::ModuleOp>> createConvertEIRToLLVMPass(
    llvm::TargetMachine *targetMachine);
}  // namespace eir
}  // namespace lumen

#endif
