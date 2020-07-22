#ifndef LUMEN_COMPILER_DIALECT_EIR_CONVERSION_EIRTOLLVM_H_
#define LUMEN_COMPILER_DIALECT_EIR_CONVERSION_EIRTOLLVM_H_

#include <memory>

#include "mlir/Pass/Pass.h"

namespace llvm {
class TargetMachine;
}  // namespace llvm

namespace lumen {
namespace eir {
std::unique_ptr<mlir::Pass> createConvertEIRToLLVMPass(
    llvm::TargetMachine *targetMachine);
}  // namespace eir
}  // namespace lumen

#endif
