#pragma once

#include <memory>

namespace mlir {
class LowerToLLVMOptions;
class ModuleOp;
template <typename T> class OperationPass;
class Pass;

namespace cir {

std::unique_ptr<OperationPass<ModuleOp>> createInjectYieldPointsPass();
std::unique_ptr<OperationPass<ModuleOp>> createConvertCIRToLLVMPass();
std::unique_ptr<OperationPass<ModuleOp>>
createConvertCIRToLLVMPass(bool enableNanboxing,
                           const LowerToLLVMOptions &options);

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "CIR/Passes.h.inc"
} // namespace cir
} // namespace mlir
