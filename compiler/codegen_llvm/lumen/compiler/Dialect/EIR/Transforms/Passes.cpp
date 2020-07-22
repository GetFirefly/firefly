#include "lumen/compiler/Dialect/EIR/Transforms/Passes.h"

#include <memory>

#include "llvm/Target/TargetMachine.h"
#include "lumen/compiler/Dialect/EIR/Conversion/EIRToLLVM/ConvertEIRToLLVM.h"
#include "lumen/compiler/Dialect/EIR/IR/EIROps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace lumen {
namespace eir {

void buildEIRTransformPassPipeline(mlir::OpPassManager &passManager,
                                   llvm::TargetMachine *targetMachine) {
  passManager.addPass(createConvertEIRToLLVMPass(targetMachine));
  OpPassManager &optPM = passManager.nest<::mlir::LLVM::LLVMFuncOp>();
  optPM.addPass(mlir::createCanonicalizerPass());
  optPM.addPass(mlir::createCSEPass());
  // passManager.addPass(createGlobalInitializationPass());
  // TODO: run symbol DCE pass.
}

}  // namespace eir
}  // namespace lumen
