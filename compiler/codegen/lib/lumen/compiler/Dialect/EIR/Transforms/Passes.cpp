#include "lumen/compiler/Dialect/EIR/Transforms/Passes.h"
#include "lumen/compiler/Dialect/EIR/Conversion/EIRToLLVM/ConvertEIRToLLVM.h"
#include "lumen/compiler/Dialect/EIR/IR/EIROps.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Target/TargetMachine.h"

#include <memory>

namespace lumen {
namespace eir {

void buildEIRTransformPassPipeline(mlir::OpPassManager &passManager,
                                   llvm::TargetMachine *targetMachine) {
  passManager.addPass(createConvertEIRToLLVMPass(targetMachine));
  passManager.addPass(mlir::createCanonicalizerPass());
  passManager.addPass(mlir::createCSEPass());
  // passManager.addPass(createGlobalInitializationPass());
  // TODO: run symbol DCE pass.
}

}  // namespace eir
}  // namespace lumen
