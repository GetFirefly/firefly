#include "lumen/compiler/Dialect/EIR/Transforms/Passes.h"
#include "lumen/compiler/Dialect/EIR/Conversion/EIRToStandard/ConvertEIRToStandard.h"
#include "lumen/compiler/Dialect/EIR/IR/EIROps.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

#include <memory>

namespace lumen {
namespace eir {

void buildEIRTransformPassPipeline(mlir::OpPassManager &passManager) {
  passManager.addPass(createConvertEIRToStandardPass());
  passManager.addPass(mlir::createCanonicalizerPass());
  //passManager.addPass(createGlobalInitializationPass());
  passManager.addPass(mlir::createCSEPass());

  // TODO: run symbol DCE pass.
}

static mlir::PassPipelineRegistration<> transformPassPipeline(
    "lumen-eir-transformation-pipeline",
    "Runs the full EIR dialect transformation pipeline",
    [](mlir::OpPassManager &passManager) {
      buildEIRTransformPassPipeline(passManager);
    });

}  // namespace eir
}  // namespace lumen
