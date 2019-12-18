#include "eir/Dialect.h"
#include "eir/Ops.h"
#include "eir/Passes.h"

#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/Sequence.h"

namespace M = mlir;
namespace L = llvm;

using namespace eir;

//===----------------------------------------------------------------------===//
// EIRToStandardLoweringPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to the Standard dialect for operations in EIR
/// that can be represented in that dialect. For all other operations, they will
/// be lowered directly to the LLVM dialect when that pass runs.
namespace {
struct EIRToStandardLoweringPass
    : public M::FunctionPass<EIRToStandardLoweringPass> {
  void runOnFunction() final;
};
} // end namespace

void EIRToStandardLoweringPass::runOnFunction() {
  auto function = getFunction();

  // Verify that the given main has no inputs and results.
  // if (function.getNumArguments() || function.getType().getNumResults()) {
  // function.emitError("expected 'main' to have 0 inputs and 0 results");
  // return signalPassFailure();
  //}

  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  M::ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering. In this case, we are lowering to the Standard dialect only.
  target.addLegalDialect<StandardOpsDialect>();

  // We also define the EIR dialect as illegal, and specify the specific
  // operations that we are not lowering to Standard. This will cause conversion
  // to fail if any EIR operations are not lowered, unless explicitly legalized
  // here.
  //
  // For now, our lowering to standard is primarily focused on lowering calls
  // and control flow operations, everything else has no direct Standard
  // representation
  target.addIllegalDialect<eir::EirDialect>();
  target.addLegalOp<eir::UnreachableOp>();
  target.addLegalOp<eir::AllocaOp>();
  target.addLegalOp<eir::LoadOp>();
  target.addLegalOp<eir::StoreOp>();
  target.addLegalOp<eir::MallocOp>();
  target.addLegalOp<eir::MapInsertOp>();
  target.addLegalOp<eir::MapUpdateOp>();
  target.addLegalOp<eir::BinaryPushOp>();
  target.addLegalOp<eir::TraceCaptureOp>();
  target.addLegalOp<eir::TraceConstructOp>();

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the Toy operations.
  M::OwningRewritePatternList patterns;
  // patterns.insert<IfOpLowering, MatchOpLowering,
  // CallOpLowering>(&getContext());

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion.
  //
  // The conversion will signal failure if any of our `illegal` operations
  // were not converted successfully.
  if (failed(applyPartialConversion(getFunction(), target, patterns)))
    signalPassFailure();
}

std::unique_ptr<M::Pass> eir::createLowerToStandardPass() {
  return std::make_unique<EIRToStandardLoweringPass>();
}
