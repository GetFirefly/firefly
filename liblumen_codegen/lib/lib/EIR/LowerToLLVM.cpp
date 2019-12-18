#include "eir/Dialect.h"
#include "eir/Ops.h"
#include "eir/Passes.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/Sequence.h"

namespace M = mlir;
namespace L = llvm;

using namespace eir;

//===----------------------------------------------------------------------===//
// RewritePatterns
//===----------------------------------------------------------------------===//

namespace {} // end namespace

//===----------------------------------------------------------------------===//
// EIRToLLVMLoweringPass
//===----------------------------------------------------------------------===//

namespace {
struct EIRToLLVMLoweringPass : public M::ModulePass<EIRToLLVMLoweringPass> {
  void runOnModule() final;

private:
  void generateStaticInitializers(M::ModuleOp &mod);
};
} // end anonymous namespace

void EIRToLLVMLoweringPass::runOnModule() {
  M::ConversionTarget target(getContext());
  target.addLegalDialect<M::LLVM::LLVMDialect>();
  target.addLegalOp<M::ModuleOp, M::ModuleTerminatorOp>();

  // We use a TypeConverter which handles conversion to types in the LLVM
  // dialect
  M::LLVMTypeConverter typeConverter(&getContext());

  // At this point in lowering, a module may consist of a combination of
  // operations in various dialects, namely `eir`, `affine`, `loop`, and `std`.
  //
  // We use the existing patterns for the `affine`, `loop` and `std` dialects,
  // and just provide our own patterns for lowering `eir` operations to LLVM.
  //
  // The patterns lower in multiple stages, since we're relying on some
  // transitive lowerings, which require lowering from one dialect to an
  // intermediate dialect, and from that dialect to LLVM.
  M::OwningRewritePatternList patterns;
  populateAffineToStdConversionPatterns(patterns, &getContext());
  populateLoopToStdConversionPatterns(patterns, &getContext());
  populateStdToLLVMConversionPatterns(typeConverter, patterns);

  // Provide the patterns for the remaining EIR operations
  /*
  patterns.insert<eir::UnreachableOpLowering>(&getContext());
  patterns.insert<eir::AllocaOpLowering>(&getContext());
  patterns.insert<eir::LoadOpLowering>(&getContext());
  patterns.insert<eir::StoreOpLowering>(&getContext());
  patterns.insert<eir::MallocOpLowering>(&getContext());
  patterns.insert<eir::MapInsertOpLowering>(&getContext());
  patterns.insert<eir::MapUpdateOpLowering>(&getContext());
  patterns.insert<eir::BinaryPushOpLowering>(&getContext());
  patterns.insert<eir::TraceCaptureOpLowering>(&getContext());
  patterns.insert<eir::TraceConstructOpLowering>(&getContext());
  */

  // We want to completely lower to LLVM, so we use a `FullConversion`.
  //
  // This ensures that only legal operations will remain after the conversion.
  auto mod = getModule();

  generateStaticInitializers(mod);
  if (M::failed(applyFullConversion(mod, target, patterns, &typeConverter)))
    signalPassFailure();
}

void EIRToLLVMLoweringPass::generateStaticInitializers(M::ModuleOp &mod) {
  // TODO: Generate static constructor which adds atoms from this module
  // to the atom table during init
  return;
}

std::unique_ptr<M::Pass> eir::createLowerToLLVMPass() {
  return std::make_unique<EIRToLLVMLoweringPass>();
}
