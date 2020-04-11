#include "lumen/compiler/Dialect/EIR/Conversion/EIRToLLVM/ConvertEIRToLLVM.h"

#include "lumen/compiler/Dialect/EIR/Conversion/EIRToLLVM/AggregateOpConversions.h"
#include "lumen/compiler/Dialect/EIR/Conversion/EIRToLLVM/BinaryOpConversions.h"
#include "lumen/compiler/Dialect/EIR/Conversion/EIRToLLVM/BuiltinOpConversions.h"
#include "lumen/compiler/Dialect/EIR/Conversion/EIRToLLVM/ComparisonOpConversions.h"
#include "lumen/compiler/Dialect/EIR/Conversion/EIRToLLVM/ConstantOpConversions.h"
#include "lumen/compiler/Dialect/EIR/Conversion/EIRToLLVM/ControlFlowOpConversions.h"
#include "lumen/compiler/Dialect/EIR/Conversion/EIRToLLVM/ConversionSupport.h"
#include "lumen/compiler/Dialect/EIR/Conversion/EIRToLLVM/FuncLikeOpConversions.h"
#include "lumen/compiler/Dialect/EIR/Conversion/EIRToLLVM/MapOpConversions.h"
#include "lumen/compiler/Dialect/EIR/Conversion/EIRToLLVM/MathOpConversions.h"
#include "lumen/compiler/Dialect/EIR/Conversion/EIRToLLVM/MemoryOpConversions.h"

namespace lumen {
namespace eir {

// A pass converting the EIR dialect into the Standard dialect.
class ConvertEIRToLLVMPass : public mlir::PassWrapper<ConvertEIRToLLVMPass, OperationPass<ModuleOp>> {
 public:
  ConvertEIRToLLVMPass(TargetMachine *targetMachine_)
      : targetMachine(targetMachine_),
        mlir::PassWrapper<ConvertEIRToLLVMPass, OperationPass<ModuleOp>>() {}

  ConvertEIRToLLVMPass(const ConvertEIRToLLVMPass &other)
      : targetMachine(other.targetMachine),
        mlir::PassWrapper<ConvertEIRToLLVMPass, OperationPass<ModuleOp>>() {}

  void runOnOperation() final {
    // Create the type converter for lowering types to Standard/LLVM IR types
    auto &context = getContext();
    LLVMTypeConverter converter(&context);

    // Initialize target-specific type information, using
    // the LLVMDialect contained in the type converter to
    // create named types
    auto targetInfo = TargetInfo(targetMachine, *converter.getDialect());

    // Populate conversion patterns
    OwningRewritePatternList patterns;
    mlir::populateStdToLLVMConversionPatterns(converter, patterns,
                                              /*emitCWrappers=*/false);
    populateAggregateOpConversionPatterns(patterns, &context, converter,
                                          targetInfo);
    populateBinaryOpConversionPatterns(patterns, &context, converter,
                                       targetInfo);
    populateBuiltinOpConversionPatterns(patterns, &context, converter,
                                        targetInfo);
    populateComparisonOpConversionPatterns(patterns, &context, converter,
                                           targetInfo);
    populateConstantOpConversionPatterns(patterns, &context, converter,
                                         targetInfo);
    populateControlFlowOpConversionPatterns(patterns, &context, converter,
                                            targetInfo);
    populateFuncLikeOpConversionPatterns(patterns, &context, converter,
                                         targetInfo);
    populateMapOpConversionPatterns(patterns, &context, converter, targetInfo);
    populateMathOpConversionPatterns(patterns, &context, converter, targetInfo);
    populateMemoryOpConversionPatterns(patterns, &context, converter,
                                       targetInfo);

    // Populate the type conversions for EIR types.
    converter.addConversion(
        [&](Type type) { return convertType(type, converter, targetInfo); });

    // Define the legality of the operations we're converting to
    mlir::ConversionTarget conversionTarget(context);
    conversionTarget.addLegalDialect<mlir::LLVM::LLVMDialect>();
    conversionTarget.addDynamicallyLegalOp<mlir::FuncOp>([&](mlir::FuncOp op) {
      return converter.isSignatureLegal(op.getType());
    });
    conversionTarget.addLegalOp<ModuleOp, ModuleTerminatorOp>();

    mlir::ModuleOp moduleOp = getOperation();
    if (failed(applyFullConversion(moduleOp, conversionTarget, patterns,
                                   &converter))) {
      moduleOp.emitError() << "conversion to LLVM IR dialect failed";
      return signalPassFailure();
    }
  }

 private:
  TargetMachine *targetMachine;
};

std::unique_ptr<mlir::Pass> createConvertEIRToLLVMPass(
    TargetMachine *targetMachine) {
  return std::make_unique<ConvertEIRToLLVMPass>(targetMachine);
}

}  // namespace eir
}  // namespace lumen
