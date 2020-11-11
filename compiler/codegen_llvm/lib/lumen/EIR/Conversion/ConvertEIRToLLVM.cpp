#include "lumen/EIR/Conversion/ConvertEIRToLLVM.h"

#include "mlir/Rewrite/FrozenRewritePatternList.h"

#include "lumen/EIR/Conversion/AggregateOpConversions.h"
#include "lumen/EIR/Conversion/BinaryOpConversions.h"
#include "lumen/EIR/Conversion/BuiltinOpConversions.h"
#include "lumen/EIR/Conversion/ComparisonOpConversions.h"
#include "lumen/EIR/Conversion/ConstantOpConversions.h"
#include "lumen/EIR/Conversion/ControlFlowOpConversions.h"
#include "lumen/EIR/Conversion/ConversionSupport.h"
#include "lumen/EIR/Conversion/FuncLikeOpConversions.h"
#include "lumen/EIR/Conversion/MapOpConversions.h"
#include "lumen/EIR/Conversion/MathOpConversions.h"
#include "lumen/EIR/Conversion/MemoryOpConversions.h"

using ::mlir::FrozenRewritePatternList;

namespace lumen {
namespace eir {

// A pass converting the EIR dialect into the Standard dialect.
class ConvertEIRToLLVMPass
    : public mlir::PassWrapper<ConvertEIRToLLVMPass,
                               mlir::OperationPass<ModuleOp>> {
   public:
    ConvertEIRToLLVMPass(TargetMachine *targetMachine_)
        : mlir::PassWrapper<ConvertEIRToLLVMPass,
                            mlir::OperationPass<ModuleOp>>(),
          targetMachine(targetMachine_) {}

    ConvertEIRToLLVMPass(const ConvertEIRToLLVMPass &other)
        : mlir::PassWrapper<ConvertEIRToLLVMPass,
                            mlir::OperationPass<ModuleOp>>(),
          targetMachine(other.targetMachine) {}

    void getDependentDialects(mlir::DialectRegistry &registry) const override {
        registry.insert<mlir::StandardOpsDialect, mlir::LLVM::LLVMDialect,
                        lumen::eir::eirDialect>();
    }

    void runOnOperation() final {
        // Create the LLVM type converter for lowering types to Standard/LLVM IR
        // types
        auto &context = getContext();
        auto targetInfo = TargetInfo(targetMachine, &context);

        auto llvmOpts = mlir::LowerToLLVMOptions::getDefaultOptions();
        llvmOpts.useAlignedAlloc = true;
        llvmOpts.dataLayout = targetMachine->createDataLayout();

        LLVMTypeConverter llvmConverter(&context, llvmOpts);
        EirTypeConverter converter(targetInfo.pointerSizeInBits, llvmConverter);
        // Initialize target-specific type information
        converter.addConversion([&](Type type) {
            return convertType(type, converter, targetInfo);
        });
        converter.addConversion([](LLVMType type) { return type; });

        // Populate conversion patterns
        OwningRewritePatternList patterns;

        // Add conversions from Standard to LLVM
        LLVMTypeConverter stdTypeConverter(&context, llvmOpts);
        stdTypeConverter.addConversion([&](Type type) {
            return convertType(type, converter, targetInfo);
        });
        mlir::populateStdToLLVMConversionPatterns(stdTypeConverter, patterns);

        // Add conversions from EIR to LLVM
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
        populateMapOpConversionPatterns(patterns, &context, converter,
                                        targetInfo);
        populateMathOpConversionPatterns(patterns, &context, converter,
                                         targetInfo);
        populateMemoryOpConversionPatterns(patterns, &context, converter,
                                           targetInfo);

        // Define the legality of the operations we're converting to
        mlir::ConversionTarget conversionTarget(context);
        conversionTarget.addLegalDialect<mlir::LLVM::LLVMDialect>();
        conversionTarget.addLegalOp<ModuleOp, mlir::ModuleTerminatorOp>();

        mlir::ModuleOp moduleOp = getOperation();
        FrozenRewritePatternList frozenPatterns(std::move(patterns));
        if (failed(applyFullConversion(moduleOp, conversionTarget,
                                       frozenPatterns))) {
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
