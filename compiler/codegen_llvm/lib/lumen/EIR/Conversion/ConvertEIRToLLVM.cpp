#include "lumen/EIR/Conversion/ConvertEIRToLLVM.h"

#include "llvm/Target/TargetMachine.h"
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

using ::llvm::Optional;
using ::llvm::TargetMachine;
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
        TargetPlatform platform(targetMachine->getTargetTriple());

        auto llvmOpts = mlir::LowerToLLVMOptions::getDefaultOptions();
        llvmOpts.useAlignedAlloc = true;
        llvmOpts.dataLayout = targetMachine->createDataLayout();

        LLVMTypeConverter llvmConverter(&context, llvmOpts);
        EirTypeConverter converter(platform.getPointerWidth(), llvmConverter);
        // Initialize target-specific type information
        converter.addConversion([&](Type type) -> Optional<Type> {
            return convertType(type, converter, platform);
        });

        // Populate conversion patterns
        OwningRewritePatternList patterns;

        // Add conversions from Standard to LLVM
        LLVMTypeConverter stdTypeConverter(&context, llvmOpts);
        stdTypeConverter.addConversion([&](Type type) -> Optional<Type> {
            return convertType(type, converter, platform);
        });
        mlir::populateStdToLLVMConversionPatterns(stdTypeConverter, patterns);

        // Add conversions from EIR to LLVM
        populateAggregateOpConversionPatterns(patterns, &context, converter,
                                              platform);
        populateBinaryOpConversionPatterns(patterns, &context, converter,
                                           platform);
        populateBuiltinOpConversionPatterns(patterns, &context, converter,
                                            platform);
        populateComparisonOpConversionPatterns(patterns, &context, converter,
                                               platform);
        populateConstantOpConversionPatterns(patterns, &context, converter,
                                             platform);
        populateControlFlowOpConversionPatterns(patterns, &context, converter,
                                                platform);
        populateFuncLikeOpConversionPatterns(patterns, &context, converter,
                                             platform);
        populateMapOpConversionPatterns(patterns, &context, converter,
                                        platform);
        populateMathOpConversionPatterns(patterns, &context, converter,
                                         platform);
        populateMemoryOpConversionPatterns(patterns, &context, converter,
                                           platform);

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
