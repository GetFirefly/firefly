#include "lumen/compiler/Dialect/EIR/Conversion/EIRToStandard/ConvertEIRToStandard.h"

#include "lumen/compiler/Dialect/EIR/Conversion/TypeConverter.h"
#include "lumen/compiler/Dialect/EIR/IR/EIRTypes.h"
#include "lumen/compiler/Dialect/EIR/IR/EIRAttributes.h"
#include "lumen/compiler/Dialect/EIR/IR/EIROps.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace lumen {
namespace eir {
namespace {

template <typename T, typename Adaptor = typename T::OperandAdaptor>
class EIROpConversion : public mlir::OpConversionPattern<T> {
 public:
  EIROpConversion(mlir::MLIRContext *context,
                  mlir::TypeConverter &typeConverter,
                  TargetInfo &targetInfo)
      : OpConversionPattern<T>(context),
        targetInfo(targetInfo),
        typeConverter(typeConverter) {}

 protected:
  TargetInfo &targetInfo;
  TypeConverter &typeConverter;
};

struct UnreachableOpConversion : public EIROpConversion<eir::UnreachableOp> {
  using EIROpConversion::EIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(eir::UnreachableOp op,
                  llvm::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::LLVM::UnreachableOp>(op, operands);
    return matchSuccess();
  }
};

struct ConstantAtomOpConversion : public EIROpConversion<eir::ConstantAtomOp> {
  using EIROpConversion::EIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(eir::ConstantAtomOp op,
                  llvm::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto atomAttr = op.getValue().cast<AtomAttr>();
    auto i = atomAttr.getValue();
    auto ty = rewriter.getIntegerType(i.getBitWidth());
    auto val = rewriter.getIntegerAttr(ty, i);
    rewriter.replaceOpWithNewOp<mlir::ConstantOp>(op, ty, val);
    return matchSuccess();
  }
};
 
struct ConstantListOpConversion : public EIROpConversion<eir::ConstantListOp> {
  using EIROpConversion::EIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(eir::ConstantListOp op,
                  llvm::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto usizeTy = rewriter.getIntegerType(targetInfo.pointerSizeInBits);

    // FIXME
    auto val = rewriter.getIntegerAttr(usizeTy, 0);
    rewriter.replaceOpWithNewOp<mlir::ConstantOp>(op, usizeTy, val);
    return matchSuccess();
  }
};

struct ConstantTupleOpConversion : public EIROpConversion<eir::ConstantTupleOp> {
  using EIROpConversion::EIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(eir::ConstantTupleOp op,
                  llvm::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto usizeTy = rewriter.getIntegerType(targetInfo.pointerSizeInBits);
    auto seqAttr = op.getValue().cast<SeqAttr>();
    auto elements = seqAttr.getValue();
    llvm::SmallVector<Type, 2> elementTypes;
    llvm::SmallVector<Attribute, 2> newElements;
    for (auto element : elements) {
        auto elementType = element.getType();
        if (auto a = element.dyn_cast_or_null<eir::AtomAttr>()) {
            auto ai = a.getValue();
            auto val = rewriter.getIntegerAttr(usizeTy, ai);
            elementTypes.push_back(usizeTy);
            newElements.push_back(val);
        } else {
            assert(false && "unimplemented");
        }
    }
    auto ty = rewriter.getTupleType(elementTypes);
    auto val = rewriter.getArrayAttr(newElements);
    rewriter.replaceOpWithNewOp<mlir::ConstantOp>(op, ty, val);

    return matchSuccess();
  }
};

}  // namespace

void populateEIRToStandardPatterns(OwningRewritePatternList &patterns,
                                   mlir::MLIRContext *context,
                                   TypeConverter &typeConverter,
                                   TargetInfo &targetInfo) {
  patterns.insert<UnreachableOpConversion>(context, typeConverter, targetInfo);
  patterns.insert<ConstantAtomOpConversion>(context, typeConverter, targetInfo);
  patterns.insert<ConstantListOpConversion>(context, typeConverter, targetInfo);
  patterns.insert<ConstantTupleOpConversion>(context, typeConverter, targetInfo);
  patterns.insert<ConstantTupleOpConversion>(context, typeConverter, targetInfo);
}

namespace {

// A pass converting the EIR dialect into the Standard dialect.
class ConvertEIRToStandardPass : public mlir::ModulePass<ConvertEIRToStandardPass> {
 public:
  void runOnModule() override {
    TargetInfo targetInfo;
    StandardTypeConverter typeConverter(targetInfo);

    auto &context = getContext();
    mlir::ConversionTarget conversionTarget(context);
    conversionTarget.addLegalDialect<mlir::StandardOpsDialect>();
    conversionTarget.addLegalDialect<mlir::LLVM::LLVMDialect>();
    conversionTarget.addDynamicallyLegalOp<mlir::FuncOp>(
        [&](mlir::FuncOp op) { return typeConverter.isSignatureLegal(op.getType()); });

    OwningRewritePatternList patterns;
    populateFuncOpTypeConversionPattern(patterns, &context, typeConverter);
    populateEIRToStandardPatterns(patterns, &context, typeConverter, targetInfo);

    mlir::ModuleOp moduleOp = getModule();
    if (failed(applyFullConversion(moduleOp, conversionTarget,
                                   patterns, &typeConverter))) {
      moduleOp.emitError() << "conversion to standard dialect failed";
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OpPassBase<mlir::ModuleOp>> createConvertEIRToStandardPass() {
  return std::make_unique<ConvertEIRToStandardPass>();
}

static mlir::PassRegistration<ConvertEIRToStandardPass> pass(
    "lumen-convert-eir-to-std",
    "Convert the EIR dialect to the Standard dialect");

}  // namespace eir
}  // namespace lumen
