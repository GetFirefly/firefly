#include "lumen/compiler/Dialect/EIR/Conversion/EIRToLLVM/ComparisonOpConversions.h"

namespace lumen {
namespace eir {

template <typename Op, typename OperandAdaptor>
class ComparisonOpConversion : public EIROpConversion<Op> {
 public:
  explicit ComparisonOpConversion(MLIRContext *context,
                                  LLVMTypeConverter &converter_,
                                  TargetInfo &targetInfo_,
                                  mlir::PatternBenefit benefit = 1)
      : EIROpConversion<Op>::EIROpConversion(context, converter_, targetInfo_,
                                             benefit) {}

  PatternMatchResult matchAndRewrite(
      Op op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    edsc::ScopedContext context(rewriter, op.getLoc());
    OperandAdaptor adaptor(operands);
    StringRef builtinSymbol = Op::builtinSymbol();

    ModuleOp parentModule = op.template getParentOfType<ModuleOp>();
    auto termTy = getUsizeType();
    auto int1ty = getI1Type();
    auto callee = getOrInsertFunction(rewriter, parentModule, builtinSymbol,
                                      int1ty, {termTy, termTy});

    auto lhs = adaptor.lhs();
    auto rhs = adaptor.rhs();
    ArrayRef<Value> args({lhs, rhs});
    auto callOp = rewriter.create<mlir::CallOp>(op.getLoc(), callee,
                                                ArrayRef<Type>{int1ty}, args);
    auto result = callOp.getResult(0);

    rewriter.replaceOp(op, result);
    return matchSuccess();
  }

 private:
  using EIROpConversion<Op>::matchSuccess;
  using EIROpConversion<Op>::getUsizeType;
  using EIROpConversion<Op>::getI1Type;
  using EIROpConversion<Op>::getOrInsertFunction;
};

struct CmpEqOpConversion
    : public ComparisonOpConversion<CmpEqOp, CmpEqOpOperandAdaptor> {
  using ComparisonOpConversion::ComparisonOpConversion;
};
struct CmpNeqOpConversion
    : public ComparisonOpConversion<CmpNeqOp, CmpNeqOpOperandAdaptor> {
  using ComparisonOpConversion::ComparisonOpConversion;
};
struct CmpLtOpConversion
    : public ComparisonOpConversion<CmpLtOp, CmpLtOpOperandAdaptor> {
  using ComparisonOpConversion::ComparisonOpConversion;
};
struct CmpLteOpConversion
    : public ComparisonOpConversion<CmpLteOp, CmpLteOpOperandAdaptor> {
  using ComparisonOpConversion::ComparisonOpConversion;
};
struct CmpGtOpConversion
    : public ComparisonOpConversion<CmpGtOp, CmpGtOpOperandAdaptor> {
  using ComparisonOpConversion::ComparisonOpConversion;
};
struct CmpGteOpConversion
    : public ComparisonOpConversion<CmpGteOp, CmpGteOpOperandAdaptor> {
  using ComparisonOpConversion::ComparisonOpConversion;
};

void populateComparisonOpConversionPatterns(OwningRewritePatternList &patterns,
                                            MLIRContext *context,
                                            LLVMTypeConverter &converter,
                                            TargetInfo &targetInfo) {
  patterns.insert<CmpEqOpConversion, CmpNeqOpConversion, CmpLtOpConversion,
                  CmpLteOpConversion, CmpGtOpConversion, CmpGteOpConversion>(
      context, converter, targetInfo);
}

}  // namespace eir
}  // namespace lumen
