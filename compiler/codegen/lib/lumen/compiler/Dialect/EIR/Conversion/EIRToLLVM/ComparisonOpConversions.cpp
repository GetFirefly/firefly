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

  LogicalResult matchAndRewrite(
      Op op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    OperandAdaptor adaptor(operands);
    auto ctx = getRewriteContext(op, rewriter);

    StringRef builtinSymbol = Op::builtinSymbol();

    auto termTy = ctx.getUsizeType();
    auto int1ty = ctx.getI1Type();

    auto callee =
        ctx.getOrInsertFunction(builtinSymbol, int1ty, {termTy, termTy});

    auto lhs = adaptor.lhs();
    auto rhs = adaptor.rhs();
    ArrayRef<Value> args({lhs, rhs});
    auto calleeSymbol =
        FlatSymbolRefAttr::get(builtinSymbol, callee->getContext());
    Operation *callOp = std_call(calleeSymbol, ArrayRef<Type>{int1ty}, args);

    rewriter.replaceOp(op, callOp->getResult(0));
    return success();
  }

 private:
  using EIROpConversion<Op>::getRewriteContext;
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
