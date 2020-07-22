#include "lumen/compiler/Dialect/EIR/Conversion/EIRToLLVM/MathOpConversions.h"

namespace lumen {
namespace eir {
template <typename Op, typename T>
static Value specializeIntegerMathOp(RewritePatternContext<Op> &ctx, Value lhs,
                                     Value rhs) {
  Value lhsInt = ctx.decodeImmediate(lhs);
  Value rhsInt = ctx.decodeImmediate(rhs);
  auto mathOp = ctx.rewriter.template create<T>(ctx.getLoc(), lhsInt, rhsInt);
  return mathOp.getResult();
}

template <typename Op, typename T>
static Value specializeFloatMathOp(RewritePatternContext<Op> &ctx, Value lhs,
                                   Value rhs) {
  auto fpTy = LLVMType::getDoubleTy(ctx.dialect);
  Value l = eir_cast(lhs, fpTy);
  Value r = eir_cast(rhs, fpTy);
  auto fpOp = ctx.rewriter.template create<T>(ctx.getLoc(), l, r);
  return fpOp.getResult();
}

template <typename Op, typename OperandAdaptor, typename IntOp,
          typename FloatOp>
class MathOpConversion : public EIROpConversion<Op> {
 public:
  explicit MathOpConversion(MLIRContext *context, LLVMTypeConverter &converter_,
                            TargetInfo &targetInfo_,
                            mlir::PatternBenefit benefit = 1)
      : EIROpConversion<Op>::EIROpConversion(context, converter_, targetInfo_,
                                             benefit) {}

  LogicalResult matchAndRewrite(
      Op op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    OperandAdaptor adaptor(operands);
    auto ctx = getRewriteContext(op, rewriter);

    Value lhs = adaptor.lhs();
    Value rhs = adaptor.rhs();
    Type lhsTy = op.getOperand(0).getType();
    Type rhsTy = op.getOperand(1).getType();

    // Use specialized lowerings if types are compatible
    if (lhsTy.isa<FixnumType>() && rhsTy.isa<FixnumType>()) {
      auto newOp = specializeIntegerMathOp<Op, IntOp>(ctx, lhs, rhs);
      rewriter.replaceOp(op, newOp);
      return success();
    }
    if (lhsTy.isa<FloatType>() && rhsTy.isa<FloatType>()) {
      auto newOp = specializeFloatMathOp<Op, FloatOp>(ctx, lhs, rhs);
      rewriter.replaceOp(op, newOp);
      return success();
    }

    // Call builtin function
    StringRef builtinSymbol = Op::builtinSymbol();
    auto termTy = ctx.getUsizeType();
    auto callee =
        ctx.getOrInsertFunction(builtinSymbol, termTy, {termTy, termTy});

    ArrayRef<Value> args({lhs, rhs});
    auto calleeSymbol =
        FlatSymbolRefAttr::get(builtinSymbol, callee->getContext());

    rewriter.replaceOpWithNewOp<mlir::CallOp>(op, calleeSymbol,
                                              ArrayRef<Type>{termTy}, args);
    return success();
  }

 private:
  using EIROpConversion<Op>::getRewriteContext;
};

struct AddOpConversion : public MathOpConversion<AddOp, AddOpOperandAdaptor,
                                                 LLVM::AddOp, LLVM::FAddOp> {
  using MathOpConversion::MathOpConversion;
};
struct SubOpConversion : public MathOpConversion<SubOp, SubOpOperandAdaptor,
                                                 LLVM::SubOp, LLVM::FSubOp> {
  using MathOpConversion::MathOpConversion;
};
struct MulOpConversion : public MathOpConversion<MulOp, MulOpOperandAdaptor,
                                                 LLVM::MulOp, LLVM::FMulOp> {
  using MathOpConversion::MathOpConversion;
};

template <typename Op, typename OperandAdaptor, typename IntOp>
class IntegerMathOpConversion : public EIROpConversion<Op> {
 public:
  explicit IntegerMathOpConversion(MLIRContext *context,
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

    Value lhs = adaptor.lhs();
    Value rhs = adaptor.rhs();
    Type lhsTy = op.getOperand(0).getType();
    Type rhsTy = op.getOperand(1).getType();

    // Use specialized lowerings if types are compatible
    if (lhsTy.isa<FixnumType>() && rhsTy.isa<FixnumType>()) {
      auto newOp = specializeIntegerMathOp<Op, IntOp>(ctx, lhs, rhs);
      rewriter.replaceOp(op, newOp);
      return success();
    }

    // Call builtin function
    StringRef builtinSymbol = Op::builtinSymbol();
    auto termTy = ctx.getUsizeType();
    auto callee =
        ctx.getOrInsertFunction(builtinSymbol, termTy, {termTy, termTy});

    ArrayRef<Value> args({lhs, rhs});
    auto calleeSymbol =
        FlatSymbolRefAttr::get(builtinSymbol, callee->getContext());
    rewriter.replaceOpWithNewOp<mlir::CallOp>(op, calleeSymbol,
                                              ArrayRef<Type>{termTy}, args);
    return success();
  }

 private:
  using EIROpConversion<Op>::getRewriteContext;
};

struct DivOpConversion
    : public IntegerMathOpConversion<DivOp, DivOpOperandAdaptor, LLVM::SDivOp> {
  using IntegerMathOpConversion::IntegerMathOpConversion;
};
struct RemOpConversion
    : public IntegerMathOpConversion<RemOp, RemOpOperandAdaptor, LLVM::SRemOp> {
  using IntegerMathOpConversion::IntegerMathOpConversion;
};
struct BandOpConversion
    : public IntegerMathOpConversion<BandOp, BandOpOperandAdaptor,
                                     LLVM::AndOp> {
  using IntegerMathOpConversion::IntegerMathOpConversion;
};
struct BorOpConversion
    : public IntegerMathOpConversion<BorOp, BorOpOperandAdaptor, LLVM::OrOp> {
  using IntegerMathOpConversion::IntegerMathOpConversion;
};
struct BxorOpConversion
    : public IntegerMathOpConversion<BxorOp, BxorOpOperandAdaptor,
                                     LLVM::XOrOp> {
  using IntegerMathOpConversion::IntegerMathOpConversion;
};
struct BslOpConversion
    : public IntegerMathOpConversion<BslOp, BslOpOperandAdaptor, LLVM::ShlOp> {
  using IntegerMathOpConversion::IntegerMathOpConversion;
};
struct BsrOpConversion
    : public IntegerMathOpConversion<BsrOp, BsrOpOperandAdaptor, LLVM::AShrOp> {
  using IntegerMathOpConversion::IntegerMathOpConversion;
};

template <typename Op, typename OperandAdaptor, typename FloatOp>
class FloatMathOpConversion : public EIROpConversion<Op> {
 public:
  explicit FloatMathOpConversion(MLIRContext *context,
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

    Value lhs = adaptor.lhs();
    Value rhs = adaptor.rhs();
    Type lhsTy = op.getOperand(0).getType();
    Type rhsTy = op.getOperand(1).getType();

    // Use specialized lowerings if types are compatible
    if (lhsTy.isa<FloatType>() && rhsTy.isa<FloatType>()) {
      auto newOp = specializeIntegerMathOp<Op, FloatOp>(ctx, lhs, rhs);
      rewriter.replaceOp(op, newOp);
      return success();
    }

    // Call builtin function
    StringRef builtinSymbol = Op::builtinSymbol();
    auto termTy = ctx.getUsizeType();
    auto callee =
        ctx.getOrInsertFunction(builtinSymbol, termTy, {termTy, termTy});

    ArrayRef<Value> args({lhs, rhs});
    auto calleeSymbol =
        FlatSymbolRefAttr::get(builtinSymbol, callee->getContext());

    rewriter.replaceOpWithNewOp<mlir::CallOp>(op, calleeSymbol,
                                              ArrayRef<Type>{termTy}, args);
    return success();
  }

 private:
  using EIROpConversion<Op>::getRewriteContext;
};

struct FDivOpConversion
    : public FloatMathOpConversion<FDivOp, FDivOpOperandAdaptor, LLVM::FDivOp> {
  using FloatMathOpConversion::FloatMathOpConversion;
};

void populateMathOpConversionPatterns(OwningRewritePatternList &patterns,
                                      MLIRContext *context,
                                      LLVMTypeConverter &converter,
                                      TargetInfo &targetInfo) {
  patterns.insert<AddOpConversion, SubOpConversion, MulOpConversion,
                  DivOpConversion, FDivOpConversion, RemOpConversion,
                  BslOpConversion, BsrOpConversion, BandOpConversion,
                  BorOpConversion, BxorOpConversion>(context, converter,
                                                     targetInfo);
}

}  // namespace eir
}  // namespace lumen
