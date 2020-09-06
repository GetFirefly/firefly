#include "lumen/EIR/Conversion/ComparisonOpConversions.h"

namespace lumen {
namespace eir {

template <typename Op, typename OperandAdaptor>
class ComparisonOpConversion : public EIROpConversion<Op> {
 public:
  explicit ComparisonOpConversion(MLIRContext *context,
                                  EirTypeConverter &converter_,
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

    Value lhs = adaptor.lhs();
    Value rhs = adaptor.rhs();

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

struct CmpLtOpConversion
    : public ComparisonOpConversion<CmpLtOp, CmpLtOpAdaptor> {
  using ComparisonOpConversion::ComparisonOpConversion;
};
struct CmpLteOpConversion
    : public ComparisonOpConversion<CmpLteOp, CmpLteOpAdaptor> {
  using ComparisonOpConversion::ComparisonOpConversion;
};
struct CmpGtOpConversion
    : public ComparisonOpConversion<CmpGtOp, CmpGtOpAdaptor> {
  using ComparisonOpConversion::ComparisonOpConversion;
};
struct CmpGteOpConversion
    : public ComparisonOpConversion<CmpGteOp, CmpGteOpAdaptor> {
  using ComparisonOpConversion::ComparisonOpConversion;
};

struct CmpEqOpConversion : public EIROpConversion<CmpEqOp> {
  using EIROpConversion::EIROpConversion;

  LogicalResult matchAndRewrite(
      CmpEqOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    CmpEqOpAdaptor adaptor(operands);
    auto ctx = getRewriteContext(op, rewriter);
    auto i1Ty = ctx.getI1Type();

    Value lhs = adaptor.lhs();
    Value rhs = adaptor.rhs();
    Type lhsType = op.lhs().getType();
    Type rhsType = op.rhs().getType();
    bool strict = false;
    if (auto attr = op.getAttrOfType<BoolAttr>("strict")) {
      strict = attr.getValue();
    }

    Value lhsOperand;
    Value rhsOperand;
    bool useICmp = true;
    if (auto lTy = lhsType.dyn_cast_or_null<OpaqueTermType>()) {
      if (auto rTy = rhsType.dyn_cast_or_null<OpaqueTermType>()) {
        if (lTy.isBoolean() && rTy.isBoolean()) {
          lhsOperand = lhs;
          rhsOperand = rhs;
        } else if (lTy.isBoolean() && rTy.isAtom()) {
          lhsOperand = lhs;
          rhsOperand = rewriter.create<CastOp>(op.getLoc(), rhs, i1Ty);
        } else if (lTy.isAtom() && rTy.isBoolean()) {
          lhsOperand = rewriter.create<CastOp>(op.getLoc(), lhs, i1Ty);
        } else {
          lhsOperand = lhs;
          rhsOperand = rhs;
          useICmp = strict && lTy.isImmediate() && rTy.isImmediate();
        }
      } else {
        op.emitError("invalid type for rhs operand");
        return failure();
      }
    } else {
      op.emitError("invalid type for lhs operand");
      return failure();
    }

    if (useICmp) {
      rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(op, LLVM::ICmpPredicate::eq, lhsOperand, rhsOperand);
      return success();
    }

    // If we reach here, fall back to the slow path
    auto termTy = ctx.getUsizeType();

    StringRef builtinSymbol = CmpEqOp::builtinSymbol();
    auto callee =
        ctx.getOrInsertFunction(builtinSymbol, i1Ty, {termTy, termTy});

    ArrayRef<Value> args({lhs, rhs});
    auto calleeSymbol =
        FlatSymbolRefAttr::get(builtinSymbol, callee->getContext());
    Operation *callOp = std_call(calleeSymbol, ArrayRef<Type>{i1Ty}, args);

    rewriter.replaceOp(op, callOp->getResult(0));
    return success();
  }
};

struct CmpNeqOpConversion : public EIROpConversion<CmpNeqOp> {
  using EIROpConversion::EIROpConversion;

  LogicalResult matchAndRewrite(
      CmpNeqOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    CmpNeqOpAdaptor adaptor(operands);

    auto eqOp = rewriter.create<CmpEqOp>(op.getLoc(), adaptor.lhs(), adaptor.rhs());
    Value isEqual = eqOp.getResult();
    rewriter.replaceOpWithNewOp<LLVM::XOrOp>(op, isEqual, isEqual);
    return success();
  }
};

void populateComparisonOpConversionPatterns(OwningRewritePatternList &patterns,
                                            MLIRContext *context,
                                            EirTypeConverter &converter,
                                            TargetInfo &targetInfo) {
  patterns.insert<CmpEqOpConversion, CmpNeqOpConversion, CmpLtOpConversion,
                  CmpLteOpConversion, CmpGtOpConversion, CmpGteOpConversion>(
      context, converter, targetInfo);
}

}  // namespace eir
}  // namespace lumen
