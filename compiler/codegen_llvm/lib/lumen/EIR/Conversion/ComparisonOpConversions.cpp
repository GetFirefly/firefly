#include "lumen/EIR/Conversion/ComparisonOpConversions.h"

namespace lumen {
namespace eir {

template <typename Op, typename OperandAdaptor>
class ComparisonOpConversion : public EIROpConversion<Op> {
   public:
    explicit ComparisonOpConversion(MLIRContext *context,
                                    EirTypeConverter &converter_,
                                    TargetPlatform &platform_,
                                    mlir::PatternBenefit benefit = 1)
        : EIROpConversion<Op>::EIROpConversion(context, converter_, platform_,
                                               benefit) {}

    LogicalResult matchAndRewrite(
        Op op, ArrayRef<Value> operands,
        ConversionPatternRewriter &rewriter) const override {
        OperandAdaptor adaptor(operands);
        auto ctx = getRewriteContext(op, rewriter);

        StringRef builtinSymbol = Op::builtinSymbol();

        auto termTy = ctx.getOpaqueTermType();
        auto int1ty = ctx.getI1Type();

        auto callee =
            ctx.getOrInsertFunction(builtinSymbol, int1ty, {termTy, termTy});

        Value lhs = adaptor.lhs();
        Value rhs = adaptor.rhs();

        ArrayRef<Value> args({lhs, rhs});
        auto calleeSymbol = rewriter.getSymbolRefAttr(builtinSymbol);
        Operation *callOp =
            std_call(calleeSymbol, ArrayRef<Type>{int1ty}, args);

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
        auto termTy = ctx.getOpaqueTermType();

        Value lhs = adaptor.lhs();
        Value rhs = adaptor.rhs();
        Type lhsType = op.lhs().getType();
        Type rhsType = op.rhs().getType();
        bool strict = false;
        if (auto attr = op->getAttrOfType<UnitAttr>("is_strict")) {
            strict = true;
        }

        bool useICmp = true;
        Value lhsOperand;
        Value rhsOperand;
        Optional<Type> targetType =
            ctx.typeConverter.coalesceOperandTypes(lhsType, rhsType);
        if (targetType.hasValue()) {
            // We were able to decide which type to lower to, insert casts where
            // necessary
            auto tt = targetType.getValue();
            if (lhsType != tt)
                lhsOperand = rewriter.create<CastOp>(op.getLoc(), lhs, tt);
            else
                lhsOperand = lhs;
            if (rhsType != tt)
                rhsOperand = rewriter.create<CastOp>(op.getLoc(), rhs, tt);
            else
                rhsOperand = rhs;
        } else {
            useICmp = false;
            if (lhsType.isa<TermType>())
                lhsOperand = lhs;
            else
                lhsOperand = rewriter.create<CastOp>(
                    op.getLoc(), lhs, rewriter.getType<TermType>());
            if (rhsType.isa<TermType>())
                rhsOperand = rhs;
            else
                rhsOperand = rewriter.create<CastOp>(
                    op.getLoc(), rhs, rewriter.getType<TermType>());
        }

        if (strict && useICmp) {
            rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(
                op, LLVM::ICmpPredicate::eq, lhsOperand, rhsOperand);
            return success();
        }

        // If we reach here, fall back to the slow path
        StringRef builtinSymbol;
        if (strict)
            builtinSymbol = "__lumen_builtin_cmp.eq.strict";
        else
            builtinSymbol = "__lumen_builtin_cmp.eq";

        auto callee =
            ctx.getOrInsertFunction(builtinSymbol, i1Ty, {termTy, termTy});

        auto calleeSymbol = rewriter.getSymbolRefAttr(builtinSymbol);
        Operation *callOp =
            std_call(calleeSymbol, ArrayRef<Type>{i1Ty}, ValueRange{lhs, rhs});

        rewriter.replaceOp(op, callOp->getResult(0));
        return success();
    }
};

void populateComparisonOpConversionPatterns(OwningRewritePatternList &patterns,
                                            MLIRContext *context,
                                            EirTypeConverter &converter,
                                            TargetPlatform &platform) {
    patterns.insert<CmpEqOpConversion, CmpLtOpConversion, CmpLteOpConversion,
                    CmpGtOpConversion, CmpGteOpConversion>(context, converter,
                                                           platform);
}

}  // namespace eir
}  // namespace lumen
