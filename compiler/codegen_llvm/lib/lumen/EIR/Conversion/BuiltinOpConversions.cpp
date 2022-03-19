#include "lumen/EIR/Conversion/BuiltinOpConversions.h"

namespace lumen {
namespace eir {

struct IncrementReductionsOpConversion
    : public EIROpConversion<IncrementReductionsOp> {
    using EIROpConversion::EIROpConversion;

    LogicalResult matchAndRewrite(
        IncrementReductionsOp op, ArrayRef<Value> operands,
        ConversionPatternRewriter &rewriter) const override {
        auto ctx = getRewriteContext(op, rewriter);

        auto i32Ty = ctx.getI32Type();

        auto reductionCount = ctx.getOrInsertGlobal(
            "CURRENT_REDUCTION_COUNT", i32Ty, nullptr, LLVM::Linkage::External,
            LLVM::ThreadLocalMode::LocalExec);

        Value increment = llvm_constant(i32Ty, ctx.getI32Attr(op.increment()));
        llvm_atomicrmw(i32Ty, LLVM::AtomicBinOp::add, reductionCount, increment,
                       LLVM::AtomicOrdering::monotonic);
        rewriter.eraseOp(op);
        return success();
    }
};

struct IsTypeOpConversion : public EIROpConversion<IsTypeOp> {
    using EIROpConversion::EIROpConversion;

    LogicalResult matchAndRewrite(
        IsTypeOp op, ArrayRef<Value> operands,
        ConversionPatternRewriter &rewriter) const override {
        IsTypeOpAdaptor adaptor(operands);
        auto ctx = getRewriteContext(op, rewriter);

        auto immedTy = ctx.getOpaqueImmediateType();
        auto termTy = ctx.getOpaqueTermType();
        auto termTyAddr0 = ctx.getOpaqueTermTypeAddr0();
        auto int1Ty = ctx.getI1Type();
        auto int32Ty = ctx.getI32Type();

        Value input = adaptor.value();

        auto matchType = op.getMatchType();
        auto matchTypeInfo = cast<TermTypeInterface>(matchType);
        // Boxed types and immediate types are dispatched differently
        if (matchType.isa<BoxType>() ||
            matchTypeInfo.isBoxable(ctx.targetInfo.immediateBits())) {
            Type boxedType;
            if (auto box = matchType.dyn_cast<BoxType>()) {
                boxedType = box.getPointeeType();
            } else {
                boxedType = matchType;
            }

            // Lists have a unique pointer tag, so we can avoid the function
            // call
            if (boxedType.isa<ConsType>()) {
                Value listTag = llvm_constant(
                    immedTy, ctx.getIntegerAttr(ctx.targetInfo.listTag()));
                Value listMask = llvm_constant(
                    immedTy, ctx.getIntegerAttr(ctx.targetInfo.listMask()));
                Value addr0 = llvm_addrspacecast(termTyAddr0,
                                                 llvm_bitcast(termTy, input));
                Value masked =
                    llvm_and(llvm_ptrtoint(immedTy, addr0), listMask);
                rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(
                    op, LLVM::ICmpPredicate::eq, listTag, masked);
                return success();
            }

            // For tuples we have a dedicated op
            if (auto tupleType = boxedType.dyn_cast_or_null<eir::TupleType>()) {
                auto size = tupleType.size();
                if (size > 0) {
                    Value arity =
                        llvm_constant(immedTy, ctx.getIntegerAttr(size));
                    rewriter.replaceOpWithNewOp<IsTupleOp>(op, adaptor.value(),
                                                           arity);
                    return success();
                } else {
                    rewriter.replaceOpWithNewOp<IsTupleOp>(op, adaptor.value(),
                                                           llvm::None);
                    return success();
                }
            }

            // For functions we have a dedicated op
            if (auto closureType =
                    boxedType.dyn_cast_or_null<eir::ClosureType>()) {
                rewriter.replaceOpWithNewOp<IsFunctionOp>(op, adaptor.value());
                return success();
            }

            // For all other boxed types, the check is performed via builtin
            StringRef symbolName("__lumen_builtin_is_boxed_type");

            auto boxedTypeInfo = cast<TermTypeInterface>(boxedType);
            auto matchKind = boxedTypeInfo.getTypeKind().getValue();
            Value matchConst =
                llvm_constant(int32Ty, ctx.getI32Attr(matchKind));
            auto callee =
                ctx.getOrInsertFunction(symbolName, int1Ty, {int32Ty, termTy});
            auto calleeSymbol = rewriter.getSymbolRefAttr(symbolName);
            rewriter.replaceOpWithNewOp<mlir::CallOp>(
                op, calleeSymbol, int1Ty, ValueRange{matchConst, input});
            return success();
        }

        // For immediates, the check is performed via builtin
        //
        // TODO: With some additional foundation-laying, we could lower
        // these checks to precise bit masking/shift operations, rather
        // than a function call
        auto matchKind = matchTypeInfo.getTypeKind().getValue();
        Value matchConst = llvm_constant(int32Ty, ctx.getI32Attr(matchKind));
        StringRef symbolName("__lumen_builtin_is_type");
        auto callee =
            ctx.getOrInsertFunction(symbolName, int1Ty, {int32Ty, termTy});
        auto calleeSymbol = rewriter.getSymbolRefAttr(symbolName);
        rewriter.replaceOpWithNewOp<mlir::CallOp>(
            op, calleeSymbol, int1Ty, ValueRange{matchConst, input});
        return success();
    }
};

struct IsTupleOpConversion : public EIROpConversion<IsTupleOp> {
    using EIROpConversion::EIROpConversion;

    LogicalResult matchAndRewrite(
        IsTupleOp op, ArrayRef<Value> operands,
        ConversionPatternRewriter &rewriter) const override {
        IsTupleOpAdaptor adaptor(operands);
        auto ctx = getRewriteContext(op, rewriter);

        Value input = adaptor.value();
        Value arity = adaptor.arity();
        auto termTy = ctx.getOpaqueTermType();
        auto int1Ty = ctx.getI1Type();
        auto int32Ty = ctx.getI32Type();

        // When an arity is given, we use a special builtin
        if (arity) {
            ArrayRef<Type> argTypes({termTy, termTy});
            StringRef symbolName("__lumen_builtin_is_tuple");
            auto callee = ctx.getOrInsertFunction(symbolName, int1Ty, argTypes);
            auto calleeSymbol = rewriter.getSymbolRefAttr(symbolName);
            rewriter.replaceOpWithNewOp<mlir::CallOp>(op, calleeSymbol, int1Ty,
                                                      ValueRange{arity, input});
            return success();
        }

        // Otherwise we fall back to the generic boxed type builtin
        Value matchConst =
            llvm_constant(int32Ty, ctx.getI32Attr(TypeKind::Tuple));
        StringRef symbolName("__lumen_builtin_is_boxed_type");
        auto callee =
            ctx.getOrInsertFunction(symbolName, int1Ty, {int32Ty, termTy});
        auto calleeSymbol = rewriter.getSymbolRefAttr(symbolName);
        rewriter.replaceOpWithNewOp<mlir::CallOp>(
            op, calleeSymbol, int1Ty, ValueRange{matchConst, input});
        return success();
    }
};

struct IsFunctionOpConversion : public EIROpConversion<IsFunctionOp> {
    using EIROpConversion::EIROpConversion;

    LogicalResult matchAndRewrite(
        IsFunctionOp op, ArrayRef<Value> operands,
        ConversionPatternRewriter &rewriter) const override {
        IsFunctionOpAdaptor adaptor(operands);
        auto ctx = getRewriteContext(op, rewriter);

        Value input = adaptor.value();
        Value arity = adaptor.arity();
        auto termTy = ctx.getOpaqueTermType();
        auto int1Ty = ctx.getI1Type();
        auto int32Ty = ctx.getI32Type();

        // When an arity is given, we use a special builtin
        if (arity) {
            ArrayRef<Type> argTypes({termTy, termTy});
            StringRef symbolName("__lumen_builtin_is_function");
            auto callee = ctx.getOrInsertFunction(symbolName, int1Ty, argTypes);
            auto calleeSymbol = rewriter.getSymbolRefAttr(symbolName);
            rewriter.replaceOpWithNewOp<mlir::CallOp>(op, calleeSymbol, int1Ty,
                                                      ValueRange{arity, input});
            return success();
        }

        // Otherwise we fall back to the generic boxed type builtin
        Value matchConst =
            llvm_constant(int32Ty, ctx.getI32Attr(TypeKind::Closure));
        StringRef symbolName("__lumen_builtin_is_boxed_type");
        auto callee =
            ctx.getOrInsertFunction(symbolName, int1Ty, {int32Ty, termTy});
        auto calleeSymbol = rewriter.getSymbolRefAttr(symbolName);
        rewriter.replaceOpWithNewOp<mlir::CallOp>(
            op, int1Ty, calleeSymbol, ValueRange{matchConst, input});
        return success();
    }
};

struct PrintOpConversion : public EIROpConversion<PrintOp> {
    using EIROpConversion::EIROpConversion;

    LogicalResult matchAndRewrite(
        PrintOp op, ArrayRef<Value> operands,
        ConversionPatternRewriter &rewriter) const override {
        auto ctx = getRewriteContext(op, rewriter);

        // If print is called with no operands, just remove it for now
        if (operands.empty()) {
            rewriter.eraseOp(op);
            return success();
        }

        auto termTy = ctx.getOpaqueTermType();
        StringRef symbolName("__lumen_builtin_printf");
        auto callee = ctx.getOrInsertFunction(symbolName, termTy, {termTy});

        auto calleeSymbol = rewriter.getSymbolRefAttr(symbolName);
        rewriter.replaceOpWithNewOp<mlir::CallOp>(op, calleeSymbol, termTy,
                                                  operands);
        return success();
    }
};

struct TraceCaptureOpConversion : public EIROpConversion<TraceCaptureOp> {
    using EIROpConversion::EIROpConversion;

    LogicalResult matchAndRewrite(
        TraceCaptureOp op, ArrayRef<Value> operands,
        ConversionPatternRewriter &rewriter) const override {
        auto ctx = getRewriteContext(op, rewriter);

        auto termTy = ctx.getOpaqueTermType();
        StringRef symbolName("__lumen_builtin_trace.capture");
        auto callee = ctx.getOrInsertFunction(symbolName, termTy, {});

        auto calleeSymbol = rewriter.getSymbolRefAttr(symbolName);
        rewriter.replaceOpWithNewOp<mlir::CallOp>(op, calleeSymbol,
                                                  ArrayRef<Type>{termTy});
        return success();
    }
};

struct TracePrintOpConversion : public EIROpConversion<TracePrintOp> {
    using EIROpConversion::EIROpConversion;

    LogicalResult matchAndRewrite(
        TracePrintOp op, ArrayRef<Value> operands,
        ConversionPatternRewriter &rewriter) const override {
        auto ctx = getRewriteContext(op, rewriter);
        TracePrintOpAdaptor adaptor(operands);

        Value kind = adaptor.kind();
        Value reason = adaptor.reason();
        Value traceRef = adaptor.traceRef();

        auto termTy = ctx.getOpaqueTermType();
        auto voidTy = ctx.getVoidType();

        StringRef symbolName("__lumen_builtin_trace.print");
        auto callee = ctx.getOrInsertFunction(symbolName, voidTy,
                                              {termTy, termTy, termTy});
        auto calleeSymbol = rewriter.getSymbolRefAttr(symbolName);
        rewriter.replaceOpWithNewOp<mlir::CallOp>(
            op, calleeSymbol, ArrayRef<Type>{},
            ArrayRef<Value>{kind, reason, traceRef});
        return success();
    }
};

struct TraceConstructOpConversion : public EIROpConversion<TraceConstructOp> {
    using EIROpConversion::EIROpConversion;

    LogicalResult matchAndRewrite(
        TraceConstructOp op, ArrayRef<Value> operands,
        ConversionPatternRewriter &rewriter) const override {
        auto ctx = getRewriteContext(op, rewriter);
        TraceConstructOpAdaptor adaptor(operands);

        Value traceRef = adaptor.traceRef();

        auto termTy = ctx.getOpaqueTermType();

        StringRef symbolName("__lumen_builtin_trace.construct");
        auto callee = ctx.getOrInsertFunction(symbolName, termTy, {termTy});

        auto calleeSymbol = rewriter.getSymbolRefAttr(symbolName);
        rewriter.replaceOpWithNewOp<mlir::CallOp>(op, calleeSymbol, termTy,
                                                  ValueRange(traceRef));
        return success();
    }
};

void populateBuiltinOpConversionPatterns(OwningRewritePatternList &patterns,
                                         MLIRContext *context,
                                         EirTypeConverter &converter,
                                         TargetPlatform &platform) {
    patterns.insert<IncrementReductionsOpConversion, IsTypeOpConversion,
                    IsTupleOpConversion, IsFunctionOpConversion,
                    PrintOpConversion, TraceCaptureOpConversion,
                    TraceConstructOpConversion, TracePrintOpConversion>(
        context, converter, platform);
}

}  // namespace eir
}  // namespace lumen
