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
    ModuleOp mod = ctx.getModule();

    auto i32Ty = ctx.getI32Type();

    auto reductionCount = ctx.getOrInsertGlobal(
        "CURRENT_REDUCTION_COUNT", i32Ty, nullptr, LLVM::Linkage::External,
        LLVM::ThreadLocalMode::LocalExec);

    auto incBy = op.increment().getLimitedValue();
    Value increment = llvm_constant(i32Ty, ctx.getI32Attr(incBy));
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

    auto termTy = ctx.getUsizeType();
    auto int1Ty = ctx.getI1Type();
    auto int32Ty = ctx.getI32Type();

    auto matchType = op.getMatchType().cast<OpaqueTermType>();
    // Boxed types and immediate types are dispatched differently
    if (matchType.isBox() || matchType.isBoxable()) {
      OpaqueTermType boxedType;
      if (matchType.isBox()) {
        boxedType = matchType.cast<BoxType>().getBoxedType();
      } else {
        boxedType = matchType;
      }

      // Lists have a unique pointer tag, so we can avoid the function call
      if (boxedType.isa<ConsType>()) {
        Value listTag =
            llvm_constant(termTy, ctx.getIntegerAttr(ctx.targetInfo.listTag()));
        Value listMask = llvm_constant(
            termTy, ctx.getIntegerAttr(ctx.targetInfo.listMask()));
        Value masked = llvm_and(adaptor.value(), listMask);
        rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(op, LLVM::ICmpPredicate::eq,
                                                  listTag, masked);
        return success();
      }

      // For tuples with static shape, we use a specialized builtin
      if (auto tupleType = boxedType.dyn_cast_or_null<eir::TupleType>()) {
        if (tupleType.hasStaticShape()) {
          Value arity =
              llvm_constant(termTy, ctx.getIntegerAttr(tupleType.getArity()));
          ArrayRef<LLVMType> argTypes({termTy, termTy});
          StringRef symbolName("__lumen_builtin_is_tuple");
          auto callee = ctx.getOrInsertFunction(symbolName, int1Ty, argTypes);
          auto calleeSymbol =
              FlatSymbolRefAttr::get(symbolName, callee->getContext());
          Operation *isType = std_call(calleeSymbol, int1Ty,
                                       ArrayRef<Value>{arity, adaptor.value()});
          rewriter.replaceOp(op, isType->getResults());
          return success();
        }
      }

      // For all other boxed types, the check is performed via builtin
      auto matchKind = boxedType.getTypeKind().getValue();
      Value matchConst = llvm_constant(int32Ty, ctx.getI32Attr(matchKind));
      StringRef symbolName("__lumen_builtin_is_boxed_type");
      auto callee =
          ctx.getOrInsertFunction(symbolName, int1Ty, {int32Ty, termTy});
      Value input = adaptor.value();
      auto calleeSymbol =
          FlatSymbolRefAttr::get(symbolName, callee->getContext());
      Operation *isType =
          std_call(calleeSymbol, int1Ty, ArrayRef<Value>{matchConst, input});
      rewriter.replaceOp(op, isType->getResults());
      return success();
    }

    // For immediates, the check is performed via builtin
    //
    // TODO: With some additional foundation-laying, we could lower
    // these checks to precise bit masking/shift operations, rather
    // than a function call
    auto matchKind = matchType.getTypeKind().getValue();
    Value matchConst = llvm_constant(int32Ty, ctx.getI32Attr(matchKind));
    StringRef symbolName("__lumen_builtin_is_type");
    auto callee =
        ctx.getOrInsertFunction(symbolName, int1Ty, {int32Ty, termTy});
    auto calleeSymbol =
        FlatSymbolRefAttr::get(symbolName, callee->getContext());
    Operation *isType = std_call(calleeSymbol, int1Ty,
                                 ArrayRef<Value>{matchConst, adaptor.value()});
    rewriter.replaceOp(op, isType->getResults());
    return success();
  }
};

struct MallocOpConversion : public EIROpConversion<MallocOp> {
  using EIROpConversion::EIROpConversion;

  LogicalResult matchAndRewrite(
      MallocOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    MallocOpAdaptor adaptor(operands);
    auto ctx = getRewriteContext(op, rewriter);

    OpaqueTermType innerTy = op.getAllocType();
    auto ty = ctx.typeConverter.convertType(innerTy).cast<LLVMType>();

    if (innerTy.hasDynamicExtent()) {
      Value allocPtr = ctx.buildMalloc(ty, innerTy.getTypeKind().getValue(),
                                       adaptor.arity());
      rewriter.replaceOp(op, allocPtr);
    } else {
      Value zero = llvm_constant(ctx.getUsizeType(), ctx.getIntegerAttr(0));
      Value allocPtr =
          ctx.buildMalloc(ty, innerTy.getTypeKind().getValue(), zero);
      rewriter.replaceOp(op, allocPtr);
    }

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

    auto termTy = ctx.getUsizeType();
    StringRef symbolName("__lumen_builtin_printf");
    auto callee = ctx.getOrInsertFunction(symbolName, termTy, {termTy});

    auto calleeSymbol =
        FlatSymbolRefAttr::get(symbolName, callee->getContext());
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

    auto termTy = ctx.getUsizeType();
    auto termPtrTy = termTy.getPointerTo();
    StringRef symbolName("__lumen_builtin_trace.capture");
    auto callee = ctx.getOrInsertFunction(symbolName, termPtrTy, {});

    auto calleeSymbol =
        FlatSymbolRefAttr::get(symbolName, callee->getContext());
    rewriter.replaceOpWithNewOp<mlir::CallOp>(op, calleeSymbol,
                                              ArrayRef<Type>{termPtrTy});
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

    auto termTy = ctx.getUsizeType();
    auto termPtrTy = termTy.getPointerTo();
    auto voidTy = LLVMType::getVoidTy(ctx.context);

    StringRef symbolName("__lumen_builtin_trace.print");
    auto callee = ctx.getOrInsertFunction(symbolName, voidTy,
                                          {termTy, termTy, termPtrTy});
    auto calleeSymbol =
        FlatSymbolRefAttr::get(symbolName, callee->getContext());
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

    auto termTy = ctx.getUsizeType();
    auto termPtrTy = termTy.getPointerTo();

    StringRef symbolName("__lumen_builtin_trace.construct");
    auto callee = ctx.getOrInsertFunction(symbolName, termTy, {termPtrTy});

    auto calleeSymbol =
        FlatSymbolRefAttr::get(symbolName, callee->getContext());
    rewriter.replaceOpWithNewOp<mlir::CallOp>(op, calleeSymbol, termTy,
                                              ValueRange(traceRef));
    return success();
  }
};

void populateBuiltinOpConversionPatterns(OwningRewritePatternList &patterns,
                                         MLIRContext *context,
                                         EirTypeConverter &converter,
                                         TargetInfo &targetInfo) {
  patterns
      .insert<IncrementReductionsOpConversion, IsTypeOpConversion,
              PrintOpConversion, MallocOpConversion, TraceCaptureOpConversion,
              TraceConstructOpConversion, TracePrintOpConversion>(
          context, converter, targetInfo);
}

}  // namespace eir
}  // namespace lumen
