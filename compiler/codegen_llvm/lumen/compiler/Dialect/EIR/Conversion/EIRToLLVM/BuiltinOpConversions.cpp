#include "lumen/compiler/Dialect/EIR/Conversion/EIRToLLVM/BuiltinOpConversions.h"

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
    IsTypeOpOperandAdaptor adaptor(operands);
    auto ctx = getRewriteContext(op, rewriter);

    auto termTy = ctx.getUsizeType();
    auto int1Ty = ctx.getI1Type();
    auto int32Ty = ctx.getI32Type();

    auto matchType = op.getMatchType().cast<OpaqueTermType>();
    // Boxed types and immediate types are dispatched differently
    if (matchType.isBox()) {
      auto boxType = matchType.cast<BoxType>();
      auto boxedType = boxType.getBoxedType();

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
      auto matchKind = boxedType.getForeignKind();
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
    auto matchKind = matchType.getForeignKind();
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
    MallocOpOperandAdaptor adaptor(operands);
    auto ctx = getRewriteContext(op, rewriter);

    OpaqueTermType innerTy = op.getAllocType();
    auto ty = ctx.typeConverter.convertType(innerTy).cast<LLVMType>();

    if (innerTy.hasDynamicExtent()) {
      Value allocPtr =
          ctx.buildMalloc(ty, innerTy.getKind(), adaptor.arity().front());
      rewriter.replaceOp(op, allocPtr);
    } else {
      Value zero = llvm_constant(ctx.getUsizeType(), ctx.getIntegerAttr(0));
      Value allocPtr = ctx.buildMalloc(ty, innerTy.getKind(), zero);
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
    StringRef symbolName("__lumen_builtin_trace_capture");
    auto callee = ctx.getOrInsertFunction(symbolName, termTy, {});

    auto calleeSymbol =
        FlatSymbolRefAttr::get(symbolName, callee->getContext());
    rewriter.replaceOpWithNewOp<mlir::CallOp>(op, calleeSymbol,
                                              ArrayRef<Type>{termTy});
    return success();
  }
};

struct TraceConstructOpConversion : public EIROpConversion<TraceConstructOp> {
  using EIROpConversion::EIROpConversion;

  LogicalResult matchAndRewrite(
      TraceConstructOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = getRewriteContext(op, rewriter);

    auto termTy = ctx.getUsizeType();
    StringRef symbolName("__lumen_builtin_trace_construct");
    auto callee = ctx.getOrInsertFunction(symbolName, termTy, {});

    auto calleeSymbol =
        FlatSymbolRefAttr::get(symbolName, callee->getContext());
    rewriter.replaceOpWithNewOp<mlir::CallOp>(op, calleeSymbol, termTy,
                                              operands);
    return success();
  }
};

void populateBuiltinOpConversionPatterns(OwningRewritePatternList &patterns,
                                         MLIRContext *context,
                                         LLVMTypeConverter &converter,
                                         TargetInfo &targetInfo) {
  patterns.insert<IncrementReductionsOpConversion, IsTypeOpConversion,
                  PrintOpConversion, MallocOpConversion,
                  TraceCaptureOpConversion, TraceConstructOpConversion>(
      context, converter, targetInfo);
}

}  // namespace eir
}  // namespace lumen
