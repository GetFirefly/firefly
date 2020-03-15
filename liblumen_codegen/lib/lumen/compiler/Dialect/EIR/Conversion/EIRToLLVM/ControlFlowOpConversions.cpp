#include "lumen/compiler/Dialect/EIR/Conversion/EIRToLLVM/ControlFlowOpConversions.h"

namespace lumen {
namespace eir {
struct BranchOpConversion : public EIROpConversion<eir::BranchOp> {
  using EIROpConversion::EIROpConversion;

  PatternMatchResult matchAndRewrite(
      eir::BranchOp op, ArrayRef<Value> _properOperands,
      ArrayRef<Block *> destinations, ArrayRef<ArrayRef<Value>> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto dest = destinations.front();
    auto destArgs = operands.front();
    rewriter.replaceOpWithNewOp<mlir::BranchOp>(op, dest, destArgs);
    return matchSuccess();
  }
};

// Need to lower condition to i1
struct CondBranchOpConversion : public EIROpConversion<eir::CondBranchOp> {
  using EIROpConversion::EIROpConversion;

  PatternMatchResult matchAndRewrite(
      eir::CondBranchOp op, ArrayRef<Value> properOperands,
      ArrayRef<Block *> destinations, ArrayRef<ArrayRef<Value>> operands,
      ConversionPatternRewriter &rewriter) const override {
    CondBranchOpOperandAdaptor adaptor(properOperands);
    auto ctx = getRewriteContext(op, rewriter);

    auto cond = adaptor.condition();
    auto trueDest = op.getTrueDest();
    auto falseDest = op.getFalseDest();
    auto trueArgs = ValueRange(operands.front());
    auto falseArgs = ValueRange(operands.back());
    auto i1Ty = ctx.getI1Type();

    /*
    Value finalCond;
    if (cond.getType().cast<LLVMType>().isInteger(1)) {
      finalCond = cond;
    } else {
      // Need to unpack the boolean term
      finalCond = llvm_trunc(i1Ty, ctx.decodeImmediate(cond));
    }
    */

    auto attrs = op.getAttrs();
    ArrayRef<Block *> dests({trueDest, falseDest});
    ArrayRef<ValueRange> destsArgs({trueArgs, falseArgs});
    rewriter.replaceOpWithNewOp<LLVM::CondBrOp>(op, cond, dests, destsArgs,
                                                attrs);
    return matchSuccess();
  }
};

struct CallOpConversion : public EIROpConversion<CallOp> {
  using EIROpConversion::EIROpConversion;

  PatternMatchResult matchAndRewrite(
      CallOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    CallOpOperandAdaptor adaptor(operands);
    auto ctx = getRewriteContext(op, rewriter);

    SmallVector<LLVMType, 2> argTypes;
    for (auto operand : operands) {
      argTypes.push_back(operand.getType().cast<LLVMType>());
    }
    auto opResultTypes = op.getResultTypes();
    SmallVector<Type, 2> resultTypes;
    LLVMType resultType;
    if (opResultTypes.size() == 1) {
      resultType =
          ctx.typeConverter.convertType(opResultTypes.front()).cast<LLVMType>();
      assert(resultType && "unable to convert result type");
      resultTypes.push_back(resultType);
    } else {
      assert((opResultTypes.size() < 2) &&
             "expected call to have no more than 1 result");
    }

    // Always increment reduction count when performing a call
    rewriter.create<IncrementReductionsOp>(op.getLoc());

    auto calleeName = op.getCallee();
    auto callee = ctx.getOrInsertFunction(calleeName, resultType, argTypes);

    auto calleeSymbol =
        FlatSymbolRefAttr::get(calleeName, callee->getContext());
    auto callOp = rewriter.create<mlir::CallOp>(
        op.getLoc(), calleeSymbol, resultTypes, adaptor.operands());
    callOp.setAttr("tail", rewriter.getUnitAttr());
    rewriter.replaceOp(op, callOp.getResults());
    return matchSuccess();
  }
};

struct ReturnOpConversion : public EIROpConversion<ReturnOp> {
  using EIROpConversion::EIROpConversion;

  PatternMatchResult matchAndRewrite(
      ReturnOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::ReturnOp>(op, operands);
    return matchSuccess();
  }
};

struct ThrowOpConversion : public EIROpConversion<ThrowOp> {
  using EIROpConversion::EIROpConversion;

  PatternMatchResult matchAndRewrite(
      ThrowOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    ThrowOpOperandAdaptor adaptor(operands);
    auto ctx = getRewriteContext(op, rewriter);

    Value reason = adaptor.reason();
    auto termTy = ctx.getUsizeType();
    auto voidTy = LLVMType::getVoidTy(ctx.dialect);
    StringRef symbolName("llvm.trap");
    auto callee = ctx.getOrInsertFunction(symbolName, voidTy);

    auto calleeSymbol =
        FlatSymbolRefAttr::get(symbolName, callee->getContext());
    auto callOp = rewriter.create<mlir::CallOp>(op.getLoc(), calleeSymbol,
                                                ArrayRef<Type>{});
    callOp.setAttr("tail", rewriter.getUnitAttr());
    callOp.setAttr("noreturn", rewriter.getUnitAttr());
    callOp.setAttr("nounwind", rewriter.getUnitAttr());
    rewriter.replaceOpWithNewOp<LLVM::UnreachableOp>(op, ArrayRef<Value>{});
    return matchSuccess();
  }
};

struct UnreachableOpConversion : public EIROpConversion<UnreachableOp> {
  using EIROpConversion::EIROpConversion;

  PatternMatchResult matchAndRewrite(
      UnreachableOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::LLVM::UnreachableOp>(op, operands);
    return matchSuccess();
  }
};

struct YieldOpConversion : public EIROpConversion<YieldOp> {
  using EIROpConversion::EIROpConversion;

  PatternMatchResult matchAndRewrite(
      YieldOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = getRewriteContext(op, rewriter);

    auto voidTy = LLVMType::getVoidTy(ctx.dialect);
    StringRef symbolName("__lumen_builtin_yield");
    auto callee = ctx.getOrInsertFunction(symbolName, voidTy);

    auto calleeSymbol =
        FlatSymbolRefAttr::get(symbolName, callee->getContext());
    rewriter.replaceOpWithNewOp<mlir::CallOp>(op, calleeSymbol,
                                              ArrayRef<Type>{});
    return matchSuccess();
  }
};

struct YieldCheckOpConversion : public EIROpConversion<YieldCheckOp> {
  using EIROpConversion::EIROpConversion;

  PatternMatchResult matchAndRewrite(
      YieldCheckOp op, ArrayRef<Value> properOperands,
      ArrayRef<Block *> destinations, ArrayRef<ArrayRef<Value>> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = getRewriteContext(op, rewriter);
    ModuleOp mod = ctx.getModule();
    YieldCheckOpOperandAdaptor adaptor(properOperands);

    const uint64_t MAX_REDUCTION_COUNT = 20;

    auto termTy = ctx.getUsizeType();
    auto reductionCountGlobal = ctx.getOrInsertGlobal(
        "CURRENT_REDUCTION_COUNT", termTy, ctx.getIntegerAttr(0),
        LLVM::Linkage::External, LLVM::ThreadLocalMode::LocalExec);

    // Load the current reduction count
    Value reductionCount = llvm_load(reductionCountGlobal);
    // If greater than or equal to the max reduction count, yield
    Value maxReductions =
        llvm_constant(termTy, ctx.getIntegerAttr(MAX_REDUCTION_COUNT));
    Value shouldYield =
        llvm_icmp(LLVM::ICmpPredicate::uge, reductionCount, maxReductions);

    auto trueDest = op.getTrueDest();
    auto falseDest = op.getFalseDest();
    auto trueArgs = ValueRange(operands.front());
    auto falseArgs = ValueRange(operands.back());

    auto attrs = op.getAttrs();
    ArrayRef<Block *> dests({trueDest, falseDest});
    ArrayRef<ValueRange> destsArgs({trueArgs, falseArgs});
    rewriter.replaceOpWithNewOp<LLVM::CondBrOp>(op, shouldYield, dests,
                                                destsArgs, attrs);
    return matchSuccess();
  }
};

void populateControlFlowOpConversionPatterns(OwningRewritePatternList &patterns,
                                             MLIRContext *context,
                                             LLVMTypeConverter &converter,
                                             TargetInfo &targetInfo) {
  patterns
      .insert</*ApplyOpConversion,*/ BranchOpConversion, CondBranchOpConversion,
              /*CallIndirectOpConversion,*/ CallOpConversion,
              ReturnOpConversion, ThrowOpConversion, UnreachableOpConversion,
              YieldOpConversion, YieldCheckOpConversion>(context, converter,
                                                         targetInfo);
}

}  // namespace eir
}  // namespace lumen
