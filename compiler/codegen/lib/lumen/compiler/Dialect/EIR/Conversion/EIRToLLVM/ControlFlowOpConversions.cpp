#include "lumen/compiler/Dialect/EIR/Conversion/EIRToLLVM/ControlFlowOpConversions.h"

namespace lumen {
namespace eir {
struct BranchOpConversion : public EIROpConversion<eir::BranchOp> {
  using EIROpConversion::EIROpConversion;

  LogicalResult matchAndRewrite(
      eir::BranchOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    ArrayRef<Block *> successors(op.getSuccessor());
    rewriter.replaceOpWithNewOp<LLVM::BrOp>(op, operands, op.getSuccessor());
    return success();
  }
};

// Need to lower condition to i1
struct CondBranchOpConversion : public EIROpConversion<eir::CondBranchOp> {
  using EIROpConversion::EIROpConversion;

  LogicalResult matchAndRewrite(
      eir::CondBranchOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::CondBrOp>(op, operands, op.getSuccessors(),
                                          op.getAttrs());
    return success();
  }
};

struct CallOpConversion : public EIROpConversion<CallOp> {
  using EIROpConversion::EIROpConversion;

  LogicalResult matchAndRewrite(
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
    return success();
  }
};

struct CallClosureOpConversion : public EIROpConversion<CallClosureOp> {
  using EIROpConversion::EIROpConversion;

  LogicalResult matchAndRewrite(
      CallClosureOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    CallClosureOpOperandAdaptor adaptor(operands);
    auto ctx = getRewriteContext(op, rewriter);

    auto opaqueFnTy = ctx.targetInfo.getOpaqueFnType();
    auto closureTy = ctx.targetInfo.makeClosureType(ctx.dialect, 1);
    auto termTy = ctx.getUsizeType();
    auto i32Ty = ctx.getI32Type();

    // Always increment reduction count when performing a call
    rewriter.create<IncrementReductionsOp>(op.getLoc());

    // Extract closure pointer from box
    auto boxedClosure = adaptor.callee();
    auto closure = ctx.decodeBox(closureTy, boxedClosure);

    // Extract code/function pointer from closure header
    Value zero = llvm_constant(i32Ty, ctx.getI32Attr(0));
    Value codeIdx = llvm_constant(i32Ty, ctx.getI32Attr(4));
    ArrayRef<Value> codePtrIndices({zero, codeIdx});
    LLVMType opaqueFnTyPtr = opaqueFnTy.getPointerTo();
    Value codePtr =
        llvm_gep(opaqueFnTyPtr.getPointerTo(), closure, codePtrIndices);

    // Build call op operands
    SmallVector<LLVMType, 2> argTypes;
    argTypes.push_back(closureTy.getPointerTo());
    for (auto operand : adaptor.operands()) {
      argTypes.push_back(operand.getType().cast<LLVMType>());
    }
    auto fnTy = LLVMType::getFunctionTy(termTy, argTypes, false);
    Value fnPtr = llvm_bitcast(fnTy.getPointerTo(), llvm_load(codePtr));
    SmallVector<Value, 2> args;
    args.push_back(fnPtr);
    args.push_back(closure);
    for (auto operand : adaptor.operands()) {
      args.push_back(operand);
    }
    auto callOp =
        rewriter.create<LLVM::CallOp>(op.getLoc(), termTy, args, op.getAttrs());
    callOp.setAttr("tail", rewriter.getUnitAttr());
    Value result = callOp.getResult(0);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ReturnOpConversion : public EIROpConversion<ReturnOp> {
  using EIROpConversion::EIROpConversion;

  LogicalResult matchAndRewrite(
      ReturnOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::ReturnOp>(op, operands);
    return success();
  }
};

struct ThrowOpConversion : public EIROpConversion<ThrowOp> {
  using EIROpConversion::EIROpConversion;

  LogicalResult matchAndRewrite(
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
    rewriter.replaceOpWithNewOp<LLVM::UnreachableOp>(op);
    return success();
  }
};

struct UnreachableOpConversion : public EIROpConversion<UnreachableOp> {
  using EIROpConversion::EIROpConversion;

  LogicalResult matchAndRewrite(
      UnreachableOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::LLVM::UnreachableOp>(op);
    return success();
  }
};

struct YieldOpConversion : public EIROpConversion<YieldOp> {
  using EIROpConversion::EIROpConversion;

  LogicalResult matchAndRewrite(
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
    return success();
  }
};

struct YieldCheckOpConversion : public EIROpConversion<YieldCheckOp> {
  using EIROpConversion::EIROpConversion;

  LogicalResult matchAndRewrite(
      YieldCheckOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = getRewriteContext(op, rewriter);
    ModuleOp mod = ctx.getModule();

    auto i32Ty = ctx.getI32Type();
    auto reductionCountGlobal = ctx.getOrInsertGlobal(
        "CURRENT_REDUCTION_COUNT", i32Ty, nullptr,
        LLVM::Linkage::External, LLVM::ThreadLocalMode::LocalExec);

    // Load the current reduction count
    Value reductionCount = llvm_load(reductionCountGlobal);
    // If greater than or equal to the max reduction count, yield
    Value maxReductions = op.getMaxReductions();
    Value shouldYield =
        llvm_icmp(LLVM::ICmpPredicate::uge, reductionCount, maxReductions);

    rewriter.replaceOpWithNewOp<LLVM::CondBrOp>(op, shouldYield, op.getSuccessors(),
                                          op.getAttrs());
    return success();
  }
};

void populateControlFlowOpConversionPatterns(OwningRewritePatternList &patterns,
                                             MLIRContext *context,
                                             LLVMTypeConverter &converter,
                                             TargetInfo &targetInfo) {
  patterns.insert<
      /*ApplyOpConversion,*/ BranchOpConversion, CondBranchOpConversion,
      /*CallIndirectOpConversion,*/ CallOpConversion, CallClosureOpConversion,
      ReturnOpConversion, ThrowOpConversion, UnreachableOpConversion,
      YieldOpConversion, YieldCheckOpConversion>(context, converter,
                                                 targetInfo);
}

}  // namespace eir
}  // namespace lumen
