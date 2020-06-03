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
    rewriter.replaceOpWithNewOp<LLVM::CondBrOp>(
        op, operands, op.getSuccessors(), op.getAttrs());
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

struct InvokeOpConversion : public EIROpConversion<InvokeOp> {
  using EIROpConversion::EIROpConversion;

  LogicalResult matchAndRewrite(
      InvokeOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = getRewriteContext(op, rewriter);

    SmallVector<LLVMType, 2> argTypes;
    for (auto operand : op.operands()) {
      auto argType =
          ctx.typeConverter.convertType(operand.getType()).cast<LLVMType>();
      argTypes.push_back(argType);
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

    auto attrs = op.getAttrs();
    auto args = op.operands();
    auto ok = op.okDest();
    ValueRange okArgs = op.okDestOperands();
    auto err = op.errDest();
    ValueRange errArgs = op.errDestOperands();
    auto calleeSymbol =
        FlatSymbolRefAttr::get(calleeName, callee->getContext());
    auto callOp = rewriter.create<LLVM::InvokeOp>(
        op.getLoc(), resultTypes, calleeSymbol, args, ok, okArgs, err, errArgs);
    for (auto attr : attrs) {
      callOp.setAttr(std::get<Identifier>(attr), std::get<Attribute>(attr));
    }

    rewriter.replaceOp(op, callOp.getResults());
    return success();
  }
};

struct InvokeClosureOpConversion : public EIROpConversion<InvokeClosureOp> {
  using EIROpConversion::EIROpConversion;

  LogicalResult matchAndRewrite(
      InvokeClosureOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = getRewriteContext(op, rewriter);

    auto opaqueFnTy = ctx.targetInfo.getOpaqueFnType();
    auto closureTy = ctx.targetInfo.makeClosureType(ctx.dialect, 1);
    auto termTy = ctx.getUsizeType();
    auto i32Ty = ctx.getI32Type();

    // Always increment reduction count when performing a call
    rewriter.create<IncrementReductionsOp>(op.getLoc());

    // Extract closure pointer from box
    auto boxedClosure = op.callee();
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
    for (auto operand : op.operands()) {
      auto argType =
          ctx.typeConverter.convertType(operand.getType()).cast<LLVMType>();
      argTypes.push_back(argType);
    }
    auto fnTy = LLVMType::getFunctionTy(termTy, argTypes, false);
    Value fnPtr = llvm_bitcast(fnTy.getPointerTo(), llvm_load(codePtr));
    SmallVector<Type, 1> resultTypes({termTy});
    SmallVector<Value, 2> args;
    args.push_back(fnPtr);
    args.push_back(closure);
    for (auto operand : op.operands()) {
      args.push_back(operand);
    }
    auto attrs = op.getAttrs();
    auto ok = op.okDest();
    auto okArgs = op.okDestOperands();
    auto err = op.errDest();
    auto errArgs = op.errDestOperands();
    auto callOp = rewriter.create<LLVM::InvokeOp>(
        op.getLoc(), resultTypes, args, ok, okArgs, err, errArgs);
    // HACK(pauls): Same deal as InvokeOp, see note there
    auto numResults = callOp.getNumResults();
    assert(numResults < 2 &&
           "support for multi-value returns is not implemented");
    if (numResults == 1 && ok->getNumArguments() > 0) {
      auto result = callOp.getResult(0);
      auto arg = ok->getArgument(0);
      rewriter.replaceUsesOfBlockArgument(arg, result);
      ok->eraseArgument(0);
      rewriter.replaceOp(op, result);
    } else {
      rewriter.replaceOp(op, callOp.getResults());
    }
    return success();
  }
};

struct LandingPadOpConversion : public EIROpConversion<LandingPadOp> {
  using EIROpConversion::EIROpConversion;

  LogicalResult matchAndRewrite(
      LandingPadOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = getRewriteContext(op, rewriter);
    auto loc = op.getLoc();

    Block *landingPadBlock = rewriter.getBlock();
    Block *canHandleBlock = rewriter.splitBlock(landingPadBlock, Block::iterator(op.getOperation()));
    Block *resumeBlock = rewriter.createBlock(landingPadBlock->getParent());

    // { i8*, i32 }
    //
    // i8* - pointer to exception object being thrown
    // i32 - a "selector" indicating which landing pad clause the
    //       exception type matched. Like Rust, we ignore this
    auto i8PtrTy = ctx.targetInfo.getI8Type().getPointerTo();
    auto i8PtrPtrTy = i8PtrTy.getPointerTo();
    auto i64Ty = LLVMType::getInt64Ty(ctx.dialect);
    auto i32Ty = ctx.getI32Type();
    auto termTy = ctx.getUsizeType();
    auto termPtrTy = termTy.getPointerTo();
    auto tupleTy = ctx.getTupleType(3);
    auto exceptionTy = ctx.targetInfo.getExceptionType();

    // Make sure we're starting in the landing pad block
    rewriter.setInsertionPointToEnd(landingPadBlock);

    // The landing pad returns the structure defined above
    Value obj = llvm_landingpad(exceptionTy, /*cleanup=*/false, op.catchClauses());
    // Extract the exception object (a pointer to the raw exception object)
    Value exPtr = llvm_extractvalue(i8PtrTy, obj, ctx.getI64ArrayAttr(0));
    // Extract the exception selector (index of the clause that matched)
    Value exSelector = llvm_extractvalue(i32Ty, obj, ctx.getI64ArrayAttr(1));

    Value erlangErrorSelector = llvm_constant(i32Ty, ctx.getI32Attr(1));
    Value canHandle =
        llvm_icmp(LLVM::ICmpPredicate::eq, exSelector, erlangErrorSelector);

    auto canHandleBrOp = rewriter.create<LLVM::CondBrOp>(loc, canHandle, canHandleBlock, resumeBlock);

    // If we can't handle this block, continue unwinding
    rewriter.setInsertionPointToEnd(resumeBlock);
    rewriter.create<LLVM::ResumeOp>(loc, obj);

    // If we can, then extract the error value from the exception
    rewriter.setInsertionPointToStart(canHandleBlock);

    // Extract the Erlang exception object (a boxed 3-tuple)
    const char *symbolName = "__lumen_get_exception";
    ArrayRef<NamedAttribute> calleeAttrs = {
        rewriter.getNamedAttr("nounwind", rewriter.getUnitAttr()),
        rewriter.getNamedAttr("readonly", rewriter.getUnitAttr()),
        rewriter.getNamedAttr("argmemonly", rewriter.getUnitAttr()),
    };
    auto callee = ctx.getOrInsertFunction(
        symbolName, termTy, ArrayRef<LLVMType>{i8PtrTy}, calleeAttrs);
    auto callOp = rewriter.create<mlir::CallOp>(
        op.getLoc(), rewriter.getSymbolRefAttr(symbolName),
        ArrayRef<Type>{termTy}, ArrayRef<Value>{exPtr});
    callOp.setAttr("tail", rewriter.getUnitAttr());
    // Cast to tuple
    Value tuplePtr = ctx.decodeBox(tupleTy, callOp.getResult(0));
    // Extract exception values
    Value zero = llvm_constant(i32Ty, ctx.getI32Attr(0));
    auto kindIdx = llvm_constant(i32Ty, ctx.getI32Attr(1));
    Value kindPtr =
        llvm_gep(termPtrTy, tuplePtr, ArrayRef<Value>{zero, kindIdx});
    Value kind = llvm_load(kindPtr);
    auto reasonIdx = llvm_constant(i32Ty, ctx.getI32Attr(2));
    auto reasonPtr =
        llvm_gep(termPtrTy, tuplePtr, ArrayRef<Value>{zero, reasonIdx});
    Value reason = llvm_load(reasonPtr);
    auto traceIdx = llvm_constant(i32Ty, ctx.getI32Attr(3));
    auto tracePtr =
        llvm_gep(termPtrTy, tuplePtr, ArrayRef<Value>{zero, traceIdx});
    Value trace = llvm_load(tracePtr);

    rewriter.replaceOp(op, {kind, reason, trace});
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
    auto loc = op.getLoc();
    auto termTy = ctx.getUsizeType();
    auto termPtrTy = termTy.getPointerTo();
    auto i32Ty = ctx.getI32Type();

    Value kind = adaptor.kind();
    Value reason = adaptor.reason();
    Value trace = adaptor.trace();

    // Allocate tuple and write values to it
    ArrayRef<Value> elements{kind, reason, trace};
    auto numElements = elements.size();
    auto tupleTy = ctx.getTupleType(numElements);

    Value arity = llvm_constant(termTy, ctx.getIntegerAttr(numElements));
    Value tuplePtr = ctx.buildMalloc(tupleTy, TypeKind::Tuple, arity);

    Value zero = llvm_constant(i32Ty, ctx.getI32Attr(0));
    auto headerRaw = ctx.targetInfo.encodeHeader(TypeKind::Tuple, numElements);
    ArrayRef<Value> headerTermIndices{zero, zero};
    Value headerTermPtr = llvm_gep(termPtrTy, tuplePtr, headerTermIndices);
    llvm_store(llvm_constant(termTy, ctx.getIntegerAttr(headerRaw)),
               headerTermPtr);

    for (auto i = 0; i < numElements; i++) {
      auto element = elements[i];
      auto elementTy = tupleTy.getStructElementType(i + 1).getPointerTo();
      auto idx = llvm_constant(i32Ty, ctx.getI32Attr(i + 1));
      ArrayRef<Value> elementIndices{zero, idx};
      auto elementPtr = llvm_gep(elementTy, tuplePtr, elementIndices);
      llvm_store(element, elementPtr);
    }

    // Box the allocated tuple and bitcast to opaque term
    auto boxed = ctx.encodeBox(tuplePtr);
    Value exception = llvm_bitcast(termTy, boxed);

    auto voidTy = LLVMType::getVoidTy(ctx.dialect);
    const char *symbolName = "__lumen_start_panic";
    ArrayRef<NamedAttribute> calleeAttrs = {
        rewriter.getNamedAttr("noreturn", rewriter.getUnitAttr()),
    };
    auto callee = ctx.getOrInsertFunction(
        symbolName, voidTy, ArrayRef<LLVMType>{termTy}, calleeAttrs);

    auto callOp = rewriter.create<mlir::CallOp>(
        op.getLoc(), rewriter.getSymbolRefAttr(symbolName), ArrayRef<Type>{},
        ArrayRef<Value>{exception});
    callOp.setAttr("tail", rewriter.getUnitAttr());
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
    const char *symbolName = "__lumen_builtin_yield";
    auto callee = ctx.getOrInsertFunction(symbolName, voidTy, {});

    rewriter.replaceOpWithNewOp<mlir::CallOp>(
        op, rewriter.getSymbolRefAttr(symbolName), ArrayRef<Type>{});
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
        "CURRENT_REDUCTION_COUNT", i32Ty, nullptr, LLVM::Linkage::External,
        LLVM::ThreadLocalMode::LocalExec);

    // Load the current reduction count
    Value reductionCount = llvm_load(reductionCountGlobal);
    // If greater than or equal to the max reduction count, yield
    Value maxReductions = op.getMaxReductions();
    Value shouldYield =
        llvm_icmp(LLVM::ICmpPredicate::uge, reductionCount, maxReductions);

    rewriter.replaceOpWithNewOp<LLVM::CondBrOp>(
        op, shouldYield, op.getSuccessors(), op.getAttrs());
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
      InvokeOpConversion, InvokeClosureOpConversion, LandingPadOpConversion,
      ReturnOpConversion, ThrowOpConversion, UnreachableOpConversion,
      YieldOpConversion, YieldCheckOpConversion>(context, converter,
                                                 targetInfo);
}

}  // namespace eir
}  // namespace lumen
