#include "lumen/EIR/Conversion/ControlFlowOpConversions.h"

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
    auto ctx = getRewriteContext(op, rewriter);
    CondBranchOpAdaptor adaptor(op);

    Value cond = adaptor.condition();
    Type condType = cond.getType();
    if (auto condTy = condType.dyn_cast_or_null<LLVMIntegerType>()) {
      if (condTy.getBitWidth() > 1) {
        auto i1Ty = ctx.getI1Type();
        Value newCond = rewriter.create<CastOp>(op.getLoc(), adaptor.condition(), i1Ty);
        rewriter.replaceOpWithNewOp<LLVM::CondBrOp>(
          op, newCond, op.trueDest(), adaptor.trueDestOperands(), op.falseDest(), op.falseDestOperands()
        );
        return success();
      }
    }

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
    CallOpAdaptor adaptor(operands);
    auto ctx = getRewriteContext(op, rewriter);
    auto termTy = ctx.getUsizeType();

    // Always increment reduction count when performing a call
    rewriter.create<IncrementReductionsOp>(op.getLoc());

    // Default argument types for function have to assume terms for all
    // arguments. In the future we'll have a pass that creates functions
    // before lowering, and gathers type info from callers so that we can
    // use more precise types, but we're not there yet
    SmallVector<LLVMType, 2> argTypes;
    for (auto operand : adaptor.operands()) {
      argTypes.push_back(termTy);
    }

    // Results need to be converted, or use void if no result is returned
    auto opResultTypes = op.getResultTypes();
    LLVMType resultType;
    SmallVector<Type, 1> resultTypes;
    if (opResultTypes.size() == 1) {
      resultType =
          ctx.typeConverter.convertType(opResultTypes.front()).cast<LLVMType>();
      assert(resultType && "unable to convert result type");
      resultTypes.push_back(resultType);
    } else {
      resultType = ctx.targetInfo.getVoidType();
    }

    auto callee = ctx.getOrInsertFunction(op.callee(), resultType, argTypes);
    LLVM::LLVMFunctionType calleeTy;
    if (auto llvmFunc = dyn_cast_or_null<LLVM::LLVMFuncOp>(callee)) {
      calleeTy = llvmFunc.getType().cast<LLVM::LLVMFunctionType>();
    } else {
      auto eirFunc = cast<FuncOp>(callee);
      calleeTy = ctx.typeConverter.convertType(eirFunc.getType()).cast<LLVM::LLVMFunctionType>();
    }

    SmallVector<Value, 2> args;
    unsigned index = 0;
    for (auto operand : adaptor.operands()) {
      LLVMType paramTy = calleeTy.getParamType(index);
      auto argTy = operand.getType();
      LLVMType llvmArgTy = ctx.typeConverter.convertType(argTy).cast<LLVMType>();

      // If the types match, we're good
      if (paramTy == llvmArgTy) {
        args.push_back(operand);
        continue;
      }

      // If the parameter type is a term type, the argument must be cast
      if (paramTy.isIntegerTy(ctx.targetInfo.pointerSizeInBits)) {
        Value cast = eir_cast(operand, rewriter.getType<TermType>());
        args.push_back(cast);
        continue;
      }

      // If the argument type is pointer-like, and the parameter type is a pointer,
      // get the pointee type and see if we can cast to an appropriate concrete
      // type
      if (paramTy.isPointerTy()) {
        LLVMType elTy = paramTy.cast<LLVM::LLVMPointerType>().getElementType();
        // Terms can be cast to pointer types when the pointee is a term structure
        if (argTy.isa<TermType>() && elTy.isStructTy()) {
          auto structTy = elTy.cast<LLVM::LLVMStructType>();
          auto structName = structTy.getName();
          // Obtain a type builder for the type to cast to based on the pointee type name
          auto castToFun =
            StringSwitch<BuildCastFnT>(structName)
            .Case("binary", [](OpBuilder &b) { return b.getType<BoxType>(b.getType<BinaryType>()); })
            .Case("cons", [](OpBuilder &b) { return b.getType<BoxType>(b.getType<ConsType>()); })
            .Case("bigint", [](OpBuilder &b) { return b.getType<BoxType>(b.getType<BigIntType>()); })
            .Case("float", [](OpBuilder &b) { return b.getType<BoxType>(b.getType<FloatType>()); })
            .Default([&](OpBuilder &b) {
              if (structName.startswith("closure")) {
                return Optional((Type)b.getType<BoxType>(b.getType<ClosureType>()));
              }
              if (structName.startswith("tuple")) {
                return Optional((Type)b.getType<BoxType>(b.getType<TupleType>()));
              }
              return (Optional<Type>)llvm::None;
            });
          // If we found a type to build, construct a cast to the boxed type first, then
          // cast to raw pointer type. In the case of types with dynamic extent, an additional
          // bitcast may be necessary
          Optional<Type> castTo = castToFun(rewriter);
          if (castTo.hasValue()) {
            BoxType boxedTy = castTo.getValue().cast<BoxType>();
            OpaqueTermType innerTy = boxedTy.getBoxedType();
            Value cast = eir_cast(operand, boxedTy);
            Value unboxed = eir_cast(cast, rewriter.getType<PtrType>(innerTy));
            if (innerTy.isClosure() || innerTy.isTuple()) {
              Value fixed = llvm_bitcast(paramTy, unboxed);
              args.push_back(fixed);
            } else {
              args.push_back(unboxed);
            }
          }

          args.push_back(operand);
          continue;
        }

        // If a boxed term is passed to a parameter of raw pointer type, we need to
        // cast the box to raw pointer form so that unboxing occurs
        if (auto boxedTy = argTy.dyn_cast_or_null<BoxType>()) {
          if (elTy.isStructTy()) {
            auto structTy = elTy.cast<LLVM::LLVMStructType>();
            auto structName = structTy.getName();
            auto isCastable =
              StringSwitch<bool>(structName)
              .Case("binary", true)
              .Case("cons", true)
              .Case("bigint", true)
              .Case("float", true)
              .Default(structName.startswith("closure") || structName.startswith("tuple"));
            if (isCastable) {
              Value cast = eir_cast(operand, rewriter.getType<PtrType>(boxedTy.getBoxedType()));
              args.push_back(cast);
              continue;
            }
          }
        }

        args.push_back(operand);
        continue;
      }
    }

    // Add tail call markers where present
    Operation *callOp = llvm_call(resultTypes, op.getCalleeAttr(), args);
    auto attrs = op.getAttrs();
    for (auto attr : op.getAttrs()) {
      callOp->setAttr(std::get<Identifier>(attr), std::get<Attribute>(attr));
    }

    rewriter.replaceOp(op, callOp->getResults());
    return success();
  }
};

struct InvokeOpConversion : public EIROpConversion<InvokeOp> {
  using EIROpConversion::EIROpConversion;

  LogicalResult matchAndRewrite(
      InvokeOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    InvokeOpAdaptor adaptor(op);
    auto ctx = getRewriteContext(op, rewriter);
    auto termTy = ctx.getUsizeType();

    // Always increment reduction count when performing a call
    rewriter.create<IncrementReductionsOp>(op.getLoc());

    // Just like CallOp, we have to assume term type for any functions
    // that don't have type information already defined
    SmallVector<LLVMType, 2> argTypes;
    for (auto operand : adaptor.operands()) {
      argTypes.push_back(termTy);
    }

    // The result types are based on the block arguments of the normal block
    auto ok = op.okDest();
    auto okBlockArgs = ok->getArguments();
    LLVMType resultType;
    if (!ok->args_empty()) {
      resultType =
          ctx.typeConverter.convertType(*ok->getArgumentTypes().begin())
              .cast<LLVMType>();
    }

    auto callee = ctx.getOrInsertFunction(op.callee(), resultType, argTypes);
    LLVM::LLVMFunctionType calleeTy;
    if (auto llvmFunc = dyn_cast_or_null<LLVM::LLVMFuncOp>(callee)) {
      calleeTy = llvmFunc.getType().cast<LLVM::LLVMFunctionType>();
    } else {
      auto eirFunc = cast<FuncOp>(callee);
      calleeTy = ctx.typeConverter.convertType(eirFunc.getType()).cast<LLVM::LLVMFunctionType>();
    }

    SmallVector<Value, 2> args;
    unsigned index = 0;
    for (auto operand : adaptor.operands()) {
      auto paramTy = calleeTy.getParamType(index);
      auto argTy = argTypes[index];
      // If the types match, we're good
      if (paramTy == argTy) {
        args.push_back(operand);
        continue;
      }
      // If we have a concrete pointer type being passed
      // as an opaque term parameter, insert a bitcast
      if (argTy.isPointerTy() && paramTy == termTy) {
        Value cast = llvm_bitcast(paramTy, operand);
        args.push_back(cast);
        continue;
      }
      // Otherwise we have an unresolvable discrepancy,
      // if we don't catch this here, LLVM will blow up
      // when lowering to LLVM IR, so raise an error here
      op.emitOpError("unexpected argument type mismatch");
      return failure();
    }

    ValueRange okArgs = op.okDestOperands();
    auto err = op.errDest();
    ValueRange errArgs = op.errDestOperands();
    Operation *callOp = llvm_invoke(ArrayRef<Type>{}, op.getCalleeAttr(), args, ok, okArgs, err, errArgs);
    for (auto attr : op.getAttrs()) {
      callOp->setAttr(std::get<Identifier>(attr), std::get<Attribute>(attr));
    }

    rewriter.eraseOp(op);
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
    Block *canHandleBlock = rewriter.splitBlock(
        landingPadBlock, Block::iterator(op.getOperation()));

    // { i8*, i32 }
    //
    // i8* - pointer to exception object being thrown
    // i32 - a "selector" indicating which landing pad clause the
    //       exception type matched. Like Rust, we ignore this
    auto i8PtrTy = ctx.targetInfo.getI8Type().getPointerTo();
    auto i8PtrPtrTy = i8PtrTy.getPointerTo();
    auto i64Ty = ctx.getI64Type();
    auto i32Ty = ctx.getI32Type();
    auto termTy = ctx.getUsizeType();
    auto termPtrTy = termTy.getPointerTo();
    auto termPtrPtrTy = termPtrTy.getPointerTo();
    auto exceptionTy = ctx.targetInfo.getExceptionType();
    auto voidTy = LLVMType::getVoidTy(ctx.context);
    auto tupleTy = ctx.getTupleType({termTy, termTy, termPtrTy});
    auto erlangErrorTy = ctx.targetInfo.getErlangErrorType();
    auto erlangErrorPtrTy = erlangErrorTy.getPointerTo();

    // Make sure we're starting in the landing pad block
    rewriter.setInsertionPointToEnd(landingPadBlock);

    auto catchClauses = op.catchClauses();
    assert(catchClauses.size() == 1 && "expected only a single catch clause");

    // The landing pad returns the structure defined above
    Value obj = llvm_landingpad(exceptionTy, /*cleanup=*/false, catchClauses);
    // Extract the exception object (a pointer to the raw exception object)
    Value exPtr = llvm_extractvalue(i8PtrTy, obj, ctx.getI64ArrayAttr(0));

    auto canHandleBrOp =
        rewriter.create<LLVM::BrOp>(loc, ValueRange{}, canHandleBlock);

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
        symbolName, erlangErrorPtrTy, ArrayRef<LLVMType>{i8PtrTy}, calleeAttrs);
    auto callOp = rewriter.create<mlir::CallOp>(
        op.getLoc(), rewriter.getSymbolRefAttr(symbolName),
        ArrayRef<Type>{erlangErrorPtrTy}, ArrayRef<Value>{exPtr});
    callOp.setAttr("tail", rewriter.getUnitAttr());
    Value erlangErrorPtr = callOp.getResult(0);
    // Extract exception values
    Value zero = llvm_constant(i32Ty, ctx.getI32Attr(0));
    auto kindIdx = llvm_constant(i32Ty, ctx.getI32Attr(1));
    Value kindPtr =
        llvm_gep(termPtrTy, erlangErrorPtr, ArrayRef<Value>{zero, kindIdx});
    Value kind = llvm_load(kindPtr);
    auto reasonIdx = llvm_constant(i32Ty, ctx.getI32Attr(2));
    auto reasonPtr =
        llvm_gep(termPtrTy, erlangErrorPtr, ArrayRef<Value>{zero, reasonIdx});
    Value reason = llvm_load(reasonPtr);
    auto traceIdx = llvm_constant(i32Ty, ctx.getI32Attr(3));
    auto tracePtr = llvm_gep(termPtrPtrTy, erlangErrorPtr,
                             ArrayRef<Value>{zero, traceIdx});
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
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, operands);
    return success();
  }
};

struct ThrowOpConversion : public EIROpConversion<ThrowOp> {
  using EIROpConversion::EIROpConversion;

  LogicalResult matchAndRewrite(
      ThrowOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    ThrowOpAdaptor adaptor(operands);
    auto ctx = getRewriteContext(op, rewriter);
    auto loc = op.getLoc();
    auto termTy = ctx.getUsizeType();
    auto termPtrTy = termTy.getPointerTo();
    auto i8PtrTy = ctx.targetInfo.getI8Type().getPointerTo();
    auto i32Ty = ctx.getI32Type();
    auto voidTy = LLVMType::getVoidTy(ctx.context);
    auto erlangErrorTy = ctx.targetInfo.getErlangErrorType();
    auto erlangErrorPtrTy = erlangErrorTy.getPointerTo();

    Value kind = adaptor.kind();
    Value reason = adaptor.reason();
    Value trace = adaptor.trace();

    // Construct exception
    const char *raiseSymbol = "__lumen_builtin_raise/3";
    ArrayRef<NamedAttribute> raiseAttrs = {
      rewriter.getNamedAttr("nounwind", rewriter.getUnitAttr()),
    };
    auto raiseCallee = ctx.getOrInsertFunction(raiseSymbol, erlangErrorPtrTy, ArrayRef<LLVMType>{termTy, termTy, termPtrTy}, raiseAttrs);
    auto raiseOp = rewriter.create<mlir::CallOp>(op.getLoc(), rewriter.getSymbolRefAttr(raiseSymbol), ArrayRef<Type>{erlangErrorPtrTy}, ArrayRef<Value>{kind, reason, trace});
    raiseOp.setAttr("tail", rewriter.getUnitAttr());
    raiseOp.setAttr("nounwind", rewriter.getUnitAttr());

    Value exception = raiseOp.getResult(0);

    // Raise panic
    const char *symbolName = "__lumen_start_panic";
    ArrayRef<NamedAttribute> calleeAttrs = {
        rewriter.getNamedAttr("noreturn", rewriter.getUnitAttr()),
    };
    auto callee = ctx.getOrInsertFunction(
        symbolName, voidTy, ArrayRef<LLVMType>{erlangErrorPtrTy}, calleeAttrs);

    auto callOp = rewriter.create<mlir::CallOp>(
        op.getLoc(), rewriter.getSymbolRefAttr(symbolName), ArrayRef<Type>{},
        ArrayRef<Value>{exception});
    callOp.setAttr("tail", rewriter.getUnitAttr());
    callOp.setAttr("noreturn", rewriter.getUnitAttr());
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

    auto voidTy = LLVMType::getVoidTy(ctx.context);
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

struct ReceiveStartOpConversion : public EIROpConversion<ReceiveStartOp> {
  using EIROpConversion::EIROpConversion;

  LogicalResult matchAndRewrite(
      ReceiveStartOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = getRewriteContext(op, rewriter);
    ReceiveStartOpAdaptor adaptor(operands);

    auto termTy = ctx.getUsizeType();
    auto recvRefTy = ctx.targetInfo.getReceiveRefType();

    StringRef symbolName("__lumen_builtin_receive_start");
    auto callee = ctx.getOrInsertFunction(symbolName, recvRefTy, {termTy});
    auto calleeSymbol =
        FlatSymbolRefAttr::get(symbolName, callee->getContext());

    Value timeout = adaptor.timeout();
    auto startOp = rewriter.create<mlir::CallOp>(
        op.getLoc(), calleeSymbol, recvRefTy, ArrayRef<Value>{timeout});

    rewriter.replaceOp(op, {startOp.getResult(0)});
    return success();
  }
};

struct ReceiveWaitOpConversion : public EIROpConversion<ReceiveWaitOp> {
  using EIROpConversion::EIROpConversion;

  LogicalResult matchAndRewrite(
      ReceiveWaitOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = getRewriteContext(op, rewriter);

    auto i8Ty = ctx.getI8Type();
    auto recvRefTy = ctx.targetInfo.getReceiveRefType();

    StringRef symbolName("__lumen_builtin_receive_wait");
    auto callee = ctx.getOrInsertFunction(symbolName, i8Ty, {recvRefTy});
    auto calleeSymbol =
        FlatSymbolRefAttr::get(symbolName, callee->getContext());

    ArrayRef<Value> args{op.recvRef()};
    rewriter.replaceOpWithNewOp<mlir::CallOp>(op, calleeSymbol, i8Ty, args);
    return success();
  }
};

struct ReceiveMessageOpConversion : public EIROpConversion<ReceiveMessageOp> {
  using EIROpConversion::EIROpConversion;

  LogicalResult matchAndRewrite(
      ReceiveMessageOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = getRewriteContext(op, rewriter);

    auto termTy = ctx.getUsizeType();
    auto recvRefTy = ctx.targetInfo.getReceiveRefType();

    StringRef symbolName("__lumen_builtin_receive_message");
    auto callee = ctx.getOrInsertFunction(symbolName, termTy, {recvRefTy});
    auto calleeSymbol =
        FlatSymbolRefAttr::get(symbolName, callee->getContext());

    ArrayRef<Value> args{op.recvRef()};
    rewriter.replaceOpWithNewOp<mlir::CallOp>(op, calleeSymbol, termTy, args);
    return success();
  }
};

struct ReceiveDoneOpConversion : public EIROpConversion<ReceiveDoneOp> {
  using EIROpConversion::EIROpConversion;

  LogicalResult matchAndRewrite(
      ReceiveDoneOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = getRewriteContext(op, rewriter);
    ReceiveDoneOpAdaptor adaptor(operands);

    auto recvRefTy = ctx.targetInfo.getReceiveRefType();
    auto voidTy = LLVMType::getVoidTy(ctx.context);

    StringRef symbolName("__lumen_builtin_receive_done");

    auto callee = ctx.getOrInsertFunction(symbolName, voidTy, {recvRefTy});
    auto calleeSymbol =
        FlatSymbolRefAttr::get(symbolName, callee->getContext());

    Value recvRef = adaptor.recvRef();
    rewriter.replaceOpWithNewOp<mlir::CallOp>(
        op, calleeSymbol, ArrayRef<Type>{}, ArrayRef<Value>{recvRef});
    return success();
  }
};

void populateControlFlowOpConversionPatterns(OwningRewritePatternList &patterns,
                                             MLIRContext *context,
                                             EirTypeConverter &converter,
                                             TargetInfo &targetInfo) {
  patterns.insert<
      /*ApplyOpConversion,*/
      BranchOpConversion, CondBranchOpConversion,
      /*CallIndirectOpConversion,*/ CallOpConversion,
      InvokeOpConversion, LandingPadOpConversion,
      ReturnOpConversion, ThrowOpConversion, UnreachableOpConversion,
      YieldOpConversion, YieldCheckOpConversion, ReceiveStartOpConversion,
      ReceiveWaitOpConversion, ReceiveMessageOpConversion,
      ReceiveDoneOpConversion>(context, converter, targetInfo);
}

}  // namespace eir
}  // namespace lumen
