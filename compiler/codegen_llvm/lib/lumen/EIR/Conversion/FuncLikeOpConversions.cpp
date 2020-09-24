#include "lumen/EIR/Conversion/FuncLikeOpConversions.h"

namespace lumen {
namespace eir {

const unsigned CLOSURE_ENV_INDEX = 5;

// The purpose of this conversion is to build a function that contains
// all of the prologue setup our Erlang functions need (in cases where
// this isn't a declaration). Specifically:
//
// - Check if reduction count is exceeded
// - Check if we should garbage collect
//   - If either of the above are true, yield
//
// TODO: Need to actually perform the above, right now we just handle
// the translation to mlir::FuncOp
struct FuncOpConversion : public EIROpConversion<eir::FuncOp> {
  using EIROpConversion::EIROpConversion;

  LogicalResult matchAndRewrite(
      eir::FuncOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = getRewriteContext(op, rewriter);

    auto i32Ty = ctx.getI32Type();

    SmallVector<NamedAttribute, 2> attrs;
    for (auto fa : op.getAttrs()) {
      if (fa.first == SymbolTable::getSymbolAttrName() ||
          fa.first == ::mlir::impl::getTypeAttrName()) {
        continue;
      }
      attrs.push_back(fa);
    }
    SmallVector<MutableDictionaryAttr, 4> argAttrs;
    for (unsigned i = 0, e = op.getNumArguments(); i < e; ++i) {
      auto aa = ::mlir::impl::getArgAttrs(op, i);
      argAttrs.push_back(MutableDictionaryAttr(aa));
    }
    auto newFunc = rewriter.create<mlir::FuncOp>(op.getLoc(), op.getName(),
                                                 op.getType(), attrs, argAttrs);
    rewriter.inlineRegionBefore(op.getBody(), newFunc.getBody(), newFunc.end());
    rewriter.eraseOp(op);

    // Insert yield check if this function is not extern
    auto *region = newFunc.getCallableRegion();
    if (region) {
      auto ip = rewriter.saveInsertionPoint();
      auto entry = &region->front();
      auto entryOp = &entry->front();
      // Splits `entry` returning a block containing all the previous
      // contents of entry since we're splitting on the first op. This
      // block is `doYield` because split it a second time to give us
      // the success block. Since the values in `entry` dominate both
      // splits, we don't have to pass arguments
      Block *doYield = entry->splitBlock(&entry->front());
      Block *dontYield = doYield->splitBlock(&doYield->front());
      // Insert yield check in original entry block
      rewriter.setInsertionPointToEnd(entry);

      const uint32_t MAX_REDUCTIONS = 20;  // TODO: Move this up in the compiler
      Value maxReductions =
          llvm_constant(i32Ty, ctx.getI32Attr(MAX_REDUCTIONS));
      rewriter.create<YieldCheckOp>(op.getLoc(), maxReductions, doYield,
                                    ValueRange{}, dontYield, ValueRange{});
      // Then insert the actual yield point in the yield block
      rewriter.setInsertionPointToEnd(doYield);
      rewriter.create<YieldOp>(op.getLoc());
      // Then post-yield, branch to the real entry block
      rewriter.create<BranchOp>(op.getLoc(), dontYield);
      // Reset the builder to where it was originally
      rewriter.restoreInsertionPoint(ip);
    }

    return success();
  }
};

struct ClosureOpConversion : public EIROpConversion<ClosureOp> {
  using EIROpConversion::EIROpConversion;

  LogicalResult matchAndRewrite(
      ClosureOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    ClosureOpAdaptor adaptor(operands);
    auto ctx = getRewriteContext(op, rewriter);

    auto loc = op.getLoc();

    auto envLen = op.envLen();
    unsigned arity = op.arity();
    unsigned index = op.index();
    unsigned oldUnique = op.oldUnique();
    StringRef unique = op.unique();
    FlatSymbolRefAttr callee = op.calleeAttr();
    bool isAnonymous = op.isAnonymous();

    LLVMType termTy = ctx.getUsizeType();
    LLVMType termPtrTy = termTy.getPointerTo();
    LLVMType i8Ty = ctx.getI8Type();
    LLVMType i8PtrTy = i8Ty.getPointerTo();
    LLVMType i32Ty = ctx.getI32Type();
    LLVMType i32PtrTy = i32Ty.getPointerTo();
    auto indexTy = rewriter.getIndexType();

    // Look for the callee, if it doesn't exist, create a default declaration
    SmallVector<LLVMType, 2> argTypes;
    // If we have an environment, the first argument is a closure
    if (envLen > 0) {
      argTypes.push_back(termPtrTy);
      unsigned withEnvArity = arity > 0 ? arity - 1 : 0;
      for (auto i = 0; i < withEnvArity; i++) {
        argTypes.push_back(termTy);
      }
    } else {
      // Otherwise, all arguments, if any, will be terms
      for (auto i = 0; i < arity; i++) {
        argTypes.push_back(termTy);
      }
    }
    auto target = ctx.getOrInsertFunction(callee.getValue(), termTy, argTypes);
    LLVMType targetType;
    if (auto tgt = dyn_cast_or_null<LLVM::LLVMFuncOp>(target))
      targetType = tgt.getType();
    else if (auto tgt = dyn_cast_or_null<mlir::FuncOp>(target))
      targetType =
          ctx.typeConverter.convertType(tgt.getType()).cast<LLVM::LLVMType>();
    else {
      auto ft = cast<FuncOp>(target).getType();
      auto tt = ctx.typeConverter.convertType(ft);
      targetType = tt.cast<LLVM::LLVMType>();
    }

    LLVMType opaqueFnTy = ctx.targetInfo.getOpaqueFnType();
    LLVMType closureTy = ctx.targetInfo.makeClosureType(envLen);
    LLVMType uniqueTy = ctx.targetInfo.getClosureUniqueType();
    LLVMType uniquePtrTy = uniqueTy.getPointerTo();
    LLVMType defTy = ctx.targetInfo.getClosureDefinitionType();

    // Allocate closure header block
    auto boxedClosureTy = BoxType::get(rewriter.getType<ClosureType>(envLen));
    auto closurePtrTy = PtrType::get(rewriter.getType<ClosureType>(envLen));
    auto headerArity = ctx.targetInfo.closureHeaderArity(envLen);
    Value headerArityConst =
        llvm_constant(termTy, ctx.getIntegerAttr(headerArity));
    auto mallocOp =
        rewriter.create<MallocOp>(loc, closurePtrTy, headerArityConst);
    auto valRef = mallocOp.getResult();

    // Calculate pointers to each field in the header and write the
    // corresponding data to it
    Value zero = llvm_constant(i32Ty, ctx.getI32Attr(0));

    // Header term
    auto closureHeader =
        ctx.targetInfo.encodeHeader(TypeKind::Closure, headerArity);
    Value header = llvm_constant(
        termTy, ctx.getIntegerAttr(closureHeader.getLimitedValue()));
    Value headerPtrGep = llvm_gep(termPtrTy, valRef, ValueRange{zero, zero});
    llvm_store(header, headerPtrGep);

    // Module atom
    Value mod = llvm_constant(
        termTy, ctx.getIntegerAttr(op.module().getValue().getLimitedValue()));
    Value modIdx = llvm_constant(i32Ty, ctx.getI32Attr(1));
    Value modPtrGep = llvm_gep(termPtrTy, valRef, ValueRange{zero, modIdx});
    llvm_store(mod, modPtrGep);

    // Arity
    // arity: u32,
    Value arityConst = llvm_constant(i32Ty, ctx.getIntegerAttr(arity));
    Value arityIdx = llvm_constant(i32Ty, ctx.getI32Attr(2));
    Value arityPtrGep = llvm_gep(i32PtrTy, valRef, ValueRange{zero, arityIdx});
    llvm_store(arityConst, arityPtrGep);

    // Definition
    Value defIdx = llvm_constant(i32Ty, ctx.getI32Attr(3));
    if (isAnonymous) {
      // Definition - tag
      Value anonTypeConst = llvm_constant(i32Ty, ctx.getI32Attr(1));
      Value defTagIdx = llvm_constant(i32Ty, ctx.getI32Attr(0));
      Value defTagPtrGep = llvm_gep(i32PtrTy, valRef, ValueRange{zero, defIdx, defTagIdx});
      llvm_store(anonTypeConst, defTagPtrGep);

      // Definition - index
      Value indexConst = llvm_constant(termTy, ctx.getIntegerAttr(index));
      Value defIndexIdx = llvm_constant(i32Ty, ctx.getI32Attr(1));
      Value definitionIndexGep = llvm_gep(termPtrTy, valRef, ValueRange{zero, defIdx, defIndexIdx});
      llvm_store(indexConst, definitionIndexGep);

      // Definition - unique
      Value uniqueConst = llvm_constant(uniqueTy, ctx.getStringAttr(unique));
      Value defUniqIdx = llvm_constant(i32Ty, ctx.getI32Attr(2));
      Value defUniqGep = llvm_gep(uniquePtrTy, valRef, ValueRange{zero, defIdx, defUniqIdx});
      llvm_store(uniqueConst, defUniqGep);

      // Definition - old_unique
      Value oldUniqueConst = llvm_constant(i32Ty, ctx.getI32Attr(oldUnique));
      Value defOldUniqIdx = llvm_constant(i32Ty, ctx.getI32Attr(3));
      Value defOldUniqGep =
        llvm_gep(i32PtrTy, valRef, ValueRange{zero, defIdx, defOldUniqIdx});
      llvm_store(oldUniqueConst, defOldUniqGep);
    } else {
      // Definition - tag
      Value exportTypeConst = llvm_constant(i32Ty, ctx.getI32Attr(0));
      Value defTagIdx = llvm_constant(i32Ty, ctx.getI32Attr(0));
      Value defTagPtrGep = llvm_gep(i32PtrTy, valRef, ValueRange{zero, defIdx, defTagIdx});
      llvm_store(exportTypeConst, defTagPtrGep);
    }

    // Code
    // code: Option<*const ()>,
    Value codePtr =
        llvm_addressof(targetType.getPointerTo(), callee.getValue());
    Value codeIdx = llvm_constant(i32Ty, ctx.getI32Attr(4));
    LLVMType opaqueFnPtrTy = opaqueFnTy.getPointerTo();
    Value codePtrGep =
      llvm_gep(opaqueFnPtrTy.getPointerTo(), valRef, ValueRange{zero, codeIdx});
    llvm_store(llvm_bitcast(opaqueFnPtrTy, codePtr), codePtrGep);

    // Env
    // env: [Term],
    if (isAnonymous) {
      auto opOperands = adaptor.operands();
      if (opOperands.size() > envLen)
        return op.emitOpError("mismatched closure env signature, expected ") << envLen << ", got " << opOperands.size();

      if (opOperands.size() > 0) {
        Value envIdx = llvm_constant(i32Ty, ctx.getI32Attr(CLOSURE_ENV_INDEX));

        for (auto it : llvm::enumerate(opOperands)) {
          Value operand = it.value();
          Value opIdx = llvm_constant(i32Ty, ctx.getI32Attr(it.index()));
          Value opPtrGep = llvm_gep(termPtrTy, valRef, ValueRange{zero, envIdx, opIdx});
          llvm_store(operand, opPtrGep);
        }
      }
    }

    rewriter.replaceOp(op, valRef);
    return success();
  }
};

struct UnpackEnvOpConversion : public EIROpConversion<UnpackEnvOp> {
  using EIROpConversion::EIROpConversion;

  LogicalResult matchAndRewrite(
      UnpackEnvOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    UnpackEnvOpAdaptor adaptor(operands);
    auto ctx = getRewriteContext(op, rewriter);

    LLVMType termTy = ctx.getUsizeType();
    LLVMType termPtrTy = termTy.getPointerTo();
    LLVMType i32Ty = ctx.getI32Type();

    Value env = adaptor.env();

    Value zero = llvm_constant(i32Ty, ctx.getI32Attr(0));
    Value fieldIdx = llvm_constant(i32Ty, ctx.getI32Attr(CLOSURE_ENV_INDEX));
    Value envIdx = llvm_constant(i32Ty, ctx.getI32Attr(op.envIndex()));

    Value ptr = llvm_gep(termPtrTy, env, ValueRange{zero, fieldIdx, envIdx});
    Value unpacked = llvm_load(ptr);

    rewriter.replaceOp(op, unpacked);
    return success();
  }
};

void populateFuncLikeOpConversionPatterns(OwningRewritePatternList &patterns,
                                          MLIRContext *context,
                                          EirTypeConverter &converter,
                                          TargetInfo &targetInfo) {
  patterns.insert<FuncOpConversion, ClosureOpConversion, UnpackEnvOpConversion>(
      context, converter, targetInfo);
}

}  // namespace eir
}  // namespace lumen
