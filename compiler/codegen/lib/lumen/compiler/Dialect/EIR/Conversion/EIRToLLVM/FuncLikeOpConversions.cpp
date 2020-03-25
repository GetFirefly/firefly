#include "lumen/compiler/Dialect/EIR/Conversion/EIRToLLVM/FuncLikeOpConversions.h"

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
    edsc::ScopedContext scope(rewriter, op.getLoc());
    auto ctx = getRewriteContext(op, rewriter);

    SmallVector<NamedAttribute, 2> attrs;
    for (auto fa : op.getAttrs()) {
      if (fa.first.is(SymbolTable::getSymbolAttrName()) ||
          fa.first.is(::mlir::impl::getTypeAttrName())) {
        continue;
      }
    }
    SmallVector<NamedAttributeList, 4> argAttrs;
    for (unsigned i = 0, e = op.getNumArguments(); i < e; ++i) {
      auto aa = ::mlir::impl::getArgAttrs(op, i);
      argAttrs.push_back(NamedAttributeList(aa));
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

      const uint32_t MAX_REDUCTIONS = 20; // TODO: Move this up in the compiler
      Value maxReductions = rewriter.create<mlir::ConstantOp>(op.getLoc(), rewriter.getI32IntegerAttr(MAX_REDUCTIONS));
      rewriter.create<YieldCheckOp>(op.getLoc(), maxReductions, doYield, ValueRange{},
                                    dontYield, ValueRange{});
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
    ClosureOpOperandAdaptor adaptor(operands);
    auto ctx = getRewriteContext(op, rewriter);

    auto loc = ctx.getLoc();

    assert(op.isAnonymous() && "expected anonymous closures only");

    unsigned arity = op.arity();
    unsigned index = op.index();
    unsigned oldUnique = op.oldUnique();
    StringRef unique = op.unique();
    FlatSymbolRefAttr callee = op.calleeAttr();

    LLVMType termTy = ctx.getUsizeType();
    LLVMType termPtrTy = termTy.getPointerTo();
    LLVMType i8Ty = ctx.getI8Type();
    LLVMType i8PtrTy = i8Ty.getPointerTo();
    LLVMType i32Ty = ctx.getI32Type();
    LLVMType i32PtrTy = i32Ty.getPointerTo();
    auto indexTy = rewriter.getIndexType();

    // Look for the callee, if it doesn't exist, create a default declaration
    SmallVector<LLVMType, 2> argTypes;
    for (auto i = 0; i < arity; i++) {
      argTypes.push_back(termTy);
    }
    auto target = ctx.getOrInsertFunction(callee.getValue(), termTy, argTypes);

    auto envLen = op.envLen();
    LLVMType opaqueFnTy = ctx.targetInfo.getOpaqueFnType();
    LLVMType closureTy = ctx.targetInfo.makeClosureType(ctx.dialect, envLen);
    LLVMType uniqueTy = ctx.targetInfo.getClosureUniqueType();
    LLVMType uniquePtrTy = uniqueTy.getPointerTo();
    LLVMType defTy = ctx.targetInfo.getClosureDefinitionType();

    // Allocate closure header block
    auto boxedClosureTy = BoxType::get(rewriter.getType<ClosureType>());
    Value arityConst = llvm_constant(termTy, ctx.getIntegerAttr(arity));
    auto mallocOp = rewriter.create<MallocOp>(loc, boxedClosureTy, arityConst);
    auto valRef = mallocOp.getResult();

    // Calculate pointers to each field in the header and write the
    // corresponding data to it
    Value zero = llvm_constant(i32Ty, ctx.getI32Attr(0));

    // Header term
    auto closureHeader = ctx.targetInfo.encodeHeader(TypeKind::Closure, envLen);
    Value header = llvm_constant(
        termTy, ctx.getIntegerAttr(closureHeader.getLimitedValue()));
    Value headerIdx = llvm_constant(i32Ty, ctx.getI32Attr(0));
    ArrayRef<Value> headerIndices({zero, headerIdx});
    Value headerPtrGep = llvm_gep(termPtrTy, valRef, headerIndices);
    llvm_store(header, headerPtrGep);

    // Module atom
    Value mod = llvm_constant(
        termTy, ctx.getIntegerAttr(op.module().getValue().getLimitedValue()));
    Value modIdx = llvm_constant(i32Ty, ctx.getI32Attr(1));
    ArrayRef<Value> modIndices({zero, modIdx});
    Value modPtrGep = llvm_gep(termPtrTy, valRef, modIndices);
    llvm_store(mod, modPtrGep);

    // Definition
    Value defIdx = llvm_constant(i32Ty, ctx.getI32Attr(2));

    // Definition - type
    Value defTypeIdx = llvm_constant(i32Ty, ctx.getI32Attr(0));
    ArrayRef<Value> defTypeIndices({zero, defIdx, defTypeIdx});
    Value definitionTypePtrGep = llvm_gep(i8PtrTy, valRef, defTypeIndices);
    Value anonTypeConst = llvm_constant(i8Ty, ctx.getI8Attr(1));
    llvm_store(anonTypeConst, definitionTypePtrGep);

    // Definition - index
    Value defIndexIdx = llvm_constant(i32Ty, ctx.getI32Attr(1));
    ArrayRef<Value> defIndexIndices({zero, defIdx, defIndexIdx});
    Value definitionIndexGep = llvm_gep(termPtrTy, valRef, defIndexIndices);
    Value indexConst = llvm_constant(termTy, ctx.getIntegerAttr(index));
    llvm_store(indexConst, definitionIndexGep);

    // Definition - unique
    Value defUniqueIdx = llvm_constant(i32Ty, ctx.getI32Attr(2));
    ArrayRef<Value> defUniqueIndices({zero, defIdx, defUniqueIdx});
    Value definitionUniqueGep = llvm_gep(uniquePtrTy, valRef, defUniqueIndices);
    Value uniqueConst = llvm_constant(uniqueTy, ctx.getStringAttr(unique));
    llvm_store(uniqueConst, definitionUniqueGep);

    // Definition - old_unique
    Value defOldUniqueIdx = llvm_constant(i32Ty, ctx.getI32Attr(3));
    ArrayRef<Value> defOldUniqueIndices({zero, defIdx, defOldUniqueIdx});
    Value definitionOldUniqueGep =
        llvm_gep(i32PtrTy, valRef, defOldUniqueIndices);
    Value oldUniqueConst = llvm_constant(i32Ty, ctx.getI32Attr(oldUnique));
    llvm_store(oldUniqueConst, definitionOldUniqueGep);

    // Arity
    // arity: u8,
    Value arityIdx = llvm_constant(i32Ty, ctx.getI32Attr(3));
    ArrayRef<Value> arityIndices({zero, arityIdx});
    Value arityPtrGep = llvm_gep(i8PtrTy, valRef, arityIndices);
    llvm_store(llvm_trunc(i8Ty, arityConst), arityPtrGep);

    // Code
    // code: Option<*const ()>,
    Value codeIdx = llvm_constant(i32Ty, ctx.getI32Attr(4));
    ArrayRef<Value> codeIndices({zero, codeIdx});
    LLVMType opaqueFnPtrTy = opaqueFnTy.getPointerTo();
    Value codePtrGep =
        llvm_gep(opaqueFnPtrTy.getPointerTo(), valRef, codeIndices);
    Value codePtr =
        llvm_constant(opaqueFnPtrTy, rewriter.getSymbolRefAttr(target));
    llvm_store(llvm_bitcast(opaqueFnPtrTy, codePtr), codePtrGep);

    auto opOperands = adaptor.operands();
    if (opOperands.size() > 0) {
      Value envIdx = llvm_constant(i32Ty, ctx.getI32Attr(CLOSURE_ENV_INDEX));
      // Env
      // env: [Term],
      unsigned opIndex = 0;
      for (auto operand : opOperands) {
        Value opIdx = llvm_constant(i32Ty, ctx.getI32Attr(opIndex));
        ArrayRef<Value> opIndices({zero, envIdx, opIdx});
        Value opPtrGep = llvm_gep(termPtrTy, valRef, opIndices);
        llvm_store(operand, opPtrGep);
      }
    }

    // Box the allocated closure
    auto boxed = ctx.encodeBox(valRef);

    rewriter.replaceOp(op, boxed);
    return success();
  }
};

struct UnpackEnvOpConversion : public EIROpConversion<UnpackEnvOp> {
  using EIROpConversion::EIROpConversion;

  LogicalResult matchAndRewrite(
      UnpackEnvOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    UnpackEnvOpOperandAdaptor adaptor(operands);
    auto ctx = getRewriteContext(op, rewriter);

    LLVMType termTy = ctx.getUsizeType();
    LLVMType termPtrTy = termTy.getPointerTo();
    LLVMType i32Ty = ctx.getI32Type();

    Value env = adaptor.env();

    Value zero = llvm_constant(i32Ty, ctx.getI32Attr(0));
    Value fieldIndex = llvm_constant(i32Ty, ctx.getI32Attr(CLOSURE_ENV_INDEX));
    Value envIndex = llvm_constant(i32Ty, ctx.getI32Attr(op.envIndex()));

    ArrayRef<Value> indices({zero, fieldIndex, envIndex});
    Value ptr = llvm_gep(termPtrTy, env, indices);
    Value unpacked = llvm_load(ptr);

    rewriter.replaceOp(op, unpacked);
    return success();
  }
};

void populateFuncLikeOpConversionPatterns(OwningRewritePatternList &patterns,
                                          MLIRContext *context,
                                          LLVMTypeConverter &converter,
                                          TargetInfo &targetInfo) {
  patterns.insert<FuncOpConversion, ClosureOpConversion, UnpackEnvOpConversion>(
      context, converter, targetInfo);
}

}  // namespace eir
}  // namespace lumen
