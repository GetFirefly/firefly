#include "lumen/compiler/Dialect/EIR/Conversion/EIRToLLVM/MapOpConversions.h"

namespace lumen {
namespace eir {

struct ConstructMapOpConversion : public EIROpConversion<ConstructMapOp> {
  using EIROpConversion::EIROpConversion;

  LogicalResult matchAndRewrite(
      ConstructMapOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = getRewriteContext(op, rewriter);
    auto loc = ctx.getLoc();
    ConstructMapOpOperandAdaptor adaptor(operands);

    auto termTy = ctx.getUsizeType();
    StringRef symbolName("__lumen_builtin_map.new");
    auto callee = ctx.getOrInsertFunction(symbolName, termTy, {});

    auto calleeSymbol =
        FlatSymbolRefAttr::get(symbolName, callee->getContext());
    auto newMapOp = rewriter.create<mlir::CallOp>(loc, calleeSymbol, termTy, ArrayRef<Value>{});
    Value newMap = newMapOp.getResult(0);

    auto numElements = operands.size();

    if (numElements == 0) {
      rewriter.replaceOp(op, {newMap});
      return success();
    }

    assert(numElements % 2 == 0 && "expected an even number of elements");

    StringRef insertSymbolName("__lumen_builtin_map.insert");
    auto insertCallee = ctx.getOrInsertFunction(insertSymbolName, termTy, {termTy, termTy});
    auto insertCalleeSymbol =
      FlatSymbolRefAttr::get(insertSymbolName, insertCallee->getContext());

    for (unsigned i = 0; i < numElements; i++) {
      Value key = operands[i];
      Value val = operands[++i];

      auto insertMapOp = rewriter.create<mlir::CallOp>(loc, insertCalleeSymbol, termTy, ArrayRef<Value>{newMap, key, val});
      newMap = insertMapOp.getResult(0);
    }

    rewriter.replaceOp(op, {newMap});
    return success();
  }
};

struct MapInsertOpConversion : public EIROpConversion<MapInsertOp> {
  using EIROpConversion::EIROpConversion;

  LogicalResult matchAndRewrite(
      MapInsertOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = getRewriteContext(op, rewriter);
    auto loc = ctx.getLoc();
    MapInsertOpOperandAdaptor adaptor(operands);

    auto termTy = ctx.getUsizeType();
    StringRef symbolName("__lumen_builtin_map.insert");
    auto callee = ctx.getOrInsertFunction(symbolName, termTy, {termTy, termTy, termTy});
    auto calleeSymbol =
        FlatSymbolRefAttr::get(symbolName, callee->getContext());

    Value map = adaptor.map();
    Value key = adaptor.key();
    Value value = adaptor.val();
    auto newMapOp = rewriter.create<mlir::CallOp>(loc, calleeSymbol, termTy, ArrayRef<Value>{map, key, value});
    Value newMap = newMapOp.getResult(0);
    Value noneVal = llvm_constant(termTy, ctx.getIntegerAttr(ctx.targetInfo.getNoneValue().getLimitedValue()));
    Value isOk = llvm_icmp(LLVM::ICmpPredicate::ne, newMap, noneVal);

    rewriter.replaceOp(op, {newMap, isOk});
    return success();
  }
};

struct MapUpdateOpConversion : public EIROpConversion<MapUpdateOp> {
  using EIROpConversion::EIROpConversion;

  LogicalResult matchAndRewrite(
      MapUpdateOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = getRewriteContext(op, rewriter);
    auto loc = ctx.getLoc();
    MapUpdateOpOperandAdaptor adaptor(operands);

    auto termTy = ctx.getUsizeType();
    StringRef symbolName("__lumen_builtin_map.update");
    auto callee = ctx.getOrInsertFunction(symbolName, termTy, {termTy, termTy, termTy});
    auto calleeSymbol =
        FlatSymbolRefAttr::get(symbolName, callee->getContext());

    Value map = adaptor.map();
    Value key = adaptor.key();
    Value value = adaptor.val();
    auto newMapOp = rewriter.create<mlir::CallOp>(loc, calleeSymbol, termTy, ArrayRef<Value>{map, key, value});
    Value newMap = newMapOp.getResult(0);
    Value noneVal = llvm_constant(termTy, ctx.getIntegerAttr(ctx.targetInfo.getNoneValue().getLimitedValue()));
    Value isOk = llvm_icmp(LLVM::ICmpPredicate::ne, newMap, noneVal);

    rewriter.replaceOp(op, {newMap, isOk});
    return success();
  }
};

void populateMapOpConversionPatterns(OwningRewritePatternList &patterns,
                                     MLIRContext *context,
                                     LLVMTypeConverter &converter,
                                     TargetInfo &targetInfo) {
  patterns.insert<ConstructMapOpConversion,
                  MapInsertOpConversion,
                  MapUpdateOpConversion>(context, converter, targetInfo);
}

}  // namespace eir
}  // namespace lumen
