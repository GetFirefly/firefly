#include "lumen/EIR/Conversion/MapOpConversions.h"

namespace lumen {
namespace eir {

struct ConstructMapOpConversion : public EIROpConversion<ConstructMapOp> {
  using EIROpConversion::EIROpConversion;

  LogicalResult matchAndRewrite(
      ConstructMapOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = getRewriteContext(op, rewriter);
    auto loc = ctx.getLoc();
    ConstructMapOpAdaptor adaptor(operands);

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
    MapInsertOpAdaptor adaptor(operands);

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
    MapUpdateOpAdaptor adaptor(operands);

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

struct MapIsKeyOpConversion : public EIROpConversion<MapIsKeyOp> {
  using EIROpConversion::EIROpConversion;

  LogicalResult matchAndRewrite(
      MapIsKeyOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = getRewriteContext(op, rewriter);
    MapIsKeyOpAdaptor adaptor(operands);

    auto termTy = ctx.getUsizeType();
    auto i1Ty = ctx.getI1Type();
    StringRef symbolName("__lumen_builtin_map.is_key");
    auto callee = ctx.getOrInsertFunction(symbolName, i1Ty, {termTy, termTy});
    auto calleeSymbol =
        FlatSymbolRefAttr::get(symbolName, callee->getContext());

    Value map = adaptor.map();
    Value key = adaptor.key();
    rewriter.replaceOpWithNewOp<mlir::CallOp>(op, calleeSymbol, i1Ty, ArrayRef<Value>{map, key});
    return success();
  }
};

struct MapGetKeyOpConversion : public EIROpConversion<MapGetKeyOp> {
  using EIROpConversion::EIROpConversion;

  LogicalResult matchAndRewrite(
      MapGetKeyOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = getRewriteContext(op, rewriter);
    MapGetKeyOpAdaptor adaptor(operands);

    auto termTy = ctx.getUsizeType();
    StringRef symbolName("__lumen_builtin_map.get");
    auto callee = ctx.getOrInsertFunction(symbolName, termTy, {termTy, termTy});
    auto calleeSymbol =
        FlatSymbolRefAttr::get(symbolName, callee->getContext());

    Value map = adaptor.map();
    Value key = adaptor.key();
    rewriter.replaceOpWithNewOp<mlir::CallOp>(op, calleeSymbol, termTy, ArrayRef<Value>{map, key});
    return success();
  }
};

void populateMapOpConversionPatterns(OwningRewritePatternList &patterns,
                                     MLIRContext *context,
                                     LLVMTypeConverter &converter,
                                     TargetInfo &targetInfo) {
  patterns.insert<ConstructMapOpConversion,
                  MapInsertOpConversion,
                  MapUpdateOpConversion,
                  MapIsKeyOpConversion,
                  MapGetKeyOpConversion>(context, converter, targetInfo);
}

}  // namespace eir
}  // namespace lumen
