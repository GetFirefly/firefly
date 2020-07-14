#include "lumen/compiler/Dialect/EIR/Conversion/EIRToLLVM/BinaryOpConversions.h"

namespace lumen {
namespace eir {

struct BinaryStartOpConversion : public EIROpConversion<BinaryStartOp> {
  using EIROpConversion::EIROpConversion;

  LogicalResult matchAndRewrite(
      BinaryStartOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = getRewriteContext(op, rewriter);

    auto termTy = ctx.getUsizeType();
    StringRef symbolName("__lumen_builtin_binary_start");
    auto callee = ctx.getOrInsertFunction(symbolName, termTy, {});

    auto calleeSymbol =
        FlatSymbolRefAttr::get(symbolName, callee->getContext());
    rewriter.replaceOpWithNewOp<mlir::CallOp>(op, calleeSymbol, termTy,
                                              operands);
    return success();
  }
};

struct BinaryFinishOpConversion : public EIROpConversion<BinaryFinishOp> {
  using EIROpConversion::EIROpConversion;

  LogicalResult matchAndRewrite(
      BinaryFinishOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = getRewriteContext(op, rewriter);

    auto termTy = ctx.getUsizeType();
    StringRef symbolName("__lumen_builtin_binary_finish");
    auto callee = ctx.getOrInsertFunction(symbolName, termTy, {termTy});

    auto calleeSymbol =
        FlatSymbolRefAttr::get(symbolName, callee->getContext());
    rewriter.replaceOpWithNewOp<mlir::CallOp>(op, calleeSymbol, termTy,
                                              operands);
    return success();
  }
};

struct BinaryPushOpConversion : public EIROpConversion<BinaryPushOp> {
  using EIROpConversion::EIROpConversion;

  LogicalResult matchAndRewrite(
      BinaryPushOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = getRewriteContext(op, rewriter);
    BinaryPushOpOperandAdaptor adaptor(operands);

    auto termTy = ctx.getUsizeType();

    Value bin = adaptor.bin();
    Value value = adaptor.value();
    ArrayRef<Value> sizeOpt = adaptor.size();
    Value size;
    if (sizeOpt.size() > 0) {
      size = sizeOpt.front();
    } else {
      auto taggedSize = ctx.targetInfo.encodeImmediate(TypeKind::Fixnum, 0);
      size = llvm_constant(termTy, ctx.getIntegerAttr(taggedSize));
    }

    auto pushType = static_cast<uint32_t>(
        op.getAttrOfType<IntegerAttr>("type").getValue().getLimitedValue());

    unsigned unit = 1;
    auto endianness = Endianness::Big;
    bool isSigned = false;

    auto termPtrTy = termTy.getPointerTo();
    auto i1Ty = ctx.getI1Type();
    auto i8Ty = ctx.getI8Type();
    auto i32Ty = ctx.getI32Type();
    auto pushTy = ctx.targetInfo.getBinaryPushResultType();

    mlir::CallOp pushOp;
    switch (pushType) {
      case BinarySpecifierType::Bytes:
      case BinarySpecifierType::Bits: {
        unit = static_cast<unsigned>(
            op.getAttrOfType<IntegerAttr>("unit").getValue().getLimitedValue());
        Value unitVal = llvm_constant(i8Ty, ctx.getI8Attr(unit));
        if (sizeOpt.size() > 0) {
          StringRef symbolName("__lumen_builtin_binary_push_binary");
          // __lumen_builtin_binary_push_binary(bin, value, size, unit)
          auto callee = ctx.getOrInsertFunction(symbolName, pushTy, {termTy, termTy, termTy, i8Ty});
          auto calleeSymbol =
            FlatSymbolRefAttr::get(symbolName, callee->getContext());
          ArrayRef<Value> args({bin, value, size, unitVal});
          pushOp = rewriter.create<mlir::CallOp>(op.getLoc(), calleeSymbol, pushTy, args);
        } else {
          StringRef symbolName("__lumen_builtin_binary_push_binary_all");
          // __lumen_builtin_binary_push_binary_all(bin, value, unit)
          auto callee = ctx.getOrInsertFunction(symbolName, pushTy, {termTy, termTy, i8Ty});
          auto calleeSymbol =
            FlatSymbolRefAttr::get(symbolName, callee->getContext());
          ArrayRef<Value> args({bin, value, unitVal});
          pushOp = rewriter.create<mlir::CallOp>(op.getLoc(), calleeSymbol, pushTy, args);
        }
        break;
      }
      case BinarySpecifierType::Utf8: {
        StringRef symbolName("__lumen_builtin_binary_push_utf8");
        // __lumen_builtin_binary_push_utf8(bin, value)
        auto callee = ctx.getOrInsertFunction(symbolName, pushTy, {termTy, termTy});
        auto calleeSymbol =
          FlatSymbolRefAttr::get(symbolName, callee->getContext());
        ArrayRef<Value> args({bin, value});
        pushOp = rewriter.create<mlir::CallOp>(op.getLoc(), calleeSymbol, pushTy, args);
        break;
      }
      case BinarySpecifierType::Utf16: {
        StringRef symbolName("__lumen_builtin_binary_push_utf16");
        // __lumen_builtin_binary_push_utf16(bin, value, signed, endianness)
        auto callee = ctx.getOrInsertFunction(symbolName, pushTy, {termTy, termTy, i1Ty, i32Ty});
        auto calleeSymbol =
          FlatSymbolRefAttr::get(symbolName, callee->getContext());
        endianness = static_cast<Endianness::Type>(
            op.getAttrOfType<IntegerAttr>("endianness")
                .getValue()
                .getLimitedValue());
        isSigned = op.getAttrOfType<BoolAttr>("signed").getValue();
        Value signedVal = llvm_constant(i1Ty, ctx.getI1Attr(isSigned));
        Value endiannessVal = llvm_constant(i32Ty, ctx.getI32Attr(endianness));
        ArrayRef<Value> args({bin, value, signedVal, endiannessVal});
        pushOp = rewriter.create<mlir::CallOp>(op.getLoc(), calleeSymbol, pushTy, args);
        break;
      }
      case BinarySpecifierType::Utf32: {
        StringRef symbolName("__lumen_builtin_binary_push_utf32");
        // __lumen_builtin_binary_push_utf32(bin, value, size, unit, signed, endianness)
        auto callee = ctx.getOrInsertFunction(symbolName, pushTy, {termTy, termTy, termTy, i8Ty, i1Ty, i32Ty});
        auto calleeSymbol =
          FlatSymbolRefAttr::get(symbolName, callee->getContext());
        unit = static_cast<unsigned>(
            op.getAttrOfType<IntegerAttr>("unit").getValue().getLimitedValue());
        endianness = static_cast<Endianness::Type>(
            op.getAttrOfType<IntegerAttr>("endianness")
                .getValue()
                .getLimitedValue());
        isSigned = op.getAttrOfType<BoolAttr>("signed").getValue();
        Value unitVal = llvm_constant(i8Ty, ctx.getI8Attr(unit));
        Value signedVal = llvm_constant(i1Ty, ctx.getI1Attr(isSigned));
        Value endiannessVal = llvm_constant(i32Ty, ctx.getI32Attr(endianness));
        ArrayRef<Value> args({bin, value, size, unitVal, signedVal, endiannessVal});
        pushOp = rewriter.create<mlir::CallOp>(op.getLoc(), calleeSymbol, pushTy, args);
        break;
      }
      case BinarySpecifierType::Integer: {
        StringRef symbolName("__lumen_builtin_binary_push_integer");
        // __lumen_builtin_binary_push_integer(bin, value, size, unit, signed, endianness)
        auto callee = ctx.getOrInsertFunction(symbolName, pushTy, {termTy, termTy, termTy, i8Ty, i1Ty, i32Ty});
        auto calleeSymbol =
          FlatSymbolRefAttr::get(symbolName, callee->getContext());
        unit = static_cast<unsigned>(
            op.getAttrOfType<IntegerAttr>("unit").getValue().getLimitedValue());
        endianness = static_cast<Endianness::Type>(
            op.getAttrOfType<IntegerAttr>("endianness")
                .getValue()
                .getLimitedValue());
        isSigned = op.getAttrOfType<BoolAttr>("signed").getValue();
        Value unitVal = llvm_constant(i8Ty, ctx.getI8Attr(unit));
        Value signedVal = llvm_constant(i1Ty, ctx.getI1Attr(isSigned));
        Value endiannessVal = llvm_constant(i32Ty, ctx.getI32Attr(endianness));
        ArrayRef<Value> args({bin, value, size, unitVal, signedVal, endiannessVal});
        pushOp = rewriter.create<mlir::CallOp>(op.getLoc(), calleeSymbol, pushTy, args);
        break;
      }
      case BinarySpecifierType::Float: {
        StringRef symbolName("__lumen_builtin_binary_push_float");
        // __lumen_builtin_binary_push_float(bin, value, size, unit, signed, endianness)
        auto callee = ctx.getOrInsertFunction(symbolName, pushTy, {termTy, termTy, termTy, i8Ty, i1Ty, i32Ty});
        auto calleeSymbol =
          FlatSymbolRefAttr::get(symbolName, callee->getContext());
        unit = static_cast<unsigned>(
            op.getAttrOfType<IntegerAttr>("unit").getValue().getLimitedValue());
        endianness = static_cast<Endianness::Type>(
            op.getAttrOfType<IntegerAttr>("endianness")
                .getValue()
                .getLimitedValue());
        isSigned = op.getAttrOfType<BoolAttr>("signed").getValue();
        Value unitVal = llvm_constant(i8Ty, ctx.getI8Attr(unit));
        Value signedVal = llvm_constant(i1Ty, ctx.getI1Attr(isSigned));
        Value endiannessVal = llvm_constant(i32Ty, ctx.getI32Attr(endianness));
        ArrayRef<Value> args({bin, value, size, unitVal, signedVal, endiannessVal});
        pushOp = rewriter.create<mlir::CallOp>(op.getLoc(), calleeSymbol, pushTy, args);
        break;
      }
      default:
        llvm_unreachable(
            "invalid binary specifier type encountered during conversion");
    }

    Value result = pushOp.getResult(0);
    Value newBin = llvm_extractvalue(termTy, result, ctx.getI64ArrayAttr(0));
    Value successFlag = llvm_extractvalue(i1Ty, result, ctx.getI64ArrayAttr(1));

    rewriter.replaceOp(op, {newBin, successFlag});
    return success();
  }
};

void populateBinaryOpConversionPatterns(OwningRewritePatternList &patterns,
                                        MLIRContext *context,
                                        LLVMTypeConverter &converter,
                                        TargetInfo &targetInfo) {
  patterns.insert<BinaryStartOpConversion, BinaryFinishOpConversion,
                  BinaryPushOpConversion>(context, converter, targetInfo);
}

}  // namespace eir
}  // namespace lumen
