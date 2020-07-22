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

template <typename Op, typename OperandAdaptor>
class BinaryMatchOpConversion : public EIROpConversion<Op> {
 public:
  explicit BinaryMatchOpConversion(MLIRContext *context,
                                   LLVMTypeConverter &converter,
                                   TargetInfo &targetInfo,
                                   mlir::PatternBenefit benefit = 1)
      : EIROpConversion<Op>::EIROpConversion(context, converter, targetInfo,
                                             benefit) {
    _termTy = targetInfo.getUsizeType();
  }

  LogicalResult matchAndRewrite(
      Op op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    OperandAdaptor adaptor(operands);
    auto ctx = getRewriteContext(op, rewriter);

    auto termTy = ctx.getUsizeType();
    auto matchResultTy = ctx.targetInfo.getMatchResultType();
    auto i1Ty = ctx.getI1Type();
    auto i8Ty = ctx.getI8Type();

    // Define match function to be called
    // __lumen_builtin_binary_match.<type>(bin, ..args.., size) -> matchResultTy
    StringRef symbolName = Op::builtinSymbol();

    SmallVector<LLVMType, 5> argTypes;
    argTypes.push_back(termTy);
    addExtraArgTypes(ctx, argTypes);
    argTypes.push_back(termTy);

    auto callee = ctx.getOrInsertFunction(symbolName, matchResultTy, argTypes);
    auto calleeSymbol =
        FlatSymbolRefAttr::get(symbolName, callee->getContext());

    // Get operands to work with
    auto bin = adaptor.bin();
    auto opArgs = adaptor.args();
    auto numOpArgs = opArgs.size();

    // Handle optional size parameter, using a none val to represent no size
    Value size;
    if (numOpArgs > 0) {
      assert(numOpArgs == 1 && "unexpected extra arguments to binary_match.raw");
      size = opArgs.front();
    } else {
      size = llvm_constant(termTy, ctx.getIntegerAttr(ctx.targetInfo.getNoneValue().getLimitedValue()));
    }

    SmallVector<Value, 5> args;
    args.push_back(bin);
    addExtraArgValues(op, ctx, args);
    args.push_back(size);
    assert(args.size() == argTypes.size() && "mismatched parameter types and values in match op");

    // Call the match function
    auto matchOp = rewriter.create<mlir::CallOp>(op.getLoc(), calleeSymbol, matchResultTy, args);

    // Obtain the result values from the match result structure and map them to the op outputs
    Value result = matchOp.getResult(0);
    Value matched = llvm_extractvalue(termTy, result, ctx.getI64ArrayAttr(0));
    Value tail = llvm_extractvalue(termTy, result, ctx.getI64ArrayAttr(1));
    Value successFlag = llvm_extractvalue(i1Ty, result, ctx.getI64ArrayAttr(2));

    rewriter.replaceOp(op, {matched, tail, successFlag});
    return success();
  }

 protected:
  virtual void addExtraArgTypes(RewritePatternContext<Op> &ctx, SmallVectorImpl<LLVMType> &types) const { return; };
  virtual void addExtraArgValues(Op &op, RewritePatternContext<Op> &ctx, SmallVectorImpl<Value> &args) const { return; };

 private:
  using EIROpConversion<Op>::getRewriteContext;

  Type _termTy;
};

struct BinaryMatchRawOpConversion
    : public BinaryMatchOpConversion<BinaryMatchRawOp, BinaryMatchRawOpOperandAdaptor> {
  using BinaryMatchOpConversion::BinaryMatchOpConversion;

    void addExtraArgTypes(RewritePatternContext<BinaryMatchRawOp> &ctx, SmallVectorImpl<LLVMType> &types) const override {
      types.push_back(ctx.getI8Type());
    }

    void addExtraArgValues(BinaryMatchRawOp &op, RewritePatternContext<BinaryMatchRawOp> &ctx, SmallVectorImpl<Value> &args) const override {
      auto i8Ty = ctx.getI8Type();
      Value unit = llvm_constant(i8Ty, ctx.getIntegerAttr(op.unitAttr().getValue().getLimitedValue()));
      args.push_back(unit);
    }
};

struct BinaryMatchIntegerOpConversion
    : public BinaryMatchOpConversion<BinaryMatchIntegerOp, BinaryMatchIntegerOpOperandAdaptor> {
  using BinaryMatchOpConversion::BinaryMatchOpConversion;

    void addExtraArgTypes(RewritePatternContext<BinaryMatchIntegerOp> &ctx, SmallVectorImpl<LLVMType> &types) const override {
      types.push_back(ctx.getI1Type());
      types.push_back(ctx.getUsizeType());
      types.push_back(ctx.getI8Type());
    }

    void addExtraArgValues(BinaryMatchIntegerOp &op, RewritePatternContext<BinaryMatchIntegerOp> &ctx, SmallVectorImpl<Value> &args) const override {
      auto i1Ty = ctx.getI1Type();
      auto i8Ty = ctx.getI8Type();
      auto usizeTy = ctx.getUsizeType();
      Value isSigned = llvm_constant(i1Ty, ctx.getIntegerAttr(op.isSignedAttr().getValue()));
      Value endianness = llvm_constant(usizeTy, ctx.getIntegerAttr(op.endiannessAttr().getValue().getLimitedValue()));
      Value unit = llvm_constant(i8Ty, ctx.getIntegerAttr(op.unitAttr().getValue().getLimitedValue()));
      args.push_back(isSigned);
      args.push_back(endianness);
      args.push_back(unit);
    }
};

struct BinaryMatchFloatOpConversion
    : public BinaryMatchOpConversion<BinaryMatchFloatOp, BinaryMatchFloatOpOperandAdaptor> {
  using BinaryMatchOpConversion::BinaryMatchOpConversion;

    void addExtraArgTypes(RewritePatternContext<BinaryMatchFloatOp> &ctx, SmallVectorImpl<LLVMType> &types) const override {
      types.push_back(ctx.getUsizeType());
      types.push_back(ctx.getI8Type());
    }

    void addExtraArgValues(BinaryMatchFloatOp &op, RewritePatternContext<BinaryMatchFloatOp> &ctx, SmallVectorImpl<Value> &args) const override {
      auto i8Ty = ctx.getI8Type();
      auto usizeTy = ctx.getUsizeType();
      Value endianness = llvm_constant(usizeTy, ctx.getIntegerAttr(op.endiannessAttr().getValue().getLimitedValue()));
      Value unit = llvm_constant(i8Ty, ctx.getIntegerAttr(op.unitAttr().getValue().getLimitedValue()));
      args.push_back(endianness);
      args.push_back(unit);
    }
};

struct BinaryMatchUtf8OpConversion
    : public BinaryMatchOpConversion<BinaryMatchUtf8Op, BinaryMatchUtf8OpOperandAdaptor> {
  using BinaryMatchOpConversion::BinaryMatchOpConversion;
};

struct BinaryMatchUtf16OpConversion
    : public BinaryMatchOpConversion<BinaryMatchUtf16Op, BinaryMatchUtf16OpOperandAdaptor> {
  using BinaryMatchOpConversion::BinaryMatchOpConversion;

    void addExtraArgTypes(RewritePatternContext<BinaryMatchUtf16Op> &ctx, SmallVectorImpl<LLVMType> &types) const override {
      types.push_back(ctx.getUsizeType());
    }

    void addExtraArgValues(BinaryMatchUtf16Op &op, RewritePatternContext<BinaryMatchUtf16Op> &ctx, SmallVectorImpl<Value> &args) const override {
      auto usizeTy = ctx.getUsizeType();
      Value endianness = llvm_constant(usizeTy, ctx.getIntegerAttr(op.endiannessAttr().getValue().getLimitedValue()));
      args.push_back(endianness);
    }
};

struct BinaryMatchUtf32OpConversion
    : public BinaryMatchOpConversion<BinaryMatchUtf32Op, BinaryMatchUtf32OpOperandAdaptor> {
  using BinaryMatchOpConversion::BinaryMatchOpConversion;

    void addExtraArgTypes(RewritePatternContext<BinaryMatchUtf32Op> &ctx, SmallVectorImpl<LLVMType> &types) const override {
      types.push_back(ctx.getUsizeType());
    }

    void addExtraArgValues(BinaryMatchUtf32Op &op, RewritePatternContext<BinaryMatchUtf32Op> &ctx, SmallVectorImpl<Value> &args) const override {
      auto usizeTy = ctx.getUsizeType();
      Value endianness = llvm_constant(usizeTy, ctx.getIntegerAttr(op.endiannessAttr().getValue().getLimitedValue()));
      args.push_back(endianness);
    }
};


void populateBinaryOpConversionPatterns(OwningRewritePatternList &patterns,
                                        MLIRContext *context,
                                        LLVMTypeConverter &converter,
                                        TargetInfo &targetInfo) {
  patterns.insert<BinaryStartOpConversion, BinaryFinishOpConversion,
                  BinaryPushOpConversion, BinaryMatchRawOpConversion,
                  BinaryMatchIntegerOpConversion, BinaryMatchFloatOpConversion,
                  BinaryMatchUtf8OpConversion, BinaryMatchUtf16OpConversion,
                  BinaryMatchUtf32OpConversion>(context, converter, targetInfo);
}

}  // namespace eir
}  // namespace lumen
