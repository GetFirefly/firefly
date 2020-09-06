#include "lumen/EIR/Conversion/MemoryOpConversions.h"

namespace lumen {
namespace eir {

struct CastOpConversion : public EIROpConversion<CastOp> {
  using EIROpConversion::EIROpConversion;

  LogicalResult matchAndRewrite(
      CastOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    CastOpAdaptor adaptor(operands);
    auto ctx = getRewriteContext(op, rewriter);

    Value in = adaptor.input();

    auto termTy = ctx.getUsizeType();
    TypeAttr fromAttr = op.getAttrOfType<TypeAttr>("from");
    assert(fromAttr && "expected cast to contain 'from' attribute!");
    TypeAttr toAttr = op.getAttrOfType<TypeAttr>("to");
    assert(toAttr && "expected cast to contain 'to' attribute!");
    Type fromTy = fromAttr.getValue();
    Type toTy = toAttr.getValue();

    // Remove redundant casts
    if (fromTy == toTy) {
      rewriter.replaceOp(op, in);
      return success();
    }

    // Casts to term types
    if (auto tt = toTy.dyn_cast_or_null<OpaqueTermType>()) {
      // ..from another term type
      if (auto ft = fromTy.dyn_cast_or_null<OpaqueTermType>()) {
        if (ft.isBoolean()) {
          if (tt.isAtom() || tt.isOpaque()) {
            // Extend and encode as atom immediate
            Value extended = llvm_zext(termTy, in);
            auto atomTy = ctx.rewriter.getType<AtomType>();
            rewriter.replaceOp(op, ctx.encodeImmediate(atomTy, extended));
            return success();
          }
        }
        if (ft.isAtom() && tt.isBoolean()) {
          // Decode and truncate
          auto i1Ty = ctx.getI1Type();
          Value decoded = ctx.decodeImmediate(in);
          Value truncated = llvm_trunc(i1Ty, decoded);
          rewriter.replaceOp(op, truncated);
          return success();
        }
        if (ft.isImmediate() && tt.isOpaque()) {
          rewriter.replaceOp(op, in);
          return success();
        }
        if (ft.isBox() && tt.isOpaque()) {
          auto box = ft.cast<BoxType>();
          auto boxed = box.getBoxedType();
          Value encoded;
          if (boxed.isNonEmptyList())
            encoded = ctx.encodeList(in);
          else
            encoded = ctx.encodeBox(in);
          rewriter.replaceOp(op, encoded);
          return success();
        }
        if (ft.isOpaque() && tt.isImmediate()) {
          rewriter.replaceOp(op, in);
          return success();
        }
        if (ft.isOpaque() && tt.isBox()) {
          auto box = tt.cast<BoxType>();
          auto boxed = box.getBoxedType();
          Value decoded;
          if (boxed.isNonEmptyList()) {
            decoded = ctx.decodeList(in);
          } else {
            auto tbt = ctx.typeConverter.convertType(boxed)
                          .cast<LLVMType>();
            decoded = ctx.decodeBox(tbt, in);
          }
          rewriter.replaceOp(op, decoded);
          return success();
        }
        if (ft.isBox() && tt.isBox()) {
          auto tbt = ctx.typeConverter.convertType(tt.cast<BoxType>())
                         .cast<LLVMType>();
          Value cast = llvm_bitcast(tbt, in);
          rewriter.replaceOp(op, cast);
          return success();
        }

        op.emitError("unsupported or unimplemented term type cast");
        return failure();
      }

      if (auto llvmFromTy = fromTy.dyn_cast_or_null<LLVMType>()) {
        if (llvmFromTy.isPointerTy() && tt.isBox()) {
          auto tbt = ctx.typeConverter.convertType(tt).cast<LLVMType>();
          Value cast = llvm_bitcast(tbt, in);
          rewriter.replaceOp(op, cast);
          return success();
        }

        op.emitError("unsupported or unimplemented llvm type cast");
        return failure();
      }

      op.emitError("unsupported or unimplemented source type cast");
      return failure();
    }

    op.emitError("unsupported or unimplemented target type cast");
    return failure();
  }
};

struct GetElementPtrOpConversion : public EIROpConversion<GetElementPtrOp> {
  using EIROpConversion::EIROpConversion;

  LogicalResult matchAndRewrite(
      GetElementPtrOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    GetElementPtrOpAdaptor adaptor(operands);
    auto ctx = getRewriteContext(op, rewriter);

    Value base = adaptor.base();
    LLVMType baseTy = base.getType().cast<LLVMType>();

    Value pointeeCast;
    if (baseTy.isPointerTy()) {
      pointeeCast = base;
    } else {
      Type pointeeTy = op.getPointeeType();
      if (pointeeTy.isa<ConsType>()) {
        pointeeCast = ctx.decodeList(base);
      } else if (pointeeTy.isa<TupleType>()) {
        auto innerTy =
            ctx.typeConverter.convertType(pointeeTy).cast<LLVMType>();
        pointeeCast = ctx.decodeBox(innerTy, base);
      } else {
        op.emitError("invalid pointee value: expected cons or tuple");
        return failure();
      }
    }

    LLVMType elementTy =
        ctx.typeConverter.convertType(op.getElementType()).cast<LLVMType>();
    LLVMType elementPtrTy = elementTy.getPointerTo();
    LLVMType int32Ty = ctx.getI32Type();
    Value zero = llvm_constant(int32Ty, ctx.getI32Attr(0));
    Value index = llvm_constant(int32Ty, ctx.getI32Attr(op.getIndex()));
    ArrayRef<Value> indices({zero, index});
    Value gep = llvm_gep(elementPtrTy, pointeeCast, indices);
    /*
    Type resultTyOrig = op.getType();
    auto resultTy =
    ctx.typeConverter.convertType(resultTyOrig).cast<LLVMType>(); LLVMType ptrTy
    = resultTy.getPointerTo(); auto int32Ty = ctx.getI32Type();

    Value cns0 = llvm_constant(int32Ty, ctx.getI32Attr(0));
    Value index = llvm_constant(int32Ty, ctx.getI32Attr(op.getIndex()));
    ArrayRef<Value> indices({cns0, index});
    Value gep = llvm_gep(ptrTy, base, indices);
    */

    rewriter.replaceOp(op, gep);
    return success();
  }
};

struct LoadOpConversion : public EIROpConversion<LoadOp> {
  using EIROpConversion::EIROpConversion;

  LogicalResult matchAndRewrite(
      LoadOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = getRewriteContext(op, rewriter);
    LoadOpAdaptor adaptor(operands);

    Value ptr = adaptor.ref();
    Value load = llvm_load(ptr);

    rewriter.replaceOp(op, load);
    return success();
  }
};

void populateMemoryOpConversionPatterns(OwningRewritePatternList &patterns,
                                        MLIRContext *context,
                                        EirTypeConverter &converter,
                                        TargetInfo &targetInfo) {
  patterns
      .insert<CastOpConversion, GetElementPtrOpConversion, LoadOpConversion>(
          context, converter, targetInfo);
}

}  // namespace eir
}  // namespace lumen
