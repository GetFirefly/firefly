#include "lumen/compiler/Dialect/EIR/Conversion/EIRToLLVM/MemoryOpConversions.h"

namespace lumen {
namespace eir {

struct CastOpConversion : public EIROpConversion<CastOp> {
  using EIROpConversion::EIROpConversion;

  LogicalResult matchAndRewrite(
      CastOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    CastOpOperandAdaptor adaptor(operands);
    auto ctx = getRewriteContext(op, rewriter);

    Value in = adaptor.input();

    auto termTy = ctx.getUsizeType();
    Type fromTy = op.getAttrOfType<TypeAttr>("from").getValue();
    Type toTy = op.getAttrOfType<TypeAttr>("to").getValue();

    // Remove redundant casts
    if (fromTy == toTy) {
      rewriter.replaceOp(op, in);
      return success();
    }

    // Get the lowered type of the input and result of this operation
    // LLVMType llvmFromTy =
    // ctx.typeConverter.convertType(fromTy).cast<LLVMType>(); LLVMType llvmToTy
    // = ctx.typeConverter.convertType(toTy).cast<LLVMType>();

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
        if (ft.isOpaque() && tt.isImmediate()) {
          rewriter.replaceOp(op, in);
          return success();
        }
        if (ft.isBox() && tt.isBox()) {
          auto tbt = ctx.typeConverter.convertType(tt.cast<BoxType>())
                         .cast<LLVMType>();
          Value cast = llvm_bitcast(tbt, in);
          rewriter.replaceOp(op, cast);
          return success();
        }
        llvm::outs() << "invalid opaque term cast: \n";
        llvm::outs() << "to: " << toTy << "\n";
        llvm::outs() << "from: " << fromTy << "\n";
        assert(false && "unexpected type cast");
        return failure();
      }
      /*
      // ..from boolean type
      if (llvmFromTy.isIntegerTy(1)) {
          // ..to boolean or atom type
          if (tt.isAtom()) {
            // Extend and encode as atom immediate
            Value extended = llvm_zext(termTy, in);
            rewriter.replaceOp(op, ctx.encodeImmediate(tt, extended));
            return success();
          }
          // ..to fixed-width integer type
          if (tt.isFixnum()) {
            // Extend and encode as fixnum
            Value extended = llvm_zext(termTy, in);
            rewriter.replaceOp(op, ctx.encodeImmediate(tt, extended));
            return success();
          }
          llvm::outs() << "invalid boolean cast: \nto: ";
          tt.dump();
          llvm::outs() << "\nfrom: ";
          llvmFromTy.dump();
          llvm::outs() << "\n";
          assert(false && "unexpected type cast");
          return failure();
      }
      // ..from pointer type
      if (llvmFromTy.isPointerTy()) {
          // ..to box
          if (tt.isBox()) {
              auto toBoxTy = toTy.cast<BoxType>();
              auto boxedTy = toBoxTy.getBoxedType();
              // ..boxed lists have their own encoding
              if (boxedTy.isa<ConsType>()) {
                  Value boxed = ctx.encodeList(in);
                  rewriter.replaceOp(op, boxed);
                  return success();
              } else {
                  Value boxed = ctx.encodeBox(in);
                  rewriter.replaceOp(op, boxed);
                  return success();
              }
          }
          llvm::outs() << "invalid pointer cast: \nto: ";
          tt.dump();
          llvm::outs() << "\nfrom: ";
          llvmFromTy.dump();
          llvm::outs() << "\n";
          assert(false && "unexpected type cast");
          return failure();
      }
      // ..from integer type
      if (fromTy.isIntOrIndex()) {
          // ..to fixed-width integer type
          if (tt.isFixnum()) {
            Value extended = llvm_sext(termTy, in);
            rewriter.replaceOp(op, ctx.encodeImmediate(tt, extended));
            return success();
          }
          llvm::outs() << "invalid integer cast: \nto: ";
          tt.dump();
          llvm::outs() << "\nfrom: ";
          fromTy.dump();
          llvm::outs() << "\n";
          assert(false && "unexpected type cast");
          return failure();
      }
      // ..from float type
      if (fromTy.isF64() || llvmFromTy.isFloatTy()) {
          // ..to float type
          if (tt.isFloat()) {
              // Immediate floats need no encoding, just bitcast to term type
              if (!ctx.targetInfo.requiresPackedFloats()) {
                  Value floatTerm = llvm_bitcast(termTy, in);
                  rewriter.replaceOp(op, floatTerm);
                  return success();
              }
              // Packed floats need to be allocated
              auto i32Ty = ctx.targetInfo.getI32Type();
              auto floatTy = ctx.targetInfo.getFloatType();
              auto floatPtrTy = floatTy.getPointerTo();
              auto f64PtrTy = floatTy.getStructElementType(1).getPointerTo();
              auto boxed = eir_malloc(BoxType::get(tt));
              auto valPtr = eir_cast(boxed, floatPtrTy);
              // Once we have a pointer to the float structure, calculate the
              // pointer to the value field and write the input value to that
      location Value zero = llvm_constant(i32Ty, ctx.getIntegerAttr(0)); Value
      one = llvm_constant(i32Ty, ctx.getIntegerAttr(1)); ArrayRef<Value>
      indices{zero, one}; Value fieldPtr = llvm_gep(f64PtrTy, valPtr, indices);
              llvm_store(in, fieldPtr);
              // Finally, replace the cast with the boxed value
              rewriter.replaceOp(op, {boxed});
              return success();
          }
          llvm::outs() << "invalid float cast: \nto: ";
          tt.dump();
          llvm::outs() << "\nfrom: ";
          fromTy.dump();
          llvm::outs() << "\n";
          assert(false && "unexpected type cast");
          return failure();
      }
  */
      llvm::outs() << "invalid float cast: \nto: ";
      tt.dump();
      llvm::outs() << "\nfrom: ";
      fromTy.dump();
      llvm::outs() << "\n";
      assert(false && "unexpected type cast");
      return failure();
    }

    /*
    // If we reach here, we're expecting to cast away the term wrapper into some
    lower primitive type if (auto ft =
    fromTy.dyn_cast_or_null<OpaqueTermType>()) {
        // ..to boolean type
        if (llvmToTy.isIntegerTy(1)) {
            auto i1Ty = ctx.targetInfo.getI1Type();
            // ..from boolean
            if (ft.isBoolean()) {
                // Decode and truncate
                Value decoded = ctx.decodeImmediate(in);
                Value truncated = llvm_trunc(i1Ty, decoded);
                rewriter.replaceOp(op, truncated);
                return success();
            }
            // ..from atom
            if (ft.isAtom()) {
                // Decode and insert check if value is boolean value
                Value decoded = ctx.decodeImmediate(in);
                Value one = llvm_constant(termTy, ctx.getIntegerAttr(1));
                Value cond = llvm_icmp(LLVM::ICmpPredicate::eq, decoded, one);
                rewriter.replaceOp(op, cond);
                return success();
            }
            llvm::outs() << "invalid primitive cast: \nto: ";
            llvmToTy.dump();
            llvm::outs() << "\nfrom opaque: ";
            ft.dump();
            llvm::outs() << "\n";
            assert(false && "unexpected type cast");
            return failure();
        }
        // ..to pointer type
        if (llvmToTy.isPointerTy()) {
            // ..from box
            if (ft.isBox()) {
                auto fromBoxTy = fromTy.cast<BoxType>();
                auto boxedTy = fromBoxTy.getBoxedType();
                auto innerTy =
    ctx.typeConverter.convertType(boxedTy).cast<LLVMType>();
                // Boxed lists have their own encoding
                if (boxedTy.isa<ConsType>()) {
                    Value unboxed = ctx.decodeList(in);
                    rewriter.replaceOp(op, unboxed);
                    return success();
                } else {
                    Value unboxed = ctx.decodeBox(innerTy, in);
                    rewriter.replaceOp(op, unboxed);
                    return success();
                }
            }
            llvm::outs() << "invalid pointer cast: \nto: ";
            llvmToTy.dump();
            llvm::outs() << "\nfrom opaque: ";
            ft.dump();
            llvm::outs() << "\n";
            assert(false && "unexpected type cast");
            return failure();
        }
        // ..to integer type
        if (toTy.isIntOrIndex()) {
            // ..from fixed-width integer type
            if (ft.isFixnum()) {
              Value decoded = ctx.decodeImmediate(in);
              rewriter.replaceOp(op, decoded);
              return success();
            }
            llvm::outs() << "invalid integer cast: \nto: ";
            toTy.dump();
            llvm::outs() << "\nfrom opaque: ";
            ft.dump();
            llvm::outs() << "\n";
            assert(false && "unexpected type cast");
            return failure();
        }
        // ..to float type
        if (toTy.isF64() || llvmToTy.isFloatTy()) {
            // ..from float type
            if (ft.isFloat()) {
                // Immediate floats need no decoding, just bitcast to type
                if (!ctx.targetInfo.requiresPackedFloats()) {
                    Value decoded = llvm_bitcast(llvmToTy, in);
                    rewriter.replaceOp(op, decoded);
                    return success();
                }
                // For packed floats, we just need to load the value from the
    structure auto i32Ty = ctx.targetInfo.getI32Type(); auto floatTy =
    ctx.targetInfo.getFloatType(); auto f64PtrTy =
    floatTy.getStructElementType(1).getPointerTo(); Value valPtr = eir_cast(in,
    floatTy.getPointerTo()); Value zero = llvm_constant(i32Ty,
    ctx.getIntegerAttr(0)); Value one = llvm_constant(i32Ty,
    ctx.getIntegerAttr(1)); ArrayRef<Value> indices{zero, one}; Value fieldPtr =
    llvm_gep(f64PtrTy, valPtr, indices); Value floatVal = llvm_load(fieldPtr);
                rewriter.replaceOp(op, floatVal);
                return success();
            }
            llvm::outs() << "invalid float cast: \nto: ";
            toTy.dump();
            llvm::outs() << "\nfrom opaque: ";
            ft.dump();
            llvm::outs() << "\n";
            assert(false && "unexpected type cast");
            return failure();
        }
        llvm::outs() << "invalid cast: \nto: ";
        toTy.dump();
        llvm::outs() << "\nfrom opaque: ";
        ft.dump();
        llvm::outs() << "\n";
        assert(false && "unexpected type cast");
        return failure();
    }
    */

    llvm::outs() << "invalid unknown cast: \nto: ";
    toTy.dump();
    llvm::outs() << "\nfrom: ";
    fromTy.dump();
    llvm::outs() << "\n";
    // Unsupported cast
    assert(false && "unexpected type cast");
    return failure();
  }
};

struct GetElementPtrOpConversion : public EIROpConversion<GetElementPtrOp> {
  using EIROpConversion::EIROpConversion;

  LogicalResult matchAndRewrite(
      GetElementPtrOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    GetElementPtrOpOperandAdaptor adaptor(operands);
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
    edsc::ScopedContext context(rewriter, op.getLoc());
    LoadOpOperandAdaptor adaptor(operands);

    Value ptr = adaptor.ref();
    Value load = llvm_load(ptr);

    rewriter.replaceOp(op, load);
    return success();
  }
};

void populateMemoryOpConversionPatterns(OwningRewritePatternList &patterns,
                                        MLIRContext *context,
                                        LLVMTypeConverter &converter,
                                        TargetInfo &targetInfo) {
  patterns
      .insert<CastOpConversion, GetElementPtrOpConversion, LoadOpConversion>(
          context, converter, targetInfo);
}

}  // namespace eir
}  // namespace lumen
