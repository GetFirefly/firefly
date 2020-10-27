#include "lumen/EIR/Conversion/MemoryOpConversions.h"

namespace lumen {
namespace eir {

struct MallocOpConversion : public EIROpConversion<MallocOp> {
    using EIROpConversion::EIROpConversion;

    LogicalResult matchAndRewrite(
        MallocOp op, ArrayRef<Value> operands,
        ConversionPatternRewriter &rewriter) const override {
        MallocOpAdaptor adaptor(operands);
        auto ctx = getRewriteContext(op, rewriter);

        // We're expecting malloc target type to always be of PtrType
        PtrType ptrTy = op.getAllocType().dyn_cast<PtrType>();
        // The pointee (for now) is expected to be a term type
        OpaqueTermType innerTy =
            ptrTy.getInnerType().dyn_cast<OpaqueTermType>();
        if (!innerTy) return op.emitOpError("unsupported target type");

        auto ty = ctx.typeConverter.convertType(innerTy).cast<LLVMType>();

        if (innerTy.hasDynamicExtent()) {
            Value allocPtr = ctx.buildMalloc(
                ty, innerTy.getTypeKind().getValue(), adaptor.arity());
            rewriter.replaceOp(op, allocPtr);
        } else {
            Value zero =
                llvm_constant(ctx.getUsizeType(), ctx.getIntegerAttr(0));
            Value allocPtr =
                ctx.buildMalloc(ty, innerTy.getTypeKind().getValue(), zero);
            rewriter.replaceOp(op, allocPtr);
        }

        return success();
    }
};

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
                if (ft.isAtom() && (tt.isAtom()) || tt.isOpaque()) {
                    // This cast is a no-op
                    rewriter.replaceOp(op, in);
                    return success();
                }
                if (ft.isImmediate() && tt.isOpaque()) {
                    rewriter.replaceOp(op, in);
                    return success();
                }
                if (ft.isBox() && tt.isOpaque()) {
                    Value cast = llvm_bitcast(termTy, in);
                    rewriter.replaceOp(op, cast);
                    return success();
                }
                if (ft.isOpaque() && tt.isImmediate()) {
                    rewriter.replaceOp(op, in);
                    return success();
                }
                if (ft.isOpaque() && tt.isBox()) {
                    auto tbt = ctx.typeConverter.convertType(tt.cast<BoxType>())
                                   .cast<LLVMType>();
                    Value cast = llvm_bitcast(tbt, in);
                    rewriter.replaceOp(op, cast);
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

            if (isa_eir_type(fromTy)) {
                if (fromTy.isa<PtrType>() || fromTy.isa<RefType>()) {
                    if (tt.isBox() || tt.isOpaque()) {
                        OpaqueTermType innerType;
                        auto tbt =
                            ctx.typeConverter.convertType(tt).cast<LLVMType>();
                        if (auto ptrTy = fromTy.dyn_cast_or_null<PtrType>())
                            innerType =
                                ptrTy.getInnerType().cast<OpaqueTermType>();
                        else
                            innerType = fromTy.cast<RefType>()
                                            .getInnerType()
                                            .cast<OpaqueTermType>();
                        Value encoded;
                        if (innerType.isNonEmptyList())
                            encoded = ctx.encodeList(in);
                        else
                            encoded = ctx.encodeBox(in);
                        rewriter.replaceOp(op, encoded);
                        return success();
                    }
                }
            }

            if (auto llvmFromTy = fromTy.dyn_cast_or_null<LLVMType>()) {
                // Handle converting from a raw pointer to an opaque term or
                // boxed term
                if (llvmFromTy.isPointerTy() && (tt.isOpaque() || tt.isBox())) {
                    // We are converting an unboxed pointer to a boxed pointer
                    auto ptrTy = llvmFromTy.cast<LLVM::LLVMPointerType>();
                    auto elTy = ptrTy.getElementType();
                    // If the inner type is not u8 or a struct, then this is not
                    // a valid conversion
                    if (!elTy.isInteger(8) && !elTy.isStructTy()) {
                        op.emitError(
                            "unsupported source type for pointer cast to term "
                            "type");
                        return failure();
                    }

                    // Raw pointer, this kind of cast shouldn't occur anymore,
                    // but we can handle it if we have type information, so do
                    // so
                    if (elTy.isInteger(8)) {
                        // We can simply bitcast the pointer and tag it
                        Value boxed;
                        if (auto boxTy = tt.dyn_cast_or_null<BoxType>()) {
                            auto boxedTy = boxTy.getBoxedType();
                            if (boxedTy.isNonEmptyList())
                                boxed = ctx.encodeList(in);
                            else
                                boxed = ctx.encodeBox(in);
                            rewriter.replaceOp(op, boxed);
                            return success();
                        } else {
                            op.emitError(
                                "insufficient target type information for cast "
                                "from raw "
                                "pointer");
                            return failure();
                        }
                    }

                    // A raw pointer to (presumably) a term structure
                    auto structTy = elTy.cast<LLVM::LLVMStructType>();
                    auto structName = structTy.getName();
                    auto isCastable =
                        StringSwitch<bool>(structName)
                            .Case("binary", true)
                            .Case("cons", true)
                            .Case("bigint", true)
                            .Case("float", true)
                            .Default(structName.startswith("closure") ||
                                     structName.startswith("tuple"));
                    // If we don't recognize the target type, we can't cast,
                    // otherwise we box the pointer appropriately
                    if (isCastable) {
                        Value boxed;
                        if (structName == "cons")
                            boxed = ctx.encodeList(in);
                        else
                            boxed = ctx.encodeBox(in);
                        rewriter.replaceOp(op, boxed);
                        return success();
                    } else {
                        op.emitError(
                            "unsupported cast from llvm type to term type");
                        return failure();
                    }
                }

                if (llvmFromTy.isIntegerTy(1) &&
                    (tt.isBoolean() || tt.isAtom() || tt.isOpaque())) {
                    Value extended = llvm_zext(termTy, in);
                    auto atomTy = ctx.rewriter.getType<AtomType>();
                    rewriter.replaceOp(op,
                                       ctx.encodeImmediate(atomTy, extended));
                    return success();
                }

                op.emitError("unsupported or unimplemented llvm type cast");
                return failure();
            }

            if (fromTy.isInteger(1)) {
                // Standard dialect booleans may occasionally need
                // casting to our boolean type, or promoted to an atom
                if (tt.isBoolean() || tt.isAtom() || tt.isOpaque()) {
                    Value extended = llvm_zext(termTy, in);
                    auto atomTy = ctx.rewriter.getType<AtomType>();
                    rewriter.replaceOp(op,
                                       ctx.encodeImmediate(atomTy, extended));
                    return success();
                }
            }

            op.emitError("unsupported or unimplemented source type cast");
            return failure();
        }

        if (isa_eir_type(toTy)) {
            // Raw pointers/references require encoding/decoding
            // to properly perform the cast
            if (toTy.isa<PtrType>() || toTy.isa<RefType>()) {
                if (auto ft = fromTy.dyn_cast_or_null<OpaqueTermType>()) {
                    if (ft.isBox() || ft.isOpaque()) {
                        OpaqueTermType innerTy;
                        if (auto ptrTy = toTy.dyn_cast_or_null<PtrType>())
                            innerTy =
                                ptrTy.getInnerType().cast<OpaqueTermType>();
                        else
                            innerTy = ft.cast<RefType>()
                                          .getInnerType()
                                          .cast<OpaqueTermType>();
                        Value decoded;
                        if (innerTy.isNonEmptyList())
                            decoded = ctx.decodeList(in);
                        else {
                            auto tbt = ctx.typeConverter.convertType(innerTy)
                                           .cast<LLVMType>();
                            decoded = ctx.decodeBox(tbt, in);
                        }
                        rewriter.replaceOp(op, decoded);
                        return success();
                    }

                    op.emitError(
                        "unsupported cast to pointer-like type from "
                        "non-pointer value");
                    return failure();
                }

                // If converting a raw LLVM pointer to Ptr/Ref, we don't need to
                // do anything other than a bitcast
                auto fpt =
                    ctx.typeConverter.convertType(fromTy).cast<LLVMType>();
                if (fpt.isPointerTy()) {
                    Value cast = llvm_bitcast(fpt, in);
                    rewriter.replaceOp(op, cast);
                    return success();
                }

                op.emitOpError("unsupported cast to pointer-like type");
                return failure();
            }
        }

        if (fromTy.isa<BooleanType>()) {
            if (toTy.isInteger(1)) {
                auto i1Ty = ctx.getI1Type();
                Value truncated = llvm_trunc(i1Ty, in);
                rewriter.replaceOp(op, truncated);
                return success();
            }

            if (auto llvmToTy = toTy.dyn_cast_or_null<LLVMType>()) {
                if (llvmToTy.isIntegerTy(1)) {
                    Value truncated = llvm_trunc(llvmToTy, in);
                    rewriter.replaceOp(op, truncated);
                    return success();
                }
            }
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

        Value basePtr = adaptor.base();
        LLVMType basePtrTy = basePtr.getType().cast<LLVMType>();

        if (!basePtrTy.isPointerTy())
            return op.emitOpError(
                "cannot perform this operation on a non-pointer type");

        auto index = op.getIndex();

        LLVMType baseTy = basePtrTy.getPointerElementTy();
        LLVMType elementTy;
        if (auto structTy = baseTy.dyn_cast<LLVMStructType>()) {
            auto elementTys = structTy.getBody();
            if (index >= elementTys.size())
                return op.emitOpError("invalid element index, only ")
                       << elementTys.size() << " fields, but wanted " << index;
            elementTy = elementTys[index];
        } else if (auto arrayTy = baseTy.dyn_cast<LLVMArrayType>()) {
            elementTy = arrayTy.getElementType();
        } else {
            return op.emitOpError(
                "invalid base pointer type, expected aggregate type");
        }

        LLVMType elementPtrTy = elementTy.getPointerTo();
        LLVMType int32Ty = ctx.getI32Type();
        Value zero = llvm_constant(int32Ty, ctx.getI32Attr(0));
        Value gepIndex = llvm_constant(int32Ty, ctx.getI32Attr(index));
        ArrayRef<Value> indices({zero, gepIndex});
        Value gep = llvm_gep(elementPtrTy, basePtr, indices);

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
    patterns.insert<MallocOpConversion, CastOpConversion,
                    GetElementPtrOpConversion, LoadOpConversion>(
        context, converter, targetInfo);
}

}  // namespace eir
}  // namespace lumen
