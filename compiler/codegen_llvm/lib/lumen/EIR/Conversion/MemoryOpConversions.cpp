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

        auto immedTy = ctx.getOpaqueImmediateType();
        // We're expecting malloc target type to always be of PtrType
        PtrType ptrTy = op.getAllocType().dyn_cast<PtrType>();
        // The pointee (for now) is expected to be a term type
        Type innerTy = ptrTy.getPointeeType();
        auto innerTyInfo = dyn_cast<TermTypeInterface>(innerTy);
        if (!innerTy) return op.emitOpError("unsupported target type");

        auto ty = ctx.typeConverter.convertType(innerTy);
        if (!ty) return op.emitOpError("could not convert pointee type");

        if (innerTyInfo.isImmediate()) {
            Value zero = llvm_constant(immedTy, ctx.getIntegerAttr(0));
            Value allocPtr =
                ctx.buildMalloc(ty, innerTyInfo.getTypeKind().getValue(), zero);
            rewriter.replaceOp(op, allocPtr);
        } else {
            Value allocPtr = ctx.buildMalloc(
                ty, innerTyInfo.getTypeKind().getValue(), adaptor.arity());
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

        auto immedTy = ctx.getOpaqueImmediateType();
        auto termTy = ctx.getOpaqueTermType();
        TypeAttr fromAttr = op->getAttrOfType<TypeAttr>("from");
        assert(fromAttr && "expected cast to contain 'from' attribute!");
        TypeAttr toAttr = op->getAttrOfType<TypeAttr>("to");
        assert(toAttr && "expected cast to contain 'to' attribute!");
        Type fromTy = fromAttr.getValue();
        Type toTy = toAttr.getValue();

        // Remove redundant casts
        if (fromTy == toTy) {
            rewriter.replaceOp(op, in);
            return success();
        }

        // Casts to term types
        if (auto tt = toTy.dyn_cast<TermTypeInterface>()) {
            // ..from another term type
            if (auto ft = fromTy.dyn_cast<TermTypeInterface>()) {
                if (ft.isAtomLike() && (tt.isAtomLike()) ||
                    toTy.isa<TermType>()) {
                    // This cast is a no-op
                    rewriter.replaceOp(op, in);
                    return success();
                }
                if (ft.isImmediate() && toTy.isa<TermType>()) {
                    rewriter.replaceOp(op, in);
                    return success();
                }
                if (fromTy.isa<BoxType>() && toTy.isa<TermType>()) {
                    Value cast = llvm_bitcast(termTy, in);
                    rewriter.replaceOp(op, cast);
                    return success();
                }
                if (fromTy.isa<TermType>() && tt.isImmediate()) {
                    rewriter.replaceOp(op, in);
                    return success();
                }
                if (fromTy.isa<TermType>() && toTy.isa<BoxType>()) {
                    auto tbt =
                        ctx.typeConverter.convertType(toTy.cast<BoxType>());
                    Value cast = llvm_bitcast(tbt, in);
                    rewriter.replaceOp(op, cast);
                    return success();
                }
                if (fromTy.isa<BoxType>() && toTy.isa<BoxType>()) {
                    auto tbt =
                        ctx.typeConverter.convertType(toTy.cast<BoxType>());
                    Value cast = llvm_bitcast(tbt, in);
                    rewriter.replaceOp(op, cast);
                    return success();
                }

                op.emitError("unsupported or unimplemented term type cast");
                return failure();
            }

            if (isa_eir_type(fromTy)) {
                if (fromTy.isa<PtrType>()) {
                    if (toTy.isa<BoxType>() || toTy.isa<TermType>()) {
                        auto tbt = ctx.typeConverter.convertType(toTy);
                        auto ptrTy = fromTy.cast<PtrType>();
                        auto innerType = ptrTy.getPointeeType();
                        auto innerTypeInfo = cast<TermTypeInterface>(innerType);
                        Value encoded;
                        if (innerTypeInfo.isListLike())
                            encoded = ctx.encodeList(in);
                        else
                            encoded = ctx.encodeBox(in);
                        rewriter.replaceOp(op, encoded);
                        return success();
                    }
                }
            }

            if (isa_llvm_type(fromTy)) {
                // Handle converting from a raw pointer to an opaque term or
                // boxed term
                if (fromTy.isa<LLVM::LLVMPointerType>() &&
                    (toTy.isa<TermType>() || toTy.isa<BoxType>())) {
                    // We are converting an unboxed pointer to a boxed pointer
                    auto ptrTy = fromTy.cast<LLVMPointerType>();
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
                            auto boxedTy = boxTy.getPointeeType();
                            auto boxedTyInfo = cast<TermTypeInterface>(boxedTy);
                            if (boxedTyInfo.isListLike())
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
                    auto structTy = elTy.cast<LLVMStructType>();
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

                if (fromTy.isInteger(1) &&
                    (toTy.isa<BooleanType>() || toTy.isa<AtomType>() ||
                     toTy.isa<TermType>())) {
                    Value extended = llvm_zext(immedTy, in);
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
                if (toTy.isa<BooleanType>() || toTy.isa<AtomType>() ||
                    toTy.isa<TermType>()) {
                    Value extended = llvm_zext(immedTy, in);
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
            if (toTy.isa<PtrType>()) {
                if (auto ft = dyn_cast<TermTypeInterface>(fromTy)) {
                    if (fromTy.isa<BoxType>() || fromType.isa<TermType>()) {
                        auto ptrTy = toTy.cast<PtrType>();
                        auto innerTy = ptrTy.getPointeeType();
                        auto innerTyInfo = cast<TermTypeInterface>(innerTy);
                        Value decoded;
                        if (innerTy.isListLike())
                            decoded = ctx.decodeList(in);
                        else {
                            auto tbt = ctx.typeConverter.convertType(innerTy);
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
                auto fpt = ctx.typeConverter.convertType(fromTy);
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
        LLVMPointerType basePtrTy =
            basePtr.getType().dyn_cast_or_null<LLVMPointerType>();

        if (!basePtrTy)
            return op.emitOpError(
                "cannot perform this operation on a non-pointer type");

        auto index = op.getIndex();

        Type baseTy = basePtrTy.getElementType();
        Type elementTy;
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

        Type elementPtrTy = LLVMPointerType::get(elementTy);
        Type int32Ty = ctx.getI32Type();
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
                                        TargetPlatform &platform) {
    patterns.insert<MallocOpConversion, CastOpConversion,
                    GetElementPtrOpConversion, LoadOpConversion>(
        context, converter, platform);
}

}  // namespace eir
}  // namespace lumen
