#include "lumen/EIR/Conversion/ConstantOpConversions.h"

namespace lumen {
namespace eir {

template <typename Op>
static Value lowerElementValue(RewritePatternContext<Op> &ctx,
                               Attribute elementAttr);

struct NullOpConversion : public EIROpConversion<NullOp> {
    using EIROpConversion::EIROpConversion;

    LogicalResult matchAndRewrite(
        NullOp op, ArrayRef<Value> operands,
        ConversionPatternRewriter &rewriter) const override {
        auto ctx = getRewriteContext(op, rewriter);

        auto ty = ctx.typeConverter.convertType(op.getType());
        rewriter.replaceOpWithNewOp<LLVM::NullOp>(op, ty);
        return success();
    }
};

struct ConstantAtomOpConversion : public EIROpConversion<ConstantAtomOp> {
    using EIROpConversion::EIROpConversion;

    LogicalResult matchAndRewrite(
        ConstantAtomOp op, ArrayRef<Value> operands,
        ConversionPatternRewriter &rewriter) const override {
        auto ctx = getRewriteContext(op, rewriter);

        auto atomAttr = op.getValue().cast<AtomAttr>();
        auto id = (uint64_t)atomAttr.getValue().getLimitedValue();

        if (op.getType().isa<BooleanType>()) {
            if (id > 1) {
                op.emitError("invalid atom used as boolean value");
                return failure();
            }

            // Lower to i1
            auto i1Ty = ctx.getI1Type();
            Value val = llvm_constant(i1Ty, ctx.getIntegerAttr(id));
            rewriter.replaceOp(op, {val});
        } else {
            // Lower to term
            auto immedTy = ctx.getOpaqueImmediateType();
            auto taggedAtom =
                ctx.targetInfo.encodeImmediate(TypeKind::Atom, id);
            Value val = llvm_constant(immedTy, ctx.getIntegerAttr(taggedAtom));
            rewriter.replaceOp(op, {val});
        }

        return success();
    }
};

struct ConstantBoolOpConversion : public EIROpConversion<ConstantBoolOp> {
    using EIROpConversion::EIROpConversion;

    LogicalResult matchAndRewrite(
        ConstantBoolOp op, ArrayRef<Value> operands,
        ConversionPatternRewriter &rewriter) const override {
        auto ctx = getRewriteContext(op, rewriter);

        auto attr = op.getValue().cast<BoolAttr>();
        auto isTrue = attr.getValue();
        auto valType = op.getType();

        // Can be lowered to atoms
        if (valType.isa<AtomType>() || valType.isa<BooleanType>()) {
            auto ty = ctx.getOpaqueImmediateType();
            auto taggedAtom = ctx.targetInfo.encodeImmediate(
                TypeKind::Atom, (unsigned)(isTrue));
            Value val = llvm_constant(ty, ctx.getIntegerAttr(taggedAtom));
            rewriter.replaceOp(op, {val});
            return success();
        }

        // Otherwise we are expecting this to be an integer type (i1 almost
        // always)
        if (valType.isInteger(1)) {
            auto ty = ctx.typeConverter.convertType(valType);
            Value val =
                llvm_constant(ty, ctx.getIntegerAttr((unsigned)(isTrue)));
            rewriter.replaceOp(op, {val});
            return success();
        }

        op.emitOpError("invalid type associated with constant boolean value");
        return failure();
    }
};

struct ConstantBigIntOpConversion : public EIROpConversion<ConstantBigIntOp> {
    using EIROpConversion::EIROpConversion;

    LogicalResult matchAndRewrite(
        ConstantBigIntOp op, ArrayRef<Value> _operands,
        ConversionPatternRewriter &rewriter) const override {
        auto ctx = getRewriteContext(op, rewriter);

        auto bigIntAttr = op.getValue().cast<APIntAttr>();
        auto bigIntStr = bigIntAttr.getValueAsString();
        auto termTy = ctx.getOpaqueTermType();
        auto usizeTy = ctx.getUsizeType();
        auto i8Ty = ctx.getI8Type();
        auto i8PtrTy = i8Ty.getPointerTo();

        // Create constant string to hold bigint value
        auto name = bigIntAttr.getHash();
        auto bytesGlobal = ctx.getOrInsertConstantString(name, bigIntStr);

        // Invoke the runtime function that will reify a BigInt term from the
        // constant string
        auto globalPtr = llvm_bitcast(i8PtrTy, llvm_addressof(bytesGlobal));
        Value size =
            llvm_constant(usizeTy, ctx.getIntegerAttr(bigIntStr.size()));

        StringRef symbolName("__lumen_builtin_bigint_from_cstr");
        auto callee =
            ctx.getOrInsertFunction(symbolName, termTy, {i8PtrTy, usizeTy});

        auto calleeSymbol = rewriter.getSymbolRefAttr(symbolName);
        rewriter.replaceOpWithNewOp<mlir::CallOp>(
            op, calleeSymbol, termTy, ArrayRef<Value>{globalPtr, size});

        return success();
    }
};

struct ConstantBinaryOpConversion : public EIROpConversion<ConstantBinaryOp> {
    using EIROpConversion::EIROpConversion;

    LogicalResult matchAndRewrite(
        ConstantBinaryOp op, ArrayRef<Value> operands,
        ConversionPatternRewriter &rewriter) const override {
        auto ctx = getRewriteContext(op, rewriter);

        auto binAttr = op.getValue().cast<BinaryAttr>();
        auto bytes = binAttr.getValue();
        auto ty = ctx.targetInfo.getBinaryType();
        auto usizeTy = ctx.getUsizeType();

        // We use the SHA-1 hash of the value as the name of the global,
        // this provides a nice way to de-duplicate constant strings while
        // not requiring any global state
        auto name = binAttr.getHash();
        auto bytesGlobal = ctx.getOrInsertConstantString(name, bytes);
        auto headerName = std::string("binary_") + name;
        ModuleOp mod = ctx.getModule();
        LLVM::GlobalOp headerConst =
            mod.lookupSymbol<LLVM::GlobalOp>(headerName);
        if (!headerConst) {
            auto i64Ty = ctx.getI64Type();
            auto i8Ty = ctx.getI8Type();
            auto i8PtrTy = i8Ty.getPointerTo();

            PatternRewriter::InsertionGuard insertGuard(rewriter);
            rewriter.setInsertionPointAfter(bytesGlobal);
            headerConst = ctx.getOrInsertGlobalConstantOp(headerName, ty);

            auto &initRegion = headerConst.getInitializerRegion();
            rewriter.createBlock(&initRegion);
            auto globalPtr = llvm_addressof(bytesGlobal);
            Value zero = llvm_constant(i64Ty, ctx.getIntegerAttr(0));
            Value headerTerm =
                llvm_constant(usizeTy, ctx.getIntegerAttr(binAttr.getHeader()));
            Value flags =
                llvm_constant(usizeTy, ctx.getIntegerAttr(binAttr.getFlags()));
            Value header = llvm_undef(ty);
            Value address =
                llvm_gep(i8PtrTy, globalPtr, ArrayRef<Value>{zero, zero});
            header = llvm_insertvalue(ty, header, headerTerm,
                                      rewriter.getI64ArrayAttr(0));
            header = llvm_insertvalue(ty, header, flags,
                                      rewriter.getI64ArrayAttr(1));
            header = llvm_insertvalue(ty, header, address,
                                      rewriter.getI64ArrayAttr(2));
            rewriter.create<LLVM::ReturnOp>(op.getLoc(), header);
        }

        // Box the constant address
        auto headerPtr = llvm_addressof(headerConst);
        auto boxed = ctx.encodeLiteral(headerPtr);

        rewriter.replaceOp(op, boxed);
        return success();
    }
};

// This magic constant here matches the same value in the term encoding in Rust
const uint64_t MIN_DOUBLE = ~((uint64_t)(INT64_MIN >> 12));

struct ConstantFloatOpConversion : public EIROpConversion<ConstantFloatOp> {
    using EIROpConversion::EIROpConversion;

    LogicalResult matchAndRewrite(
        ConstantFloatOp op, ArrayRef<Value> operands,
        ConversionPatternRewriter &rewriter) const override {
        auto ctx = getRewriteContext(op, rewriter);

        auto attr = op.getValue().cast<APFloatAttr>();
        auto apVal = attr.getValue();
        auto immedTy = ctx.getOpaqueImmediateType();

        // On nanboxed targets, floats are treated normally
        if (!ctx.useNanboxedFloats()) {
            auto f = apVal.bitcastToAPInt();
            // The magic constant here is MIN_DOUBLE, as defined in the term
            // encoding in Rust
            auto val = llvm_constant(
                immedTy, ctx.getIntegerAttr(f.getLimitedValue() + MIN_DOUBLE));
            rewriter.replaceOp(op, {val});
            return success();
        }

        // All other targets use boxed, packed floats
        // This requires generating a descriptor around the float,
        // which can then either be placed on the heap and boxed, or
        // passed by value on the stack and accessed directly

        auto f64Ty = ctx.getType<Float64Type>();
        auto val = llvm_constant(
            f64Ty, rewriter.getF64FloatAttr(apVal.convertToDouble()));
        auto headerName =
            std::string("float_") +
            std::to_string(apVal.bitcastToAPInt().getLimitedValue());
        ModuleOp mod = ctx.getModule();
        LLVM::GlobalOp headerConst =
            mod.lookupSymbol<LLVM::GlobalOp>(headerName);
        if (!headerConst) {
            auto i64Ty = ctx.getI64Type();
            auto i8Ty = ctx.getI8Type();

            PatternRewriter::InsertionGuard insertGuard(rewriter);
            auto body = mod.getBody();
            rewriter.setInsertionPointToStart(&body.front());
            headerConst = ctx.getOrInsertGlobalConstantOp(headerName, f64Ty);

            auto &initRegion = headerConst.getInitializerRegion();
            rewriter.createBlock(&initRegion);

            APInt headerTermVal = ctx.encodeHeaderConstant(TypeKind::Float, 2);
            Value headerTerm = llvm_constant(
                immedTy, ctx.getIntegerAttr(headerTermVal.getLimitedValue()));
            Value floatVal = llvm_constant(
                f64Ty, rewriter.getF64FloatAttr(apVal.convertToDouble()));
            Value header = llvm_undef(floatTy);
            header = llvm_insertvalue(floatTy, header, headerTerm,
                                      rewriter.getI64ArrayAttr(0));
            header = llvm_insertvalue(floatTy, header, floatVal,
                                      rewriter.getI64ArrayAttr(1));
            rewriter.create<LLVM::ReturnOp>(op.getLoc(), header);
        }

        // Box the constant address
        auto headerPtr = llvm_addressof(headerConst);
        auto boxed = ctx.encodeLiteral(headerPtr);
        rewriter.replaceOp(op, boxed);
        return success();
    }
};

struct ConstantIntOpConversion : public EIROpConversion<ConstantIntOp> {
    using EIROpConversion::EIROpConversion;

    LogicalResult matchAndRewrite(
        ConstantIntOp op, ArrayRef<Value> operands,
        ConversionPatternRewriter &rewriter) const override {
        auto ctx = getRewriteContext(op, rewriter);

        auto attr = op.getValue().cast<APIntAttr>();
        auto immedTy = ctx.getOpaqueImmediateType();
        auto value = attr.getValue();
        if (ctx.targetInfo.isValidImmediateValue(value)) {
            auto taggedInt = ctx.targetInfo.encodeImmediate(
                TypeKind::Fixnum, value.getLimitedValue());
            auto val = llvm_constant(immedTy, ctx.getIntegerAttr(taggedInt));

            rewriter.replaceOp(op, {val});
        } else {
            // Too large for an immediate, promote to bigint
            rewriter.replaceOpWithNewOp<ConstantBigIntOp>(op, value);
        }
        return success();
    }
};

struct ConstantNilOpConversion : public EIROpConversion<ConstantNilOp> {
    using EIROpConversion::EIROpConversion;

    LogicalResult matchAndRewrite(
        ConstantNilOp op, ArrayRef<Value> operands,
        ConversionPatternRewriter &rewriter) const override {
        auto ctx = getRewriteContext(op, rewriter);

        auto immedTy = ctx.getOpaqueImmediateType();
        auto val = llvm_constant(
            immedTy, ctx.getIntegerAttr(ctx.targetInfo.getNilValue()));

        rewriter.replaceOp(op, {val});
        return success();
    }
};

struct ConstantNoneOpConversion : public EIROpConversion<ConstantNoneOp> {
    using EIROpConversion::EIROpConversion;

    LogicalResult matchAndRewrite(
        ConstantNoneOp op, ArrayRef<Value> operands,
        ConversionPatternRewriter &rewriter) const override {
        auto ctx = getRewriteContext(op, rewriter);

        auto immedTy = ctx.getOpaqueImmediateType();
        auto val = llvm_constant(
            immedTy, ctx.getIntegerAttr(ctx.targetInfo.getNoneValue()));

        rewriter.replaceOp(op, {val});
        return success();
    }
};

struct ConstantListOpConversion : public EIROpConversion<ConstantListOp> {
    using EIROpConversion::EIROpConversion;

    LogicalResult matchAndRewrite(
        ConstantListOp op, ArrayRef<mlir::Value> operands,
        ConversionPatternRewriter &rewriter) const override {
        auto ctx = getRewriteContext(op, rewriter);

        auto attr = op.getValue().cast<SeqAttr>();
        auto elements = attr.getValue();
        auto numElements = elements.size();

        if (numElements == 0) {
            Value nil = eir_nil();
            rewriter.replaceOp(op, nil);
            return success();
        }

        SmallVector<Value, 4> elementValues;
        for (auto element : elements) {
            Value elementVal = lowerElementValue(ctx, element);
            assert(elementVal && "unsupported element type in cons cell");
            elementValues.push_back(elementVal);
        }

        rewriter.replaceOpWithNewOp<ListOp>(op, elementValues);
        return success();
    }
};

struct ConstantMapOpConversion : public EIROpConversion<ConstantMapOp> {
    using EIROpConversion::EIROpConversion;

    LogicalResult matchAndRewrite(
        ConstantMapOp op, ArrayRef<Value> _operands,
        ConversionPatternRewriter &rewriter) const override {
        auto ctx = getRewriteContext(op, rewriter);

        auto attr = op.getValue().cast<SeqAttr>();
        auto elementAttrs = attr.getValue();

        SmallVector<Value, 2> elements;
        for (auto elementAttr : elementAttrs) {
            auto element = lowerElementValue(ctx, elementAttr);
            assert(element && "unsupported element type in map");
            elements.push_back(element);
        }

        MapOp newMap = eir_map(elements);
        rewriter.replaceOp(op, newMap.out());

        return success();
    }
};

struct ConstantTupleOpConversion : public EIROpConversion<ConstantTupleOp> {
    using EIROpConversion::EIROpConversion;

    LogicalResult matchAndRewrite(
        ConstantTupleOp op, ArrayRef<Value> operands,
        ConversionPatternRewriter &rewriter) const override {
        auto ctx = getRewriteContext(op, rewriter);

        auto attr = op.getValue().cast<SeqAttr>();
        auto elementAttrs = attr.getValue();

        SmallVector<Value, 2> elements;
        for (auto elementAttr : elementAttrs) {
            auto element = lowerElementValue(ctx, elementAttr);
            assert(element && "unsupported element type in tuple");
            elements.push_back(element);
        }

        Value tuple = eir_tuple(elements);

        rewriter.replaceOp(op, tuple);

        return success();
    }
};

template <typename Op>
static Value lowerElementValue(RewritePatternContext<Op> &ctx,
                               Attribute elementAttr) {
    auto termTy = ctx.getOpaqueTermType();
    auto eirTermType = ctx.rewriter.template getType<TermType>();

    // Symbols
    if (auto symAttr = elementAttr.dyn_cast_or_null<FlatSymbolRefAttr>()) {
        ModuleOp mod = ctx.getModule();
        Operation *referencedOp =
            SymbolTable::lookupNearestSymbolFrom(mod, symAttr);
        auto symName = symAttr.getValue();
        // Symbol must be a global reference, with a name that contains the type
        // of global constant
        if (auto global = dyn_cast_or_null<LLVM::GlobalOp>(referencedOp)) {
            auto ptr = llvm_addressof(global);
            // Check name prefix, if we are able to in the future, I'd prefer
            // this to dispatch on type information, but the current APIs are
            // insufficient (and we don't have access to the original EIR type
            // here)
            if (symName.startswith("binary_")) {
                return ctx.encodeLiteral(ptr);
            }
            if (symName.startswith("float_")) {
                return ctx.encodeLiteral(ptr);
            }
            if (symName.startswith("closure_")) {
                return ctx.encodeLiteral(ptr);
            }
        }
        return nullptr;
    }
    // None/Nil
    if (auto typeAttr = elementAttr.dyn_cast_or_null<TypeAttr>()) {
        auto type = typeAttr.getValue();
        if (type.isa<NilType>()) {
            return eir_cast(eir_nil(), eirTermType);
        }
        if (type.isa<NoneType>()) {
            return eir_cast(eir_none(), eirTermType);
        }
        return nullptr;
    }
    // Booleans
    if (auto boolAttr = elementAttr.dyn_cast_or_null<BoolAttr>()) {
        auto b = boolAttr.getValue();
        uint64_t id = b ? 1 : 0;
        auto tagged = ctx.targetInfo.encodeImmediate(TypeKind::Atom, id);
        return llvm_constant(termTy, ctx.getIntegerAttr(tagged));
    }
    // Atoms
    if (auto atomAttr = elementAttr.dyn_cast_or_null<AtomAttr>()) {
        auto id = atomAttr.getValue().getLimitedValue();
        auto tagged = ctx.targetInfo.encodeImmediate(TypeKind::Atom, id);
        return llvm_constant(termTy, ctx.getIntegerAttr(tagged));
    }
    // Integers
    if (auto intAttr = elementAttr.dyn_cast_or_null<APIntAttr>()) {
        auto i = intAttr.getValue();
        assert(i.getBitWidth() <= ctx.targetInfo.pointerSizeInBits &&
               "support for bigint in constant aggregates not yet implemented");
        auto tagged = ctx.targetInfo.encodeImmediate(TypeKind::Fixnum,
                                                     i.getLimitedValue());
        return llvm_constant(termTy, ctx.getIntegerAttr(tagged));
    }
    if (auto intAttr = elementAttr.dyn_cast_or_null<IntegerAttr>()) {
        auto i = intAttr.getValue();
        assert(i.getBitWidth() <= ctx.targetInfo.pointerSizeInBits &&
               "support for bigint in constant aggregates not yet implemented");
        auto tagged = ctx.targetInfo.encodeImmediate(TypeKind::Fixnum,
                                                     i.getLimitedValue());
        return llvm_constant(termTy, ctx.getIntegerAttr(tagged));
    }
    // Floats
    if (auto floatAttr = elementAttr.dyn_cast_or_null<APFloatAttr>()) {
        if (!ctx.targetInfo.requiresPackedFloats()) {
            auto f = floatAttr.getValue().bitcastToAPInt() + MIN_DOUBLE;
            return llvm_constant(termTy,
                                 ctx.getIntegerAttr(f.getLimitedValue()));
        }
        // Packed float
        return eir_cast(eir_constant_float(floatAttr.getValue()), eirTermType);
    }
    if (auto floatAttr = elementAttr.dyn_cast_or_null<mlir::FloatAttr>()) {
        if (!ctx.targetInfo.requiresPackedFloats()) {
            auto f = floatAttr.getValue().bitcastToAPInt() + MIN_DOUBLE;
            return llvm_constant(termTy,
                                 ctx.getIntegerAttr(f.getLimitedValue()));
        }
        // Packed float
        return eir_cast(eir_constant_float(floatAttr.getValue()), eirTermType);
    }
    // Binaries
    if (auto binAttr = elementAttr.dyn_cast_or_null<BinaryAttr>()) {
        auto type = ctx.rewriter.template getType<BinaryType>();
        return eir_cast(eir_constant_binary(type, binAttr), eirTermType);
    }
    //  Nested aggregates
    if (auto aggAttr = elementAttr.dyn_cast_or_null<SeqAttr>()) {
        auto elementAttrs = aggAttr.getValue();
        // Tuples
        if (auto tupleTy = aggAttr.getType().dyn_cast_or_null<TupleType>()) {
            SmallVector<Value, 2> elements;
            for (auto elementAttr : elementAttrs) {
                auto element = lowerElementValue(ctx, elementAttr);
                assert(element && "unsupported element type in tuple");
                elements.push_back(element);
            }

            return eir_cast(eir_tuple(elements), eirTermType);
        }
        // Lists
        if (auto consTy = aggAttr.getType().dyn_cast_or_null<ConsType>()) {
            auto numElements = elementAttrs.size();

            if (numElements == 0) {
                return eir_cast(eir_nil(), eirTermType);
            }

            // Lower to single cons cell if it fits
            if (numElements < 2) {
                Value head = lowerElementValue(ctx, elementAttrs[0]);
                assert(head && "unsupported element type in cons cell");
                return eir_cast(eir_cons(head, eir_nil()), eirTermType);
            }

            unsigned cellsRequired = numElements;
            unsigned currentIndex = numElements;

            Value list;
            while (currentIndex > 0) {
                if (!list) {
                    Value tail =
                        lowerElementValue(ctx, elementAttrs[--currentIndex]);
                    assert(tail && "unsupported element type in cons cell");
                    Value head =
                        lowerElementValue(ctx, elementAttrs[--currentIndex]);
                    assert(head && "unsupported element type in cons cell");
                    list = eir_cons(head, tail);
                } else {
                    Value head =
                        lowerElementValue(ctx, elementAttrs[--currentIndex]);
                    assert(head && "unsupported element type in cons cell");
                    list = eir_cons(head, list);
                }
            }

            return eir_cast(list, eirTermType);
        }
    }

    return nullptr;
}

void populateConstantOpConversionPatterns(OwningRewritePatternList &patterns,
                                          MLIRContext *context,
                                          EirTypeConverter &converter,
                                          TargetPlatform &platform) {
    patterns.insert<ConstantAtomOpConversion, ConstantBoolOpConversion,
                    ConstantBigIntOpConversion, ConstantBinaryOpConversion,
                    ConstantFloatOpConversion, ConstantIntOpConversion,
                    ConstantListOpConversion, ConstantMapOpConversion,
                    ConstantNilOpConversion, ConstantNoneOpConversion,
                    ConstantTupleOpConversion, NullOpConversion>(
        context, converter, platform);
}

}  // namespace eir
}  // namespace lumen
