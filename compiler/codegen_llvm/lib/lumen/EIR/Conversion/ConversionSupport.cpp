#include "lumen/EIR/Conversion/ConversionSupport.h"

namespace lumen {
namespace eir {

bool isa_eir_type(Type t) { return isa<eirDialect>(t.getDialect()); }

bool isa_std_type(Type t) {
    auto &dialect = t.getDialect();
    return isa<mlir::StandardOpsDialect>(dialect) ||
           isa<mlir::BuiltinDialect>(dialect);
}

bool isa_llvm_type(Type t) {
    return isa<mlir::LLVM::LLVMDialect>(t.getDialect());
}

// Create an LLVM IR structure type if there is more than one result.
Type EirTypeConverter::packFunctionResults(TargetPlatform &platform,
                                           ArrayRef<Type> types) {
    assert(!types.empty() && "expected non-empty list of type");

    if (types.size() == 1) return convertType(types.front());

    SmallVector<Type, 8> resultTypes;
    resultTypes.reserve(types.size());
    for (auto t : types) {
        auto converted = convertType(t);
        if (!converted) return {};

        resultTypes.push_back(converted.dyn_cast<Type>());
    }

    MLIRContext *context = &getContext();
    auto termTy = ::mlir::IntegerType::get(context, pointerSizeInBits);
    return LLVMStructType::getLiteral(context, resultTypes, /*packed*/ false);
}

// Function types are converted to LLVM Function types by recursively converting
// argument and result types.  If MLIR Function has zero results, the LLVM
// Function has one VoidType result.  If MLIR Function has more than one result,
// they are into an LLVM StructType in their order of appearance.
LLVMFunctionType convertFunctionSignature(
    EirTypeConverter &converter, TargetPlatform &platform,
    mlir::FunctionType type, bool isVariadic,
    LLVMTypeConverter::SignatureConversion &result) {
    // Convert argument types one by one and check for errors.
    for (auto &en : llvm::enumerate(type.getInputs())) {
        auto llvmTy = convertType(en.value(), converter, platform);
        if (!llvmTy.hasValue()) return {};

        result.addInputs(en.index(), llvmTy.getValue());
    }

    SmallVector<Type, 8> argTypes;
    argTypes.reserve(llvm::size(result.getConvertedTypes()));
    for (Type ty : result.getConvertedTypes()) {
        argTypes.push_back(llvm::unwrap(ty));
    }

    // If function does not return anything, create the void result type,
    // if it returns on element, convert it, otherwise pack the result types
    // into a struct.
    Type resultType = type.getNumResults() == 0
                          ? LLVMVoidType::get(type.getContext())
                          : llvm::unwrap(converter.packFunctionResults(
                                targetInfo, type.getResults()));
    if (!resultType) return {};
    return LLVMFunctionType::get(resultType, argTypes, isVariadic);
}

Optional<Type> convertType(Type type, EirTypeConverter &converter,
                           TargetPlatform &platform) {
    // If we already have an LLVM type, we're good to go
    if (isa_llvm_type(type)) return type;

    if (auto funTy = type.dyn_cast_or_null<mlir::FunctionType>()) {
        LLVMTypeConverter::SignatureConversion conversion(funTy.getNumInputs());
        Type converted =
            convertFunctionSignature(converter, platform, funTy,
                                     /*isVariadic=*/false, conversion);
        if (!converted) return llvm::None;
        return converted;
    }

    // If this isn't otherwise an EIR type, we can't convert it
    if (!isa_eir_type(type)) return converter.deferTypeConversion(type);

    MLIRContext *context = &converter.getContext();
    auto immediateTy =
        ::mlir::IntegerType::get(context, converter.getPointerWidth());
    auto termTy = LLVMPointerType::get(immediateTy, 1);

    // Opaque terms are the most common
    if (type.isa<TermType>() || type.isa<NumberType>() || type.isa<ListType>())
        return termTy;

    // Boxes are translated as opaque terms, since they are not
    // technically valid pointers
    if (type.isa<BoxType>()) return termTy;

    if (auto ptrTy = type.dyn_cast_or_null<PtrType>()) {
        auto innerTy = converter.convertType(ptrTy.getPointeeType());
        return LLVMPointerType::get(innerTy);
    }

    if (auto recvRef = type.dyn_cast_or_null<ReceiveRefType>()) {
        return targetInfo.getReceiveRefType();
    }

    if (auto traceRef = type.dyn_cast_or_null<TraceRefType>()) {
        return targetInfo.getTraceRefType();
    }

    TermTypeInterface ty = type.cast<TermTypeInterface>();

    // All immediates are translated to the same type
    if (ty.isImmediate(platform)) return immediateTy;

    if (ty.isa<ConsType>()) return targetInfo.getConsType();
    if (ty.isNonEmptyList()) return targetInfo.getConsType();

    if (auto tupleTy = type.dyn_cast_or_null<eir::TupleType>()) {
        auto arity = tupleTy.size();
        if (arity > 0) {
            SmallVector<Type, 2> elementTypes;
            elementTypes.reserve(arity);
            for (auto ty : tupleTy.getTypes()) {
                auto elemTy = converter.convertType(ty);
                elementTypes.push_back(elemTy);
            }
            return targetInfo.makeTupleType(context, elementTypes);
        } else {
            return termTy;
        }
    }

    if (auto closureTy = type.dyn_cast_or_null<ClosureType>()) {
        return targetInfo.makeClosureType(closureTy.getEnvSize());
    }

    if (type.isa<eir::BinaryType>()) {
        return targetInfo.getBinaryType();
    }

    llvm::outs() << "\ntype: ";
    type.dump();
    llvm::outs() << "\n";
    assert(false && "unimplemented type conversion");

    return llvm::None;
}

LLVM::GlobalOp OpConversionContext::getOrInsertGlobalString(
    ModuleOp mod, StringRef name, StringRef value) const {
    assert(!name.empty() && "cannot create unnamed global string!");

    auto extendedName = name.str() + "_ptr";

    // Create the global at the entry of the module.
    LLVM::GlobalOp globalConst = getOrInsertConstantString(mod, name, value);
    LLVM::GlobalOp global = mod.lookupSymbol<LLVM::GlobalOp>(extendedName);
    if (!global) {
        auto i8Ty = getI8Type();
        Type i8PtrTy = LLVMPointerType::get(i8Ty);
        auto i64Ty = getI64Type();

        PatternRewriter::InsertionGuard insertGuard(rewriter);
        // Make sure we insert this global after the definition of the constant
        rewriter.setInsertionPointAfter(globalConst);
        // Insert the global definition
        global = getOrInsertGlobalConstantOp(mod, extendedName, i8PtrTy);
        // Initialize the global with a pointer to the first char of the
        // constant string
        auto &initRegion = global.getInitializerRegion();
        rewriter.createBlock(&initRegion);
        auto globalPtr = llvm_addressof(globalConst);
        Value zero = llvm_constant(i64Ty, getIntegerAttr(0));
        ArrayRef<Value> indices{zero, zero};
        Value address = llvm_gep(i8PtrTy, globalPtr, indices);
        rewriter.create<mlir::ReturnOp>(global.getLoc(), address);
    }

    return global;
}

Value OpConversionContext::buildMalloc(ModuleOp mod, Type ty, unsigned allocTy,
                                       Value arity) const {
    auto i8PtrTy = targetInfo.getI8Type().getPointerTo(1);
    auto ptrTy = ty.getPointerTo(1);
    auto i32Ty = targetInfo.getI32Type();
    auto usizeTy = getUsizeType();
    StringRef symbolName("__lumen_builtin_malloc");
    auto callee =
        getOrInsertFunction(mod, symbolName, i8PtrTy, {i32Ty, usizeTy});
    auto allocTyConst = llvm_constant(i32Ty, getU32Attr(allocTy));
    auto calleeSymbol = getSymbolRefAttr(symbolName);
    ArrayRef<Value> args{allocTyConst, arity};
    Operation *call = std_call(calleeSymbol, ArrayRef<Type>{i8PtrTy}, args);
    return llvm_bitcast(ptrTy, call->getResult(0));
}

Value OpConversionContext::encodeList(Value cons, bool isLiteral) const {
    // TODO: Possibly use ptrmask intrinsic
    auto immedTy = getUsizeType();
    auto termTy = getOpaqueTermType();
    auto termTyAddr0 = getOpaqueTermTypeAddr0();
    Value addr0 = llvm_addrspacecast(termTyAddr0, llvm_bitcast(termTy, cons));
    Value ptrInt = llvm_ptrtoint(immedTy, addr0);
    Value tag;
    if (isLiteral) {
        Value listTag =
            llvm_constant(immedTy, getIntegerAttr(targetInfo.listTag()));
        Value literalTag =
            llvm_constant(immedTy, getIntegerAttr(targetInfo.literalTag()));
        tag = llvm_or(listTag, literalTag);
    } else {
        tag = llvm_constant(immedTy, getIntegerAttr(targetInfo.listTag()));
    }
    Value taggedAddr0 = llvm_inttoptr(termTyAddr0, llvm_or(ptrInt, tag));
    return llvm_addrspacecast(termTy, taggedAddr0);
}

Value OpConversionContext::encodeBox(Value val) const {
    // TODO: Possibly use ptrmask intrinsic
    auto rawTag = targetInfo.boxTag();
    auto termTy = getOpaqueTermType();
    // No boxing required, pointers are pointers
    if (rawTag == 0) {
        // No boxing required, pointers are pointers,
        // we should be operating on an addrspace(1) pointer
        // here, so all we need is a bitcast to term type.
        return llvm_bitcast(termTy, val);
    } else {
        auto immedTy = getOpaqueImmediateType();
        auto termTyAddr0 = getOpaqueTermTypeAddr0();
        // We need to tag the pointer, so that means casting to addrspace(0),
        // bitcasting to term type, then ptrtoint, then back again at the end
        Value addr0 =
            llvm_addrspacecast(termTyAddr0, llvm_bitcast(termTy, val));
        Value ptrInt = llvm_ptrtoint(immedTy, addr0);
        Value tag = llvm_constant(termTy, getIntegerAttr(rawTag));
        Value taggedAddr0 = llvm_inttoptr(termTyAddr0, llvm_or(ptrInt, tag));
        return llvm_addrspacecast(termTy, taggedAddr0);
    }
}

Value OpConversionContext::encodeLiteral(Value val) const {
    // TODO: Possibly use ptrmask intrinsic
    auto rawTag = targetInfo.literalTag();
    auto immedTy = getOpaqueImmediateType();
    auto termTy = getOpaqueTermType();
    auto termTyAddr0 = getOpaqueTermTypeAddr0();
    Value addr0 = llvm_addrspacecast(termTyAddr0, llvm_bitcast(termTy, val));
    Value ptrInt = llvm_ptrtoint(immedTy, addr0);
    Value tag = llvm_constant(termTy, getIntegerAttr(rawTag));
    Value taggedAddr0 = llvm_inttoptr(termTyAddr0, llvm_or(ptrInt, tag));
    return llvm_addrspacecast(termTy, taggedAddr0);
}

Value OpConversionContext::encodeImmediate(ModuleOp mod, Location loc, Type ty,
                                           Value val) const {
    auto tyInfo = cast<TermTypeInterface>(ty);
    auto immedTy = getOpaqueImmediateType();
    auto termTy = getOpaqueTermType();
    auto i32Ty = getI32Type();
    StringRef symbolName("__lumen_builtin_encode_immediate");
    auto callee =
        getOrInsertFunction(mod, symbolName, termTy, {i32Ty, immedTy});
    auto calleeSymbol = getSymbolRefAttr(symbolName);

    Value kind =
        llvm_constant(i32Ty, getI32Attr(tyInfo.getTypeKind().getValue()));
    ArrayRef<Value> args{kind, val};
    Operation *call = std_call(calleeSymbol, ArrayRef<Type>{termTy}, args);
    return call->getResult(0);
}

Value OpConversionContext::decodeBox(Type innerTy, Value box) const {
    // TODO: Possibly use ptrmask intrinsic
    auto immedTy = getOpaqueImmediateType();
    auto termTy = getOpaqueTermType();
    auto termTyAddr0 = getOpaqueTermTypeAddr0();
    auto boxTy = box.getType();
    assert(boxTy == termTy && "expected boxed pointer type");
    auto rawTag = targetInfo.boxTag();
    // No unboxing required, pointers are pointers
    if (rawTag == 0) {
        return llvm_bitcast(innerTy.getPointerTo(1), box);
    } else {
        Value addr0 =
            llvm_addrspacecast(termTyAddr0, llvm_bitcast(termTy, box));
        Value ptrInt = llvm_ptrtoint(immedTy, addr0);
        Value tag = llvm_constant(immedTy, getIntegerAttr(rawTag));
        Value neg1 = llvm_constant(immedTy, getIntegerAttr(-1));
        Value untagged = llvm_and(ptrInt, llvm_xor(tag, neg1));
        Value untaggedAddr0 = llvm_inttoptr(innerTy.getPointerTo(), untagged);
        return llvm_addrspacecast(innerTy.getPointerTo(1), untaggedAddr0);
    }
}

Value OpConversionContext::decodeList(Value box) const {
    // TODO: Possibly use ptrmask intrinsic
    auto immedTy = getOpaqueImmediateType();
    auto termTy = getOpaqueTermType();
    auto termTyAddr0 = getOpaqueTermTypeAddr0();
    auto consTy = targetInfo.getConsType();
    Value addr0 = llvm_addrspacecast(termTyAddr0, llvm_bitcast(termTy, box));
    Value ptrInt = llvm_ptrtoint(immedTy, addr0);
    Value mask = llvm_constant(immedTy, getIntegerAttr(targetInfo.listMask()));
    Value neg1 = llvm_constant(immedTy, getIntegerAttr(-1));
    Value untagged = llvm_and(ptrInt, llvm_xor(mask, neg1));
    Value untaggedAddr0 = llvm_inttoptr(consTy.getPointerTo(), untagged);
    return llvm_addrspacecast(consTy.getPointerTo(1), untaggedAddr0);
}

Value OpConversionContext::decodeImmediate(Value val) const {
    auto immedTy = getOpaqueImmediateType();
    auto termTy = getOpaqueTermType();
    auto termTyAddr0 = getOpaqueTermTypeAddr0();
    auto maskInfo = targetInfo.immediateMask();

    Value addr0 = llvm_addrspacecast(termTyAddr0, llvm_bitcast(termTy, val));
    Value ptrInt = llvm_ptrtoint(immedTy, addr0);
    Value mask = llvm_constant(immedTy, getIntegerAttr(maskInfo.mask));
    Value masked = llvm_and(ptrInt, mask);
    if (maskInfo.requiresShift()) {
        Value shift = llvm_constant(immedTy, getIntegerAttr(maskInfo.shift));
        return llvm_shr(masked, shift);
    } else {
        return masked;
    }
}
}  // namespace eir
}  // namespace lumen
