#include "lumen/EIR/Conversion/TargetPlatformBuilder.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

using ::llvm::ArrayRef;
using ::llvm::SmallVector;
using ::llvm::StringRef;
using ::mlir::Identifier;
using ::mlir::Location;
using ::mlir::ModuleOp;
using ::mlir::NamedAttribute;
using ::mlir::Operation;
using ::mlir::Type;
using ::mlir::Value;
using ::mlir::edsc::OperationBuilder;
using ::mlir::edsc::ValueBuilder;
using ::mlir::LLVM::LLVMArrayType;
using ::mlir::LLVM::LLVMPointerType;
using ::mlir::LLVM::LLVMStructType;
using ::mlir::LLVM::LLVMTokenType;
using ::mlir::LLVM::LLVMVoidType;

namespace LLVM = ::mlir::LLVM;

using llvm_call = ValueBuilder<LLVM::CallOp>;
using llvm_and = ValueBuilder<LLVM::AndOp>;
using llvm_or = ValueBuilder<LLVM::OrOp>;
using llvm_xor = ValueBuilder<LLVM::XOrOp>;
using llvm_shr = ValueBuilder<LLVM::LShrOp>;
using llvm_bitcast = ValueBuilder<LLVM::BitcastOp>;
using llvm_addrspacecast = ValueBuilder<LLVM::AddrSpaceCastOp>;
using llvm_constant = ValueBuilder<LLVM::ConstantOp>;
using llvm_ptrtoint = ValueBuilder<LLVM::PtrToIntOp>;
using llvm_inttoptr = ValueBuilder<LLVM::IntToPtrOp>;

using namespace ::lumen;
using namespace ::lumen::eir;

Type TargetPlatformBuilder::getTokenType() const {
    return LLVMTokenType::get(context);
}
Type TargetPlatformBuilder::getVoidType() const {
    return LLVMVoidType::get(context);
}
Type TargetPlatformBuilder::getUsizeType() const {
    return getIntegerType(platform.getEncoder().getPointerWidth(),
                          /*signed=*/false);
}
Type TargetPlatformBuilder::getOpaqueTermType() const {
    return getPointerType(getUsizeType(), 1);
}
Type TargetPlatformBuilder::getOpaqueTermTypeAddr0() const {
    return getPointerType(getUsizeType());
}
Type TargetPlatformBuilder::getPointerType(Type pointeeType,
                                           unsigned addrspace) const {
    return LLVMPointerType::get(pointeeType, addrspace);
}
Type TargetPlatformBuilder::getConsType() const {
    auto consTy =
        LLVMStructType::getIdentified(getContext(), StringRef("cons"));
    if (consTy.getBody().size() == 0) {
        Type termTy = getOpaqueTermType();
        consTy.setBody({termTy, termTy}, /*packed*/ false);
    }
    return consTy;
}
Type TargetPlatformBuilder::getErlangTupleType(unsigned arity) const {
    const char *fmt = "tuple%d";
    int bufferSize = std::snprintf(nullptr, 0, fmt, arity);
    std::vector<char> buffer(bufferSize + 1);
    int strSize = std::snprintf(buffer.data(), buffer.size(), fmt, arity);
    StringRef typeName(buffer.data(), strSize);

    auto tupleTy = LLVMStructType::getIdentified(getContext(), typeName);
    if (tupleTy.getBody().size() != 0) return tupleTy;

    auto termTy = getOpaqueTermType();
    if (arity == 0) {
        tupleTy.setBody({termTy}, /*packed=*/false);
        return tupleTy;
    }

    auto usizeTy = getUsizeType();
    SmallVector<Type, 2> fieldTypes;
    fieldTypes.reserve(1 + arity);
    fieldTypes.push_back(usizeTy);
    for (auto i = 0; i < arity; i++) {
        fieldTypes.push_back(usizeTy);
    }
    tupleTy.setBody(fieldTypes, /*packed=*/false);
    return tupleTy;
}
Type TargetPlatformBuilder::getFloatType() const {
    TargetPlatformEncoder &encoder = platform.getEncoder();
    if (encoder.supportsNanboxing()) return getF64Type();

    auto floatTy =
        LLVMStructType::getIdentified(getContext(), StringRef("float"));
    if (floatTy.getBody().size() == 0) {
        Type f64Ty = getF64Type();
        Type usizeTy = getUsizeType();
        floatTy.setBody({usizeTy, f64Ty}, /*packed=*/false);
    }
    return floatTy;
}
Type TargetPlatformBuilder::getBigIntType() const {
    auto bigIntTy =
        LLVMStructType::getIdentified(getContext(), StringRef("bigint"));
    if (bigIntTy.getBody().size() == 0) {
        Type usizeTy = getUsizeType();
        bigIntTy.setBody({usizeTy}, /*packed*/ false);
    }
    return bigIntTy;
}
Type TargetPlatformBuilder::getBinaryType() const {
    auto binaryTy =
        LLVMStructType::getIdentified(getContext(), StringRef("binary"));
    if (binaryTy.getBody().size() == 0) {
        Type usizeTy = getUsizeType();
        Type i8PtrTy = getPointerType(getI8Type());
        binaryTy.setBody({usizeTy, usizeTy, i8PtrTy}, /*packed*/ false);
    }
    return binaryTy;
}
Type TargetPlatformBuilder::getBinaryBuilderType() const {
    return getPointerType(
        LLVMStructType::getOpaque(getContext(), StringRef("binary.builder")));
}
Type TargetPlatformBuilder::getBinaryPushResultType() const {
    auto pushResultTy =
        LLVMStructType::getIdentified(getContext(), StringRef("binary.pushed"));
    if (pushResultTy.getBody().size() == 0) {
        Type termTy = getOpaqueTermType();
        pushResultTy.setBody({termTy, termTy}, /*packed*/ false);
    }
    return pushResultTy;
}
Type TargetPlatformBuilder::getMatchResultType() const {
    auto matchResultTy =
        LLVMStructType::getIdentified(getContext(), StringRef("match.result"));
    if (matchResultTy.getBody().size() == 0) {
        Type termTy = getOpaqueTermType();
        Type i1Ty = getI1Type();
        matchResultTy.setBody({termTy, termTy, i1Ty}, /*packed*/ false);
    }
    return matchResultTy;
}
Type TargetPlatformBuilder::getTraceRefType() const {
    return getPointerType(
        LLVMStructType::getOpaque(getContext(), StringRef("trace")));
}
Type TargetPlatformBuilder::getRecvContextType() const {
    return getPointerType(
        LLVMStructType::getOpaque(getContext(), StringRef("recv.context")));
}
Type TargetPlatformBuilder::getClosureDefinitionType() const {
    // [i8 x 16]
    auto uniqueTy = LLVMArrayType::get(getI8Type(), 16);
    // struct { u32 tag, usize index_or_function_atom, [i8 x 16] unique, i32
    // oldUnique }
    auto defTy = LLVMStructType::getIdentified(getContext(),
                                               StringRef("closure.definition"));
    if (defTy.getBody().size() == 0) {
        Type i32Ty = getI32Type();
        Type usizeTy = getUsizeType();
        defTy.setBody({i32Ty, usizeTy, uniqueTy, i32Ty}, /*packed*/ false);
    }
    return defTy;
}
/*
#[repr(C)]
pub struct Closure {
    header: Header<Closure>,
    module: Atom,
    arity: u32,
    definition: Definition,
    code: Option<NonNull<*const c_void>>,
    env: [Term],
}
*/
Type TargetPlatformBuilder::getClosureType(unsigned size) const {
    // Name the type based on the arity of the env, makes IR more readable
    const char *fmt = "closure%d";
    int bufferSize = std::snprintf(nullptr, 0, fmt, size);
    std::vector<char> buffer(bufferSize + 1);
    int strSize = std::snprintf(buffer.data(), buffer.size(), fmt, size);
    StringRef typeName(buffer.data(), strSize);

    auto closureTy = LLVMStructType::getIdentified(context, typeName);
    if (closureTy.getBody().size() != 0) return closureTy;

    // Construct type of the fields
    auto usizeTy = getUsizeType();
    auto int32Ty = getI32Type();
    auto defTy = getClosureDefinitionType();
    auto voidTy = getVoidType();
    auto voidFnPtrTy = getPointerType(LLVMFunctionType::get(voidTy, {}, false));
    auto termTy = getOpaqueTermType();
    auto envTy = LLVMArrayType::get(termTy, size);
    ArrayRef<Type> fields{usizeTy, usizeTy, int32Ty, defTy, voidFnPtrTy, envTy};

    closureTy.setBody(fields, /*packed=*/false);
    return closureTy;
}
Type TargetPlatformBuilder::getExceptionType() const {
    auto exceptionTy = LLVMStructType::getIdentified(
        getContext(), StringRef("lumen.exception"));
    if (exceptionTy.getBody().size() == 0) {
        Type i8PtrTy = getPointerType(getI8Type());
        Type i32Ty = getI32Type();
        exceptionTy.setBody({i8PtrTy, i32Ty}, /*packed*/ false);
    }
    return exceptionTy;
}
Type TargetPlatformBuilder::getErrorType() const {
    auto erlangErrorTy = LLVMStructType::getIdentified(
        getContext(), StringRef("erlang.exception"));
    if (erlangErrorTy.getBody().size() == 0) {
        Type usizeTy = getUsizeType();
        Type termTy = getOpaqueTermType();
        Type usizePtrTy = getPointerType(usizeTy);
        Type i8PtrTy = getPointerType(getI8Type());
        erlangErrorTy.setBody({usizeTy, termTy, termTy, usizePtrTy, i8PtrTy},
                              /*packed*/ false);
    }
    return erlangErrorTy;
}

IntegerAttr getU32Attr(int32_t i) const {
    return getIntegerAttr(getI32Type(), APInt(32, i, /*signed=*/false));
}

Operation *TargetPlatformBuilder::getOrInsertFunction(
    ModuleOp mod, StringRef symbol, Type resultTy, ArrayRef<Type> argTypes,
    ArrayRef<NamedAttribute> attrs) const {
    Operation *funcOp = SymbolTable::lookupNearestSymbolFrom(mod, symbol);
    if (funcOp) return funcOp;

    // Create a function declaration for the symbol
    Type fnResultTy = resultTy ? resultTy : LLVMVoidType::get(context);
    auto fnTy = LLVMFunctionType::get(fnResultTy, argTypes, /*isVarArg=*/false);

    auto ip = saveInsertionPoint();
    setInsertionPointToStart(mod.body());
    auto op = create<LLVM::LLVMFuncOp>(mod.getLoc(), symbol, fnTy);
    for (auto attr : attrs) {
        op->setAttr(std::get<Identifier>(attr), std::get<Attribute>(attr));
    }
    return op;
}

Value TargetPlatformBuilder::buildMalloc(ModuleOp mod, Type ty,
                                         unsigned allocTy, Value arity) const {
    Type i8Ty = getI8Type();
    Type i32Ty = getI32Type();
    Type i8PtrTy = getPointerType(i8Ty, 1);
    Type ptrTy = getPointerType(ty, 1);
    Type usizeTy = getUsizeType();

    StringRef symbolName("__lumen_builtin_malloc");
    auto callee =
        getOrInsertFunction(mod, symbolName, i8PtrTy, {i32Ty, usizeTy})
            .cast<LLVM::LLVMFuncOp>();
    auto allocTyConst = llvm_constant(i32Ty, getU32Attr(allocTy));
    auto calleeSymbol = getSymbolRefAttr(symbolName);
    Value result = llvm_call(callee, ArrayRef<Value>{allocTyConst, arity});
    return llvm_bitcast(ptrTy, result);
}

Value TargetPlatformBuilder::encodeList(Value cons, bool isLiteral) const {
    // TODO: Possibly use ptrmask intrinsic
    TargetPlatformEncoder encoder = platform.getEncoder();
    Type usizeTy = getUsizeType();
    Type termTy = getOpaqueTermType();
    Type termTyAddr0 = getOpaqueTermTypeAddr0();
    Value addr0 = llvm_addrspacecast(termTyAddr0, llvm_bitcast(termTy, cons));
    Value ptrInt = llvm_ptrtoint(immedTy, addr0);
    Value tag;
    if (isLiteral) {
        Value listTag =
            llvm_constant(usizeTy, getIntegerAttr(encoder.getListTag()));
        Value literalTag =
            llvm_constant(usizeTy, getIntegerAttr(encoder.getLiteralTag()));
        tag = llvm_or(listTag, literalTag);
    } else {
        tag = llvm_constant(usizeTy, getIntegerAttr(encoder.getListTag()));
    }
    Value taggedAddr0 = llvm_inttoptr(termTyAddr0, llvm_or(ptrInt, tag));
    return llvm_addrspacecast(termTy, taggedAddr0);
}

Value TargetPlatformBuilder::encodeBox(Value val) const {
    // TODO: Possibly use ptrmask intrinsic
    TargetPlatformEncoder encoder = platform.getEncoder();
    auto rawTag = encoder.getBoxTag();
    auto termTy = getOpaqueTermType();
    // No boxing required, pointers are pointers
    if (rawTag == 0) {
        // No boxing required, pointers are pointers,
        // we should be operating on an addrspace(1) pointer
        // here, so all we need is a bitcast to term type.
        return llvm_bitcast(termTy, val);
    } else {
        auto usizeTy = getUsizeType();
        auto termTyAddr0 = getOpaqueTermTypeAddr0();
        // We need to tag the pointer, so that means casting to addrspace(0),
        // bitcasting to term type, then ptrtoint, then back again at the end
        Value addr0 =
            llvm_addrspacecast(termTyAddr0, llvm_bitcast(termTy, val));
        Value ptrInt = llvm_ptrtoint(usizeTy, addr0);
        Value tag = llvm_constant(termTy, getIntegerAttr(rawTag));
        Value taggedAddr0 = llvm_inttoptr(termTyAddr0, llvm_or(ptrInt, tag));
        return llvm_addrspacecast(termTy, taggedAddr0);
    }
}

Value TargetPlatformBuilder::encodeLiteral(Value val) const {
    // TODO: Possibly use ptrmask intrinsic
    TargetPlatformEncoder encoder = platform.getEncoder();
    auto rawTag = encoder.getLiteralTag();
    auto usizeTy = getUsizeType();
    auto termTy = getOpaqueTermType();
    auto termTyAddr0 = getOpaqueTermTypeAddr0();
    Value addr0 = llvm_addrspacecast(termTyAddr0, llvm_bitcast(termTy, val));
    Value ptrInt = llvm_ptrtoint(usizeTy, addr0);
    Value tag = llvm_constant(termTy, getIntegerAttr(rawTag));
    Value taggedAddr0 = llvm_inttoptr(termTyAddr0, llvm_or(ptrInt, tag));
    return llvm_addrspacecast(termTy, taggedAddr0);
}

Value TargetPlatformBuilder::encodeImmediate(ModuleOp mod, Location loc,
                                             Type ty, Value val) const {
    auto tyInfo = cast<TermTypeInterface>(ty);
    auto usizeTy = getUsizeType();
    auto termTy = getOpaqueTermType();
    auto i32Ty = getI32Type();

    StringRef symbolName("__lumen_builtin_encode_immediate");
    auto callee = getOrInsertFunction(mod, symbolName, termTy, {i32Ty, usizeTy})
                      .cast<LLVM::LLVMFuncOp>();

    Value kind =
        llvm_constant(i32Ty, getI32Attr(tyInfo.getTypeKind().getValue()));
    return llvm_call(callee, ArrayRef<Value>{kind, val});
}

Value TargetPlatformBuilder::decodeBox(Type innerTy, Value box) const {
    // TODO: Possibly use ptrmask intrinsic
    TargetPlatformEncoder encoder = platform.getEncoder();
    auto usizeTy = getUsizeType();
    auto termTy = getOpaqueTermType();
    auto termTyAddr0 = getOpaqueTermTypeAddr0();
    auto boxTy = box.getType();
    assert(boxTy == termTy && "expected boxed pointer type");
    auto rawTag = encoder.getBoxTag();
    // No unboxing required, pointers are pointers
    if (rawTag == 0) {
        return llvm_bitcast(getPointerType(innerTy, 1), box);
    } else {
        Value addr0 =
            llvm_addrspacecast(termTyAddr0, llvm_bitcast(termTy, box));
        Value ptrInt = llvm_ptrtoint(usizeTy, addr0);
        Value tag = llvm_constant(usizeTy, getIntegerAttr(rawTag));
        Value neg1 = llvm_constant(usizeTy, getIntegerAttr(-1));
        Value untagged = llvm_and(ptrInt, llvm_xor(tag, neg1));
        Value untaggedAddr0 = llvm_inttoptr(getPointerType(innerTy), untagged);
        return llvm_addrspacecast(getPointerType(innerTy, 1), untaggedAddr0);
    }
}

Value TargetPlatformBuilder::decodeList(Value box) const {
    // TODO: Possibly use ptrmask intrinsic
    TargetPlatformEncoder encoder = platform.getEncoder();
    auto usizeTy = getUsizeType();
    auto termTy = getOpaqueTermType();
    auto termTyAddr0 = getOpaqueTermTypeAddr0();
    auto consTy = getConsType();
    Value addr0 = llvm_addrspacecast(termTyAddr0, llvm_bitcast(termTy, box));
    Value ptrInt = llvm_ptrtoint(usizeTy, addr0);
    Value mask = llvm_constant(usizeTy, getIntegerAttr(encoder.getListMask()));
    Value neg1 = llvm_constant(usizeTy, getIntegerAttr(-1));
    Value untagged = llvm_and(ptrInt, llvm_xor(mask, neg1));
    Value untaggedAddr0 = llvm_inttoptr(getPointerType(consTy), untagged);
    return llvm_addrspacecast(getPointerType(consTy, 1), untaggedAddr0);
}

Value TargetPlatformBuilder::decodeImmediate(Value val) const {
    TargetPlatformEncoder encoder = platform.getEncoder();
    auto usizeTy = getUsizeType();
    auto termTy = getOpaqueTermType();
    auto termTyAddr0 = getOpaqueTermTypeAddr0();
    auto maskInfo = encoder.getImmediateMask();

    Value addr0 = llvm_addrspacecast(termTyAddr0, llvm_bitcast(termTy, val));
    Value ptrInt = llvm_ptrtoint(usizeTy, addr0);
    Value mask = llvm_constant(usizeTy, getIntegerAttr(maskInfo.mask));
    Value masked = llvm_and(ptrInt, mask);
    if (maskInfo.requiresShift()) {
        Value shift = llvm_constant(usizeTy, getIntegerAttr(maskInfo.shift));
        return llvm_shr(masked, shift);
    } else {
        return masked;
    }
}
