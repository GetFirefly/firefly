#include "CIR/Ops.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/SHA1.h"
#include <algorithm>
#include <functional>

#include "PassDetail.h"

using namespace mlir;
using namespace mlir::cir;

#define PASS_NAME "convert-cir-to-llvm"

//===----------------------------------------------------------------------===//
// Pattern Rewrites
//===----------------------------------------------------------------------===//
#include "CIR/Patterns.cpp.inc"

//===----------------------------------------------------------------------===//
// CIRTypeConverter
//
// This type converter builds on top of the LLVMTypeConverter class to provide
// additional type-oriented helpers for CIR. It is responsible for translating
// CIR types to their low-level LLVM representation.
//===----------------------------------------------------------------------===//
namespace {
class CIRTypeConverter : public LLVMTypeConverter {
public:
  using LLVMTypeConverter::convertType;
  using LLVMTypeConverter::getContext;

  CIRTypeConverter(MLIRContext *ctx, bool enableNanboxing, bool isMachO,
                   const LowerToLLVMOptions &options,
                   const DataLayoutAnalysis *analysis = nullptr)
      : LLVMTypeConverter(ctx, options, analysis), useMachOMangling(isMachO),
        enableNanboxing(enableNanboxing) {
    addConversion([&](CIRNoneType) { return getIsizeType(); });
    addConversion([&](CIROpaqueTermType) { return getTermType(); });
    addConversion([&](CIRNumberType) { return getTermType(); });
    addConversion([&](CIRIntegerType) { return getTermType(); });
    addConversion([&](CIRFloatType) { return getFloatType(); });
    addConversion([&](CIRAtomType) { return getIsizeType(); });
    addConversion([&](CIRBoolType) { return getIsizeType(); });
    addConversion([&](CIRIsizeType) { return getIsizeType(); });
    addConversion([&](CIRBigIntType) { return getBigIntType(); });
    addConversion([&](CIRNilType) { return getIsizeType(); });
    addConversion([&](CIRConsType) { return getConsType(); });
    addConversion([&](CIRMapType) { return getMapType(); });
    addConversion([&](CIRBitsType) { return getBitstringType(); });
    addConversion([&](CIRHeapbinType) { return getBitstringType(); });
    addConversion([&](CIRPidType) { return getTermType(); });
    addConversion([&](CIRPortType) { return getTermType(); });
    addConversion([&](CIRReferenceType) { return getTermType(); });
    addConversion([&](CIRBoxType type) { return convertBoxType(type); });
    addConversion([&](CIRFunType type) { return convertFunType(type); });
    addConversion([&](CIRExceptionType) { return getExceptionType(); });
    addConversion([&](CIRTraceType) { return getTraceType(); });
    addConversion([&](CIRRecvContextType) { return getRecvContextType(); });
    addConversion([&](PtrType type) {
      return LLVM::LLVMPointerType::get(convertType(type.getElementType()));
    });
    // The MLIR tuple type has no lowering out of the box, so we handle it
    addConversion([&](TupleType type) { return convertTupleType(type); });
  }

  bool isMachO() { return useMachOMangling; }

  bool isNanboxingEnabled() {
    return enableNanboxing && getPointerBitwidth() == 64;
  }

  Type getVoidType() { return LLVM::LLVMVoidType::get(&getContext()); }

  Type getI8Type() { return IntegerType::get(&getContext(), 8); }

  // The following get*Type functions are all used to get the LLVM
  // representation of either a built-in type or a CIR type, _not_ the named
  // type.

  // Currently we use this in some places for term-sized values, we should
  // prefer to always use getTermType in those cases instead, if you see those
  // occurring, change them.
  Type getIsizeType() {
    return IntegerType::get(&getContext(), getPointerBitwidth());
  }

  // NOTE: This is likely to change, so do not depend on this representation.
  // To get stackmaps working for garbage collection, we're likely going to
  // switch to a pointer type in a non-zero address space, and when that
  // happens, we can't have terms somtimes represented as integers and sometimes
  // as pointers, or it will cause gc roots to be missed.
  Type getTermType() { return getIsizeType(); }

  // Floats are immediates on nanboxed platforms, boxed types everywhere else
  Type getFloatType() {
    auto f64ty = Float64Type::get(&getContext());
    if (isNanboxingEnabled()) {
      return f64ty;
    } else {
      auto termTy = getTermType();
      return LLVM::LLVMStructType::getLiteral(&getContext(), {termTy, f64ty});
    }
  }

  // We've done the work to determine the actual structural layout of a BigInt
  // term value here, but we don't actually (currently) rely on this. If we do
  // end up wanting to rely on this, we should spend the time to confirm the
  // layout is fixed
  Type getBigIntType() {
    MLIRContext *context = &getContext();
    auto bigIntTy =
        LLVM::LLVMStructType::getIdentified(context, "erlang::BigInt");
    if (bigIntTy.isInitialized())
      return bigIntTy;
    // *BigIntTerm { header: isize, BigInt { val: BigUint { digits: Vec<isize> {
    // { *isize, isize }, isize } }, sign: i8, padding: [N x i8]}}
    auto isizeTy = getIsizeType();
    auto i8Ty = IntegerType::get(context, 8);
    auto bitwidth = getPointerBitwidth();
    auto paddingTy = LLVM::LLVMArrayType::get(i8Ty, (bitwidth % 8) - 1);
    auto isizePtrTy = LLVM::LLVMPointerType::get(getIsizeType());
    auto digitsInnerTy =
        LLVM::LLVMStructType::getLiteral(context, {isizePtrTy, isizeTy});
    auto digitsTy =
        LLVM::LLVMStructType::getLiteral(context, {digitsInnerTy, isizeTy});
    auto innerTy = LLVM::LLVMStructType::getLiteral(
        &getContext(), {digitsTy, i8Ty, paddingTy});
    assert(succeeded(bigIntTy.setBody({isizeTy, innerTy}, /*packed=*/false)) &&
           "failed to set body of bigint struct!");
    return bigIntTy;
  }

  // This layout matches what our runtime produces/expects, and we rely on this
  // to optimize operations on them.
  Type getConsType() {
    MLIRContext *context = &getContext();
    auto termTy = getTermType();
    return LLVM::LLVMStructType::getLiteral(context, {termTy, termTy});
  }

  // This layout matches what our runtime produces/expects, and we rely on this
  // to optimize operations on them
  Type getTupleType(unsigned arity) {
    auto termTy = getTermType();
    auto dataTy = LLVM::LLVMArrayType::get(termTy, arity);
    return LLVM::LLVMStructType::getLiteral(&getContext(), {termTy, dataTy});
  }

  // This layout is intentionally incomplete, as we don't control the layout of
  // our map implementation, and currently all operations on them are done via
  // the runtime only
  Type getMapType() {
    MLIRContext *context = &getContext();
    auto mapTy = LLVM::LLVMStructType::getIdentified(context, "erlang::Map");
    if (mapTy.isInitialized())
      return mapTy;
    // *MapTerm { header: isize, internal: opaque }
    auto termTy = getTermType();
    auto opaqueTy =
        LLVM::LLVMStructType::getOpaque("hashbrown::HashMap", context);
    assert(succeeded(mapTy.setBody({termTy, opaqueTy}, /*packed=*/false)) &&
           "failed to set body of map struct!");
    return mapTy;
  }

  // This layout matches what our runtime produces/expects for heapbin/literal
  // binaries, and we rely on it. For other binary types, namely procbin, all
  // operations are via library calls anyway, so it doesn't matter.
  Type getBitstringType() {
    MLIRContext *context = &getContext();
    auto bitsTy = LLVM::LLVMStructType::getIdentified(context, "erlang::Bits");
    if (bitsTy.isInitialized())
      return bitsTy;
    // *Bits { header: isize, flags: isize, data: [? x i8]}
    auto termTy = getTermType();
    auto dataTy = LLVM::LLVMArrayType::get(IntegerType::get(context, 8), 0);
    assert(
        succeeded(bitsTy.setBody({termTy, termTy, dataTy}, /*packed=*/false)) &&
        "failed to set body of bitstring struct!");
    return bitsTy;
  }

  // This layout matches what our runtime produces/expects for closure values,
  // and we rely on it to optimize function calls.
  Type getClosureType() {
    MLIRContext *context = &getContext();
    auto closureTy =
        LLVM::LLVMStructType::getIdentified(context, "erlang::Fun");
    if (closureTy.isInitialized())
      return closureTy;

    auto isizeTy = getIsizeType();
    auto i32Ty = IntegerType::get(context, 32);
    auto headerTy = isizeTy;
    auto atomTy = getTermType();
    auto arityTy = i32Ty;
    auto bareFunTy = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(context), {}, /*vararg=*/false);
    auto funPtrTy = LLVM::LLVMPointerType::get(bareFunTy);
    auto envTy = LLVM::LLVMArrayType::get(getTermType(), 1);
    auto defTagTy = i32Ty;
    auto uniqueTy = LLVM::LLVMArrayType::get(IntegerType::get(context, 8), 16);
    auto oldUniqueTy = i32Ty;
    auto defTy = LLVM::LLVMStructType::getNewIdentified(
        context, "erlang::Definition",
        {defTagTy, isizeTy, uniqueTy, oldUniqueTy});

    assert(succeeded(closureTy.setBody(
               {headerTy, atomTy, arityTy, defTy, funPtrTy, envTy},
               /*packed=*/false)) &&
           "failed to set body of closure struct!");
    return closureTy;
  }

  // This function returns the target-specific exception representation
  Type getExceptionType() {
    MLIRContext *context = &getContext();
    auto exceptionTy =
        LLVM::LLVMStructType::getIdentified(context, "erlang::Exception");
    if (exceptionTy.isInitialized())
      return exceptionTy;

    // Corresponds to ErlangException in liblumen_alloc
    // { header: isize, class: term, reason: term, trace: *mut Trace, fragment:
    // *const HeapFragment }
    Type isizeTy = getIsizeType();
    Type termTy = getTermType();
    Type isizePtrTy = LLVM::LLVMPointerType::get(isizeTy);
    assert(succeeded(exceptionTy.setBody(
               {isizeTy, termTy, termTy, isizePtrTy, isizePtrTy},
               /*packed=*/false)) &&
           "failed to set body of exception struct!");
    return exceptionTy;
  }

  // This function returns the type of handle used for the raw exception trace
  // reprsentation
  Type getTraceType() {
    MLIRContext *context = &getContext();
    auto traceTy = LLVM::LLVMStructType::getOpaque("erlang::Trace", context);
    return LLVM::LLVMPointerType::get(traceTy);
  }

  // Corresponds to Message in liblumen_alloc
  Type getMessageType() {
    MLIRContext *context = &getContext();
    auto messageTy =
        LLVM::LLVMStructType::getIdentified(context, "erlang::Message");
    if (messageTy.isInitialized())
      return messageTy;

    Type isizeTy = getIsizeType();
    Type i32Ty = IntegerType::get(context, 32);
    Type termTy = getTermType();
    auto linkTy = LLVM::LLVMStructType::getLiteral(context, {isizeTy, isizeTy});
    auto msgDataTy = LLVM::LLVMStructType::getLiteral(context, {i32Ty, termTy});
    assert(succeeded(messageTy.setBody({linkTy, msgDataTy}, /*packed=*/false)));
    return messageTy;
  }

  // Corresponds to ReceiveContext in lumen_rt_minimal
  Type getRecvContextType() {
    MLIRContext *context = &getContext();
    auto recvCtxTy =
        LLVM::LLVMStructType::getIdentified(context, "erlang::ReceiveContext");
    if (recvCtxTy.isInitialized())
      return recvCtxTy;

    auto i64Ty = IntegerType::get(context, 64);
    auto termTy = getTermType();
    auto msgPtrTy = LLVM::LLVMPointerType::get(getMessageType());
    assert(succeeded(
        recvCtxTy.setBody({i64Ty, termTy, msgPtrTy}, /*packed=*/false)));
    return recvCtxTy;
  }

  // Corresponds to DispatchEntry in liblumen_alloc
  Type getDispatchEntryType() {
    MLIRContext *context = &getContext();
    auto dispatchEntryTy =
        LLVM::LLVMStructType::getIdentified(context, "erlang::DispatchEntry");
    if (dispatchEntryTy.isInitialized())
      return dispatchEntryTy;

    auto i8Ty = getI8Type();
    auto i8PtrTy = LLVM::LLVMPointerType::get(i8Ty);
    auto opaqueFnTy =
        LLVM::LLVMFunctionType::get(getVoidType(), ArrayRef<Type>{});
    auto opaqueFnPtrTy = LLVM::LLVMPointerType::get(opaqueFnTy);
    assert(succeeded(dispatchEntryTy.setBody(
        {i8PtrTy, i8PtrTy, i8Ty, opaqueFnPtrTy}, /*packed=*/false)));
    return dispatchEntryTy;
  }

  // This function lowers a box type to its LLVM pointer equivalent
  Type convertBoxType(CIRBoxType ty) {
    Type pointee = convertType(ty.getElementType());
    return LLVM::LLVMPointerType::get(pointee);
  }

  // This function lowers a fun type to an appropriate function signature
  // depending on the nature of the fun.
  //
  // For thin funs (i.e. captures), no closure env is needed, so the callee type
  // can be used directly.
  //
  // For closures, the associated env is required, so the callee type must be
  // extended with an extra argument, the pointer to the closure metadata.
  //
  // It is expected that when lowering the actual closure, that the signature of
  // the corresponding function will match the signature produced by this
  // function.
  Type convertFunType(CIRFunType ty) {
    auto calleeTy = ty.getCalleeType();
    if (ty.isThin())
      return convertType(calleeTy);

    auto closurePtrTy = LLVM::LLVMPointerType::get(getClosureType());
    return convertType(
        calleeTy.getWithArgsAndResults({0}, {closurePtrTy}, {}, {}));
  }

  // This function lowers a tuple type to its LLVM equivalent
  Type convertTupleType(TupleType ty) {
    auto arity = ty.size();
    // If the shape is unknown, then we'll have an arity of zero,
    // not that we handle it specially here, but worth noting
    return getTupleType(arity);
  }

private:
  bool useMachOMangling;
  bool enableNanboxing;
};
} // namespace

// ==----------------------------------------------------------------------===//
// CIRConversionPattern
//
// This class provides common functionality for conversions of CIR ops to LLVM.
//===----------------------------------------------------------------------===//
namespace {
class CIRConversionPattern : public ConversionPattern {
public:
  CIRConversionPattern(StringRef rootOpName, MLIRContext *context,
                       CIRTypeConverter &typeConverter,
                       PatternBenefit benefit = 1)
      : ConversionPattern(typeConverter, rootOpName, benefit, context),
        encoding({typeConverter.getPointerBitwidth(),
                  typeConverter.isNanboxingEnabled()}){};

  using Pattern::getContext;

protected:
  // This is necessary in order to access our type converter
  CIRTypeConverter *getTypeConverter() const {
    return static_cast<CIRTypeConverter *>(
        ConversionPattern::getTypeConverter());
  }

  // The following functions Provide target-specific term encoding details

  unsigned getPointerBitwidth() const { return encoding.pointerWidth; }
  bool isNanboxingEnabled() const { return encoding.supportsNanboxing; }
  bool isMachO() const { return getTypeConverter()->isMachO(); }
  const lumen::Encoding &termEncoding() const { return encoding; }

  // We re-export type conversion functionality commonly used

  Type convertType(Type ty) const {
    return getTypeConverter()->convertType(ty);
  }
  Type getIntPtrType(unsigned addressSpace = 0) const {
    return getIntType(getPointerBitwidth());
  }
  Type getVoidType() const { return LLVM::LLVMVoidType::get(getContext()); }
  Type getIntType(unsigned bitwidth) const {
    return IntegerType::get(getContext(), bitwidth);
  }
  Type getI1Type() const { return getIntType(1); }
  Type getI8Type() const { return getIntType(8); }
  Type getI32Type() const { return getIntType(32); }
  Type getI64Type() const { return getIntType(64); }
  Type getIsizeType() const { return getTypeConverter()->getIsizeType(); }
  Type getTermType() const { return getTypeConverter()->getTermType(); }
  Type getFloatType() const { return getTypeConverter()->getFloatType(); }
  Type getBigIntType() const { return getTypeConverter()->getBigIntType(); }
  Type getConsType() const { return getTypeConverter()->getConsType(); }
  Type getTupleType(unsigned arity) const {
    return getTypeConverter()->getTupleType(arity);
  }
  Type getMapType() const { return getTypeConverter()->getMapType(); }
  Type getBitstringType() const {
    return getTypeConverter()->getBitstringType();
  }
  Type getClosureType() const { return getTypeConverter()->getClosureType(); }
  Type getExceptionType() const {
    return getTypeConverter()->getExceptionType();
  }
  Type getTraceType() const { return getTypeConverter()->getTraceType(); }
  Type getRecvContextType() const {
    return getTypeConverter()->getRecvContextType();
  }
  Type getMessageType() const { return getTypeConverter()->getMessageType(); }
  Type getDispatchEntryType() const {
    return getTypeConverter()->getDispatchEntryType();
  }

  // The following are helpers intended to handle common builder use cases

  // This function builds an LLVM::ConstantOp with an isize value, but an
  // arbitrary result type. This is commonly used for index operations which
  // vary between i32/i64
  static Value createIndexAttrConstant(OpBuilder &builder, Location loc,
                                       Type resultType, int64_t value) {
    return builder.create<LLVM::ConstantOp>(
        loc, resultType, builder.getIntegerAttr(builder.getIndexType(), value));
  }

  // This function builds an LLVM::ConstantOp with an i1 value and result type
  Value createBoolConstant(OpBuilder &builder, Location loc, bool value) const {
    auto i1Ty = builder.getI1Type();
    return builder.create<LLVM::ConstantOp>(
        loc, i1Ty, builder.getIntegerAttr(i1Ty, value));
  }

  // This function builds an LLVM::ConstantOp with an isize value and result
  // type
  Value createIsizeConstant(OpBuilder &builder, Location loc,
                            uint64_t value) const {
    return createIndexAttrConstant(builder, loc, getIsizeType(), value);
  }

  // This function builds an LLVM::ConstantOp with a value of term type
  Value createTermConstant(OpBuilder &builder, Location loc,
                           uint64_t value) const {
    return createIndexAttrConstant(builder, loc, getTermType(), value);
  }

  // This function builds an LLVM::GlobalOp representing a constnat value of the
  // given type with the provided name.
  //
  // It defaults to internal linkage/non-thread-local, with an empty
  // initializer.
  //
  // NOTE: It is the callers responsibility to ensure a global with the given
  // name isn't already defined.
  LLVM::GlobalOp
  insertGlobalConstantOp(OpBuilder &builder, Location loc, std::string name,
                         Type ty,
                         LLVM::Linkage linkage = LLVM::Linkage::Internal,
                         LLVM::ThreadLocalMode tlsMode =
                             LLVM::ThreadLocalMode::NotThreadLocal) const {
    return builder.create<LLVM::GlobalOp>(loc, ty, /*isConstant=*/true, linkage,
                                          tlsMode, name, Attribute());
  }

  // This function is used to construct an LLVM constant for Erlang integer
  // terms
  Value createIntegerConstant(OpBuilder &builder, Location loc,
                              uint64_t value) const {
    return createIsizeConstant(builder, loc,
                               encodeImmediate(lumen::TermKind::Isize, value));
  }

  // This function is used to construct an LLVM constant for an Erlang float
  // term
  Value createFloatConstant(ConversionPatternRewriter &builder, Location loc,
                            APFloat value, ModuleOp &module) const {
    // When nanboxed, floats have no tag, but cannot be NaN, as NaN bits are
    // used for term tagging
    if (isNanboxingEnabled()) {
      assert(!value.isNaN() && "invalid floating point constant for target, "
                               "floats must not be NaN!");
      return createIntegerConstant(builder, loc,
                                   value.bitcastToAPInt().getZExtValue());
    }
    // On other targets, floats are boxed
    // As this is a constant, we generate a global to hold the header + value,
    // and encode a pointer to it as the value of the constant
    auto header = encodeHeader(lumen::TermKind::Float, 0);
    auto imm = value.convertToDouble();
    auto globalName = std::string("float_") + std::to_string(imm);
    auto headerConst = module.lookupSymbol<LLVM::GlobalOp>(globalName);
    auto floatTy = getFloatType();
    // If the constant hasn't yet been defined, define it and initialize it
    if (!headerConst) {
      auto f64ty = builder.getF64Type();
      PatternRewriter::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());
      headerConst = insertGlobalConstantOp(builder, loc, globalName, f64ty);

      auto &initRegion = headerConst.getInitializerRegion();
      builder.createBlock(&initRegion);

      Value headerTerm = createIsizeConstant(builder, loc, header);
      Value valueTerm = builder.create<LLVM::ConstantOp>(
          loc, f64ty, builder.getF64FloatAttr(imm));
      Value payload = builder.create<LLVM::UndefOp>(loc, floatTy);
      payload = builder.create<LLVM::InsertValueOp>(loc, payload, headerTerm,
                                                    builder.getI64ArrayAttr(0));
      payload = builder.create<LLVM::InsertValueOp>(loc, payload, valueTerm,
                                                    builder.getI64ArrayAttr(1));
      builder.create<LLVM::ReturnOp>(loc, payload);
    }

    // Box the constant address
    Value ptr = builder.create<LLVM::AddressOfOp>(loc, headerConst);
    return encodeLiteralPtr(builder, loc, ptr);
  }

  // This function is used to obtain an atom value corresponding to the given
  // StringRef
  //
  // For the boolean atoms, this is a constant value encoded as a term.
  //
  // For all other atoms, a string constant is defined in its own section, with
  // linkonce_odr linkage, intended to be gathered together by the linker into
  // an array of cstrings from which the global atom table will be initialized.
  //
  // A call to a special builtin that constructs an atom term from its value as
  // a cstring is used to obtain the result value returned by this function.
  Value createAtom(OpBuilder &builder, Location loc, StringRef name,
                   ModuleOp &module) const {
    if (name == "false")
      return createTermConstant(builder, loc,
                                encodeImmediate(lumen::TermKind::Atom, 0));
    else if (name == "true")
      return createTermConstant(builder, loc,
                                encodeImmediate(lumen::TermKind::Atom, 1));

    auto termTy = getTermType();
    auto cstrTy = LLVM::LLVMPointerType::get(builder.getI8Type());

    // Hash the atom to get a unique id based on the content
    auto ptr = createAtomStringGlobal(builder, loc, module, name);

    // Make sure we have a definition for the builtin that lets us obtain an
    // atom from its CStr repr
    Operation *callee = module.lookupSymbol("__lumen_builtin_atom_from_cstr");
    if (!callee) {
      auto calleeType =
          LLVM::LLVMFunctionType::get(termTy, ArrayRef<Type>{cstrTy});
      insertFunctionDeclaration(builder, loc, module,
                                "__lumen_builtin_atom_from_cstr", calleeType);
    }

    // Call the builtin with the cstr pointer to get the atom value as a term
    Operation *call = builder.create<LLVM::CallOp>(
        loc, TypeRange(termTy), "__lumen_builtin_atom_from_cstr",
        ValueRange(ptr));
    return call->getResult(0);
  }

  // This function constructs a global null-terminated string constant which
  // will be added to the global atom table
  Value createAtomStringGlobal(OpBuilder &builder, Location loc,
                               ModuleOp &module, StringRef value) const {
    llvm::SHA1 hasher;
    hasher.update(value);
    auto globalName = std::string("atom_") + llvm::toHex(hasher.result(), true);

    std::string sectionName;
    if (isMachO())
      sectionName = std::string("__TEXT,__atoms");
    else
      sectionName = std::string("__") + globalName;
    auto sectionAttr =
        builder.getNamedAttr("section", builder.getStringAttr(sectionName));
    return createCStringGlobal(builder, loc, module, globalName, value,
                               {sectionAttr});
  }

  // This function constructs a global null-terminated string constant with a
  // given name and value.
  //
  // The name is optional, as a name will be generated if not provided.
  // The value does not need to be null-terminated.
  //
  // The value returned is a pointer to the first byte of the string, i.e.
  // `*const u8`
  Value createCStringGlobal(OpBuilder &builder, Location loc, ModuleOp &module,
                            StringRef name, StringRef value,
                            ArrayRef<NamedAttribute> attrs) const {
    // Hash the atom to get a unique id based on the content
    std::string globalName;
    if (name.empty()) {
      llvm::SHA1 hasher;
      hasher.update(value);
      globalName = std::string("cstr_") + llvm::toHex(hasher.result(), true);
    } else {
      globalName = name.str();
    }
    auto data = value.str() += ((char)0);

    // Create the global if it doesn't exist
    auto strConst = module.lookupSymbol<LLVM::GlobalOp>(globalName);
    auto charsTy = LLVM::LLVMArrayType::get(builder.getI8Type(), data.size());
    auto cstrTy = LLVM::LLVMPointerType::get(builder.getI8Type());
    if (!strConst) {
      PatternRewriter::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());

      auto valueAttr = builder.getStringAttr(data);
      strConst = builder.create<LLVM::GlobalOp>(
          loc, charsTy, /*isConstant=*/true, LLVM::Linkage::LinkonceODR,
          LLVM::ThreadLocalMode::NotThreadLocal, globalName, valueAttr,
          /*alignment=*/0, /*addrspace=*/0, /*dso_local=*/false, attrs);
    }

    // Get a opaque cstr pointer to the constant we just created (or that
    // already exists)
    Value contentsPtr = builder.create<LLVM::AddressOfOp>(loc, strConst);
    return builder.create<LLVM::BitcastOp>(loc, cstrTy, contentsPtr);
  }

  // The following helpers are all oriented around low-level encoding/decoding
  // of Erlang terms. It is highly unsafe to call these without ensuring their
  // invariants are held. It is expected that conversions building on these
  // helpers are handling that.

  // This function encodes a pointer to a constant Erlang term as an opaque term
  // value.
  //
  // The caller must guarantee the following:
  //
  // * The given value is a valid pointer to a term header.
  // * The pointee term is a global constant. The pointee must not be on the
  // stack or process heap.
  //
  // NOTE: This function is _not_ valid for cons cells, use encodeListPtr for
  // that.
  Value encodeLiteralPtr(OpBuilder &builder, Location loc, Value value) const {
    // TODO: Possibly use ptrmask intrinsic
    auto termTy = getTermType();
    Value valueAsInt = builder.create<LLVM::PtrToIntOp>(loc, termTy, value);
    Value tag = createTermConstant(builder, loc, boxTag() | literalTag());
    return builder.create<LLVM::OrOp>(loc, valueAsInt, tag);
  }

  // This function encodes a pointer to an Erlang term as an opaque term value.
  //
  // The caller must guarantee the following:
  //
  // * The given value is a valid pointer to a term header.
  // * The pointee term is located on the process heap. It is not safe to store
  // boxed terms on the stack currently.
  //
  // NOTE: This function is not valid for cons cells, use encodeListPtr for
  // that.
  Value encodePtr(OpBuilder &builder, Location loc, Value value) const {
    // TODO: Possibly use ptrmask intrinsic
    auto rawTag = boxTag();
    auto termTy = getTermType();
    if (rawTag == 0) {
      // No tagging required, simply cast to term type
      return builder.create<LLVM::PtrToIntOp>(loc, termTy, value);
    }
    // Boxed pointers require tagging
    Value valueAsInt = builder.create<LLVM::PtrToIntOp>(loc, termTy, value);
    Value tag = createTermConstant(builder, loc, rawTag);
    return builder.create<LLVM::OrOp>(loc, valueAsInt, tag);
  }

  // This function encodes a pointer to an Erlang list (i.e. cons cell) as an
  // opaque term value. The pointee may also be literal, by default this is not
  // the case.
  //
  // The caller must guarantee the following:
  //
  // * The given value is a valid pointer to the head of a cons cell.
  // * If isLiteral=true, the term must be a global constant. When true, the
  // cell must not be on the stack or process heap.
  // * If isLiteral=false, the term must be located on the process heap. It is
  // not safe to store boxed terms on the stack currently.
  //
  // NOTE: This function is not valid for any terms other than cons cells.
  Value encodeListPtr(OpBuilder &builder, Location loc, Value value,
                      bool isLiteral = false) const {
    // TODO: Possibly use ptrmask intrinsic
    auto termTy = getTermType();
    auto tags = isLiteral ? (listTag() | literalTag()) : listTag();
    Value tag = createTermConstant(builder, loc, tags);
    Value valueAsInt = builder.create<LLVM::PtrToIntOp>(loc, termTy, value);
    return builder.create<LLVM::OrOp>(loc, valueAsInt, tag);
  }

  // This function encodes an immediate value as an opaque Erlang term.
  //
  // The caller must guarantee the following:
  //
  // * The given value is a valid Erlang immediate, sans term tag. For example,
  // when encoding an Erlang integer, the integer value must not exceed the
  // valid range for integers on the current target. Atom symbols must
  // correspond to their id in the symbol table, etc.
  // * The given type must match the value being encoded, e.g. if encoding an
  // integer value, the type must be CIRIsizeType. The type must be an immediate
  // type. This function will assert that this constraint is violated.
  //
  // NOTE: This function is only valid for immediates, it is not valid for any
  // other term.
  Value encodeImmediateValue(OpBuilder &builder, Location loc, Type ty,
                             Value value) const {
    auto imm = ty.cast<TermType>();
    auto kind = imm.getTermKind();
    auto mask = immediateMask();
    auto tag = createTermConstant(builder, loc, immediateTag(kind));
    if (mask.requiresShift()) {
      auto shift = createTermConstant(builder, loc, mask.shift);
      Value shifted = builder.create<LLVM::ShlOp>(loc, value, shift);
      return builder.create<LLVM::OrOp>(loc, shifted, tag);
    } else {
      return builder.create<LLVM::OrOp>(loc, value, tag);
    }
  }

  // This function is the natural opposite of
  // encodePtr/encodeListPtr/encodeLiteralPtr, i.e. it can decode a value
  // encoded by encodePtr, as a pointer to a value of `pointee` type.
  //
  // The caller must guarantee the following:
  //
  // * The given value is a box term
  // * The pointer produced by decoding the box is a valid pointer
  // * The pointee value can be dereferenced as a value of type `pointee`
  //
  // NOTE: This function strips all box tags, so code that cares about whether
  // the value is literal or not must perform that check separately.
  Value decodePtr(OpBuilder &builder, Location loc, Type pointee,
                  Value box) const {
    // TODO: Possibly use ptrmask intrinsic
    auto termTy = getTermType();
    auto ptrTy = LLVM::LLVMPointerType::get(pointee);
    // Strip the tag bits for all pointer types
    auto rawTag = boxTag() | literalTag() | listTag();
    // Need to untag the pointer first
    auto tag = createTermConstant(builder, loc, rawTag);
    auto neg1 = builder.create<LLVM::ConstantOp>(
        loc, termTy, builder.getIntegerAttr(getIsizeType(), -1));
    Value mask = builder.create<LLVM::XOrOp>(loc, tag, neg1);
    Value untagged = builder.create<LLVM::AndOp>(loc, box, mask);
    return builder.create<LLVM::IntToPtrOp>(loc, ptrTy, untagged);
  }

  // An alias for decodePtr with a type representing the cons cell layout.
  Value decodeListPtr(OpBuilder &builder, Location loc, Value box) const {
    return decodePtr(builder, loc, getConsType(), box);
  }

  // This function is the natural opposite of encodeImmediateValue, i.e. it can
  // decode a value encoded by encodeImmediateValue as a raw integer value. This
  // function does not bitcast to a concrete type, as in most cases integers are
  // all we need, but in the case of floats, it is expected that the caller will
  // perform a bitcast as needed.
  //
  // The caller must guarantee the following:
  //
  // * The given value is an immediate term
  // * The given value can be treated as a valid instance of the concrete type
  // it corresponds to, e.g. it must be the case that bitcasting to float is a
  // valid, canonical float.
  Value decodeImmediateValue(OpBuilder &builder, Location loc,
                             Value value) const {
    auto maskInfo = immediateMask();
    Value mask = createTermConstant(builder, loc, maskInfo.mask);
    Value masked = builder.create<LLVM::AndOp>(loc, value, mask);
    if (maskInfo.requiresShift()) {
      Value shift = createTermConstant(builder, loc, maskInfo.shift);
      return builder.create<LLVM::LShrOp>(loc, masked, shift);
    }
    return masked;
  }

  // This function returns an integer value representing only the tag
  // information for an immediate, i.e. it strips all non-tag bits from the
  // input value.
  //
  // The caller must guarantee the following:
  //
  // * The given value is an immediate term. If the term is not an immediate,
  // then the behavior of acting on the resulting tag is undefined.
  Value extractImmediateTag(OpBuilder &builder, Location loc, Value value,
                            bool stripLiteral = false) const {
    auto maskInfo = immediateMask();
    Value mask = createIsizeConstant(
        builder, loc,
        stripLiteral ? (maskInfo.mask | literalTag()) : maskInfo.mask);
    Value neg1 = createIsizeConstant(builder, loc, -1);
    Value tagMask = builder.create<LLVM::XOrOp>(loc, mask, neg1);
    return builder.create<LLVM::AndOp>(loc, value, tagMask);
  }

  // This function returns an integer value representing only the tag
  // information in a term header, i.e. it strips all non-tag bits from the
  // input value.
  //
  // The caller must guarantee the following:
  //
  // * The given value is a term header. If the value is not a term header, then
  // the behavior of acting on the resulting tag is undefined.
  Value extractHeaderTag(OpBuilder &builder, Location loc, Value value) const {
    auto maskInfo = headerMask();
    Value mask = createTermConstant(builder, loc, maskInfo.mask);
    Value neg1 = createIsizeConstant(builder, loc, -1);
    Value tagMask = builder.create<LLVM::XOrOp>(loc, mask, neg1);
    return builder.create<LLVM::AndOp>(loc, value, tagMask);
  }

  // This function is a low-level encoding primitive which calls into
  // liblumen_term to encode the given value as an immediate term. This is
  // intended for use in constants.
  uint64_t encodeImmediate(lumen::TermKind::Kind kind, uint64_t value) const {
    return lumen_encode_immediate(&encoding, kind, value);
  }

  // This function is a low-level encoding primitive which calls into
  // liblumen_term to encode the given value as a term header. This is intended
  // for use in constants.
  uint64_t encodeHeader(lumen::TermKind::Kind kind, uint64_t arity) const {
    return lumen_encode_header(&encoding, kind, arity);
  }

  // This function is a low-level enncoding primitive which returns the raw
  // integer representation of the tag bits for a boxed list (i.e. the bits set
  // on the pointer)
  uint64_t listTag() const { return lumen_list_tag(&encoding); }

  // This function is a low-level enncoding primitive which returns the raw
  // integer representation of the tag bits for a non-list, boxed term (i.e. the
  // bits set on the pointer)
  uint64_t boxTag() const { return lumen_box_tag(&encoding); }

  // This function returns a raw integer representation of the tag bits for
  // pointers to literals
  uint64_t literalTag() const { return lumen_literal_tag(&encoding); }

  // This function is a low-level enncoding primitive which returns the raw
  // integer representation of the tag bits for a term header of the given kind
  uint64_t headerTag(lumen::TermKind::Kind kind) const {
    return lumen_header_tag(&encoding, kind);
  }

  // This function is a low-level enncoding primitive which returns the raw
  // integer representation of the tag bits for an immediate term of the given
  // kind
  uint64_t immediateTag(lumen::TermKind::Kind kind) const {
    return lumen_immediate_tag(&encoding, kind);
  }

  // This function returns the mask info used for extracting data from a term
  // header
  lumen::MaskInfo headerMask() const { return lumen_header_mask(&encoding); }

  // This function returns the mask info used for extracting the value from an
  // encoded immediate term
  lumen::MaskInfo immediateMask() const {
    return lumen_immediate_mask(&encoding);
  }

  LLVM::LLVMFuncOp
  insertFunctionDeclaration(OpBuilder &builder, Location loc, ModuleOp module,
                            StringRef name, LLVM::LLVMFunctionType type) const {
    PatternRewriter::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToEnd(module.getBody());
    return builder.create<LLVM::LLVMFuncOp>(loc, name, type);
  }

  // This function inserts a reference to the thread-local global containing the
  // current process exception pointer
  LLVM::GlobalOp insertProcessExceptionThreadLocal(OpBuilder &builder,
                                                   Location loc,
                                                   ModuleOp module) const {
    PatternRewriter::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(module.getBody());
    auto exceptionTy = getExceptionType();
    auto ty = LLVM::LLVMPointerType::get(exceptionTy);
    auto linkage = LLVM::Linkage::External;
    auto tlsMode = LLVM::ThreadLocalMode::LocalExec;
    return builder.create<LLVM::GlobalOp>(
        loc, ty, /*isConstant=*/false, linkage, tlsMode,
        "__lumen_process_exception", Attribute());
  }

  // This function inserts a reference to the thread-local global containing the
  // current process signal value
  LLVM::GlobalOp insertProcessSignalThreadLocal(OpBuilder &builder,
                                                Location loc,
                                                ModuleOp module) const {
    PatternRewriter::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(module.getBody());
    auto ty = builder.getI8Type();
    auto linkage = LLVM::Linkage::External;
    auto tlsMode = LLVM::ThreadLocalMode::LocalExec;
    return builder.create<LLVM::GlobalOp>(
        loc, ty, /*isConstant=*/false, linkage, tlsMode,
        "__lumen_process_signal", Attribute());
  }

  // This function inserts a reference to the thread-local global containing the
  // current process reduction counter
  LLVM::GlobalOp insertReductionCountThreadLocal(OpBuilder &builder,
                                                 Location loc,
                                                 ModuleOp module) const {
    PatternRewriter::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(module.getBody());
    auto ty = builder.getI32Type();
    auto linkage = LLVM::Linkage::External;
    auto tlsMode = LLVM::ThreadLocalMode::LocalExec;
    return builder.create<LLVM::GlobalOp>(
        loc, ty, /*isConstant=*/false, linkage, tlsMode,
        "__lumen_process_reductions", Attribute());
  }

private:
  lumen::Encoding encoding;
};
} // namespace

// ==----------------------------------------------------------------------===//
// ConvertCIROpToLLVMPattern
//
// This class implements the boilerplate for CIR conversion patterns, and
// provides a typed API for implementing matchAndRewrite. Implementations only
// need to override the typed matchAndRewrite function.
//===----------------------------------------------------------------------===//
namespace {
template <typename SourceOp>
class ConvertCIROpToLLVMPattern : public CIRConversionPattern {
public:
  using OpAdaptor = typename SourceOp::Adaptor;

  explicit ConvertCIROpToLLVMPattern(CIRTypeConverter &typeConverter,
                                     PatternBenefit benefit = 1)
      : CIRConversionPattern(SourceOp::getOperationName(),
                             &typeConverter.getContext(), typeConverter,
                             benefit) {}

  // Wrappers around RewritePattern methods that pass the derived op type
  void rewrite(Operation *op, ArrayRef<Value> operands,
               ConversionPatternRewriter &rewriter) const final {

    rewrite(cast<SourceOp>(op), OpAdaptor(operands, op->getAttrDictionary()),
            rewriter);
  }
  LogicalResult match(Operation *op) const final {
    return match(cast<SourceOp>(op));
  }
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    return matchAndRewrite(cast<SourceOp>(op),
                           OpAdaptor(operands, op->getAttrDictionary()),
                           rewriter);
  }

  // Rewrite/Match methods that operate on the SourceOp type.
  // These must be overridden by the derived class
  virtual LogicalResult match(SourceOp op) const {
    llvm_unreachable("must override match or matchAndRewrite");
  }
  virtual void rewrite(SourceOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const {
    llvm_unreachable("must override match or matchAndRewrite");
  }
  virtual LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const {
    if (failed(match(op)))
      return failure();
    rewrite(op, adaptor, rewriter);
    return success();
  }

private:
  using CIRConversionPattern::match;
  using CIRConversionPattern::matchAndRewrite;
};
} // namespace

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//
namespace {
//===---------===//
// ConstantOp
//===---------===//
struct ConstantOpLowering : public ConvertCIROpToLLVMPattern<cir::ConstantOp> {
  using ConvertCIROpToLLVMPattern<cir::ConstantOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = op.getResult().getType();
    Type cirType;
    if (auto boxType = resultType.dyn_cast<CIRBoxType>())
      cirType = boxType.getElementType();
    else
      cirType = resultType;

    // Determine what type of constant this is and lower appropriately
    auto loc = op.getLoc();
    auto attr = adaptor.value();
    auto module = op->getParentOfType<ModuleOp>();
    auto replacement =
        TypeSwitch<Type, Value>(cirType)
            .Case<CIRNoneType>([&](CIRNoneType) {
              return createTermConstant(
                  rewriter, loc, encodeImmediate(lumen::TermKind::None, 0));
            })
            .Case<CIRNilType>([&](CIRNilType) {
              return createTermConstant(
                  rewriter, loc, encodeImmediate(lumen::TermKind::Nil, 0));
            })
            .Case<CIRIntegerType>([&](CIRIntegerType) {
              return createIntegerConstant(rewriter, loc,
                                           attr.cast<IsizeAttr>().getInt());
            })
            .Case<CIRIsizeType>([&](CIRIsizeType) {
              return createIntegerConstant(rewriter, loc,
                                           attr.cast<IsizeAttr>().getInt());
            })
            .Case<CIRFloatType>([&](CIRFloatType) {
              return createFloatConstant(
                  rewriter, loc, attr.cast<CIRFloatAttr>().getValue(), module);
            })
            .Case<CIRAtomType>([&](CIRAtomType) {
              return createAtom(rewriter, loc, attr.cast<AtomAttr>().getName(),
                                module);
            })
            .Case<CIRBoolType>([&](CIRBoolType) {
              return createTermConstant(
                  rewriter, loc,
                  encodeImmediate(lumen::TermKind::Atom,
                                  attr.cast<CIRBoolAttr>().getValue()));
            })
            .Default([](Type) { return nullptr; });

    if (!replacement)
      return rewriter.notifyMatchFailure(
          op, "failed to lower constant, unsupported constant type");

    rewriter.replaceOp(op, {replacement});
    return success();
  }
};

//===---------===//
// ConstantNullOp
//===---------===//
struct ConstantNullOpLowering
    : public ConvertCIROpToLLVMPattern<cir::ConstantNullOp> {
  using ConvertCIROpToLLVMPattern<
      cir::ConstantNullOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::ConstantNullOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto resultType = op.getResult().getType();

    if (resultType.isa<TermType>()) {
      Value none = createTermConstant(
          rewriter, loc, encodeImmediate(lumen::TermKind::None, 0));
      rewriter.replaceOp(op, {none});
      return success();
    }

    auto ty = convertType(resultType);
    rewriter.replaceOpWithNewOp<LLVM::NullOp>(op, ty);
    return success();
  }
};

//===---------===//
// CallOp
//===---------===//
struct CallOpLowering : public ConvertCIROpToLLVMPattern<cir::CallOp> {
  using ConvertCIROpToLLVMPattern<cir::CallOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto calleeType = op.getCalleeType();
    auto resultTypes = calleeType.getResults();

    rewriter.replaceOpWithNewOp<func::CallOp>(op, adaptor.callee(), resultTypes,
                                              adaptor.operands());
    return success();
  }
};

//===---------===//
// CastOp
//===---------===//
struct CastOpLowering : public ConvertCIROpToLLVMPattern<cir::CastOp> {
  using ConvertCIROpToLLVMPattern<cir::CastOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::CastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto inputs = adaptor.inputs();
    auto inputTypes = op.getInputTypes();
    auto outputTypes = op.getResultTypes();

    SmallVector<Value, 1> results;
    for (auto it : llvm::enumerate(inputTypes)) {
      Value input = inputs[it.index()];
      Type inputType = it.value();
      Type outputType = outputTypes[it.index()];

      // Casts from concrete term type to opaque term type are no-ops
      if (inputType.isa<TermType>() && outputType.isa<CIROpaqueTermType>()) {
        results.push_back(input);
      } else if (inputType.isa<CIRBoolType>() && outputType.isInteger(1)) {
        // To cast from a boolean term to its value, we extract the atom symbol
        // id and truncate to i1
        Value symbol = decodeImmediateValue(rewriter, loc, input);
        Value truncated =
            rewriter.create<LLVM::TruncOp>(loc, getI1Type(), symbol);
        results.push_back(truncated);
      } else if (inputType.isInteger(1) && outputType.isa<TermType>()) {
        // To cast from i1 to a boolean term, we treat the value as the symbol
        // id, zext and encode as an atom
        Value symbol =
            rewriter.create<LLVM::ZExtOp>(loc, getIsizeType(), input);
        Value encoded = encodeImmediateValue(
            rewriter, loc, rewriter.getType<CIRAtomType>(), symbol);
        results.push_back(encoded);
      } else {
        // No other casts are supported currently
        return rewriter.notifyMatchFailure(
            op,
            "failed to lower cast, unsupported source/target type combination");
      }
    }

    rewriter.replaceOp(op, results);
    return success();
  }
};

//===---------===//
// IsNullOp
//===---------===//
struct IsNullOpLowering : public ConvertCIROpToLLVMPattern<cir::IsNullOp> {
  using ConvertCIROpToLLVMPattern<cir::IsNullOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::IsNullOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto value = adaptor.value();
    Value nullValue = rewriter.create<LLVM::NullOp>(loc, value.getType());
    rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(op, LLVM::ICmpPredicate::eq,
                                              value, nullValue);
    return success();
  }
};

//===---------===//
// AndOp
//===---------===//
struct AndOpLowering : public ConvertCIROpToLLVMPattern<cir::AndOp> {
  using ConvertCIROpToLLVMPattern<cir::AndOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::AndOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto lhs = adaptor.lhs();
    auto rhs = adaptor.rhs();

    auto i1Ty = rewriter.getI1Type();
    rewriter.replaceOpWithNewOp<LLVM::AndOp>(op, i1Ty, ValueRange({lhs, rhs}));
    return success();
  }
};

//===---------===//
// AndAlsoOp
//===---------===//
struct AndAlsoOpLowering : public ConvertCIROpToLLVMPattern<cir::AndAlsoOp> {
  using ConvertCIROpToLLVMPattern<cir::AndAlsoOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::AndAlsoOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto lhs = adaptor.lhs();
    auto rhs = adaptor.rhs();

    auto i1Ty = rewriter.getI1Type();

    rewriter.replaceOpWithNewOp<scf::IfOp>(
        op, TypeRange({i1Ty}), lhs,
        [&](OpBuilder &builder, Location l) {
          builder.create<scf::YieldOp>(l, ValueRange({rhs}));
        },
        [&](OpBuilder &builder, Location l) {
          Value constFalse = createBoolConstant(rewriter, l, false);
          builder.create<scf::YieldOp>(l, ValueRange({constFalse}));
        });
    return success();
  }
};

//===---------===//
// OrOp
//===---------===//
struct OrOpLowering : public ConvertCIROpToLLVMPattern<cir::OrOp> {
  using ConvertCIROpToLLVMPattern<cir::OrOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::OrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto lhs = adaptor.lhs();
    auto rhs = adaptor.rhs();

    auto i1Ty = rewriter.getI1Type();
    rewriter.replaceOpWithNewOp<LLVM::OrOp>(op, i1Ty, ValueRange({lhs, rhs}));
    return success();
  }
};

//===---------===//
// OrElseOp
//===---------===//
struct OrElseOpLowering : public ConvertCIROpToLLVMPattern<cir::OrElseOp> {
  using ConvertCIROpToLLVMPattern<cir::OrElseOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::OrElseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto lhs = adaptor.lhs();
    auto rhs = adaptor.rhs();

    auto i1Ty = rewriter.getI1Type();

    rewriter.replaceOpWithNewOp<scf::IfOp>(
        op, TypeRange({i1Ty}), lhs,
        [&](OpBuilder &builder, Location l) {
          Value constTrue = createBoolConstant(rewriter, l, true);
          builder.create<scf::YieldOp>(l, ValueRange({constTrue}));
        },
        [&](OpBuilder &builder, Location l) {
          builder.create<scf::YieldOp>(l, ValueRange({rhs}));
        });
    return success();
  }
};

//===---------===//
// XorOp
//===---------===//
struct XorOpLowering : public ConvertCIROpToLLVMPattern<cir::XorOp> {
  using ConvertCIROpToLLVMPattern<cir::XorOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::XorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto lhs = adaptor.lhs();
    auto rhs = adaptor.rhs();

    auto i1Ty = rewriter.getI1Type();

    rewriter.replaceOpWithNewOp<LLVM::XOrOp>(op, i1Ty, ValueRange({lhs, rhs}));
    return success();
  }
};

//===---------===//
// NotOp
//===---------===//
struct NotOpLowering : public ConvertCIROpToLLVMPattern<cir::NotOp> {
  using ConvertCIROpToLLVMPattern<cir::NotOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::NotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto value = adaptor.value();

    auto i1Ty = rewriter.getI1Type();

    Value constTrue = createBoolConstant(rewriter, loc, true);
    rewriter.replaceOpWithNewOp<LLVM::XOrOp>(op, i1Ty,
                                             ValueRange({value, constTrue}));
    return success();
  }
};

//===---------===//
// TypeOfImmediateOp
//===---------===//
struct TypeOfImmediateOpLowering
    : public ConvertCIROpToLLVMPattern<cir::TypeOfImmediateOp> {
  using ConvertCIROpToLLVMPattern<
      cir::TypeOfImmediateOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::TypeOfImmediateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // We extend the mask to include the literal tag bits since it isn't a
    // proper type and makes checking for boxiness more complicated
    Value tag = extractImmediateTag(rewriter, loc, adaptor.value(),
                                    /*stripLiteral=*/true);
    rewriter.replaceOp(op, ValueRange({tag}));
    return success();
  }
};

//===---------===//
// TypeOfBoxOp
//===---------===//
struct TypeOfBoxOpLowering
    : public ConvertCIROpToLLVMPattern<cir::TypeOfBoxOp> {
  using ConvertCIROpToLLVMPattern<cir::TypeOfBoxOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::TypeOfBoxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // NOTE: This op does not assert that the input value is a box, it is up to
    // higher stages of the compiler to ensure that is the case. In the future
    // we can probably add debug assertions here to catch compiler bugs, but for
    // now we operate on the assumption that the compiler did its job
    auto loc = op.getLoc();
    // TODO: Possibly use ptrmask intrinsic
    auto termTy = getTermType();
    auto ptrTy = LLVM::LLVMPointerType::get(termTy);
    // Strip the tag bits for box/literal pointer types, bitcast to pointer
    auto tag = createIsizeConstant(rewriter, loc, boxTag() | literalTag());
    auto neg1 = createIsizeConstant(rewriter, loc, -1);
    Value mask = rewriter.create<LLVM::XOrOp>(loc, tag, neg1);
    Value untagged = rewriter.create<LLVM::AndOp>(loc, adaptor.value(), mask);
    Value ptr = rewriter.create<LLVM::IntToPtrOp>(loc, ptrTy, untagged);
    // Dereference the pointer and extract the type kind and arity/value
    Value header = rewriter.create<LLVM::LoadOp>(loc, ptr);
    auto maskInfo = headerMask();
    // The mask we have is for the value, so we have to invert the mask to get a
    // tag mask
    Value valueMask = createIsizeConstant(rewriter, loc, maskInfo.mask);
    Value tagMask = rewriter.create<LLVM::XOrOp>(loc, valueMask, neg1);
    // Now that we have our masks, extract the bits for tag and value
    Value kind = rewriter.create<LLVM::AndOp>(loc, header, tagMask);
    Value untaggedHeader = rewriter.create<LLVM::AndOp>(loc, header, valueMask);
    // The value needs to be shifted (on some targets), and clamped
    Value maxValueMask =
        createIsizeConstant(rewriter, loc, maskInfo.maxAllowedValue);
    if (maskInfo.requiresShift()) {
      Value shift = createIsizeConstant(rewriter, loc, maskInfo.shift);
      Value shifted = rewriter.create<LLVM::LShrOp>(loc, untaggedHeader, shift);
      Value cleaned = rewriter.create<LLVM::AndOp>(loc, shifted, maxValueMask);
      rewriter.replaceOp(op, {kind, cleaned});
    } else {
      Value cleaned =
          rewriter.create<LLVM::AndOp>(loc, untaggedHeader, maxValueMask);
      rewriter.replaceOp(op, {kind, cleaned});
    }
    return success();
  }
};

//===---------===//
// TypeOfOp
//===---------===//
struct TypeOfOpLowering : public ConvertCIROpToLLVMPattern<cir::TypeOfOp> {
  using ConvertCIROpToLLVMPattern<cir::TypeOfOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::TypeOfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto isizeTy = getIsizeType();
    Value input = adaptor.value();
    Value immKind = rewriter.create<cir::TypeOfImmediateOp>(loc, input);
    Value tag = createIsizeConstant(rewriter, loc, boxTag());
    Value isBox = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq,
                                                immKind, tag);
    // We lower this op through scf.if, which provides us the semantics we want
    // without requiring our CIR lowering code to know that this op requires
    // branches internally
    rewriter.replaceOpWithNewOp<scf::IfOp>(
        op, TypeRange({isizeTy, isizeTy}), isBox,
        // isBox==true
        [&](OpBuilder &builder, Location l) {
          auto typeOfOp = builder.create<cir::TypeOfBoxOp>(l, input);
          builder.create<scf::YieldOp>(l, typeOfOp.getResults());
        },
        // isBox==false
        [&](OpBuilder &builder, Location l) {
          auto zero = createIsizeConstant(rewriter, l, 0);
          builder.create<scf::YieldOp>(l, ValueRange({immKind, zero}));
        });
    return success();
  }
};

//===---------===//
// IsTypeOp
//===---------===//
struct IsTypeOpLowering : public ConvertCIROpToLLVMPattern<cir::IsTypeOp> {
  using ConvertCIROpToLLVMPattern<cir::IsTypeOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::IsTypeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto expectedType = adaptor.expected();
    auto value = adaptor.value();

    // Classify the expected type as immediate/boxed, and obtain the kind we
    // expect
    lumen::TermKind::Kind kind = lumen::TermKind::None;
    bool isImmediate = false;
    bool isBoxed = false;
    if (auto boxTy = expectedType.dyn_cast<CIRBoxType>()) {
      auto innerTy = boxTy.getElementType();
      // A box of type term is equivalent to a type check that a value is any
      // boxed term; otherwise we are checking that the input value is a boxed
      // term of a specific type
      if (innerTy.isa<CIROpaqueTermType>()) {
        isImmediate = true;
        isBoxed = false;
        kind = lumen::TermKind::Box;
      } else {
        isImmediate = false;
        isBoxed = true;
        kind = innerTy.cast<TermType>().getTermKind();
      }
    } else if (expectedType.isa<ImmediateType>()) {
      isImmediate = true;
      isBoxed = false;
      kind = expectedType.cast<TermType>().getTermKind();
    } else if (expectedType.isa<BoxedType>()) {
      isImmediate = false;
      isBoxed = true;
      kind = expectedType.cast<TermType>().getTermKind();
    } else if (auto targetReprTy =
                   expectedType.dyn_cast<TargetSensitiveReprType>()) {
      // Treat this type as either immediate or boxed depending on the target
      // encoding
      isImmediate = targetReprTy.isImmediate(termEncoding());
      isBoxed = !isImmediate;
      kind = expectedType.cast<TermType>().getTermKind();
    } else if (expectedType.isa<TupleType>()) {
      isImmediate = false;
      isBoxed = true;
      kind = lumen::TermKind::Tuple;
    }

    auto i1Ty = rewriter.getI1Type();
    // If we can't say concretely that this is an immediate or boxed type, it is
    // because either:
    // * the type is generic, e.g. "number" or "integer"
    // * the type is invalid, i.e. not a term
    if (!isImmediate && !isBoxed) {
      if (expectedType.isa<CIRNumberType>() ||
          expectedType.isa<CIRIntegerType>()) {
        // Capture the concrete type of the input value
        auto typeOfOp = rewriter.create<cir::TypeOfOp>(loc, value);
        Value tag = typeOfOp.kind();
        // We need to split the current block before the current operation, add
        // a block argument to the block containing the current op which will
        // contain the result of the type check, and replace the op with the
        // block argument. In the origin block we insert a cf.switch op that
        // dispatches to the continuation block with true/false based on tag
        // equality.
        auto originBlock = op->getBlock();
        auto contBlock = rewriter.splitBlock(originBlock, Block::iterator(op));
        Value resultArg = contBlock->addArgument(rewriter.getI1Type(), loc);
        rewriter.setInsertionPointToEnd(originBlock);
        Value constFalse = createBoolConstant(rewriter, loc, false);
        Value constTrue = createBoolConstant(rewriter, loc, true);
        SmallVector<APInt> values;
        SmallVector<Block *> dests;
        SmallVector<ValueRange> operands;
        // isize
        auto bitwidth = getPointerBitwidth();
        values.push_back(APInt(immediateTag(lumen::TermKind::Isize), bitwidth));
        dests.push_back(contBlock);
        operands.push_back(ValueRange(constTrue));
        // bigint
        values.push_back(
            APInt(immediateTag(lumen::TermKind::BigInt), bitwidth));
        dests.push_back(contBlock);
        operands.push_back(ValueRange(constTrue));
        // float, but only if this is a type check for number
        if (expectedType.isa<CIRNumberType>()) {
          values.push_back(
              APInt(immediateTag(lumen::TermKind::Float), bitwidth));
          dests.push_back(contBlock);
          operands.push_back(ValueRange(constTrue));
        }
        rewriter.create<cf::SwitchOp>(loc, tag, contBlock,
                                      ValueRange({constFalse}), values, dests,
                                      operands);
        // Switch back to continuation block, and replace op with the block
        // argument to which the switch value resolves
        rewriter.setInsertionPoint(op);
        rewriter.replaceOp(op, {resultArg});
        return success();
      }

      return rewriter.notifyMatchFailure(
          op, "failed to lower is_type op, unsupported match type");
    }

    // Get the tag of the value we have
    Value tag = rewriter.create<cir::TypeOfImmediateOp>(loc, value);
    // If the expected type is an immediate, we can check that trivially
    if (isImmediate) {
      auto expectedTag = createIsizeConstant(rewriter, loc, immediateTag(kind));
      rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(op, LLVM::ICmpPredicate::eq,
                                                tag, expectedTag);
      return success();
    }

    // If the expected type is a cons cell, we can also check that trivially
    if (kind == lumen::TermKind::Cons || kind == lumen::TermKind::List) {
      auto expectedTag = createIsizeConstant(rewriter, loc, listTag());
      rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(op, LLVM::ICmpPredicate::eq,
                                                tag, expectedTag);
      return success();
    }

    // If the expected type is any other boxed type, we must check if the input
    // value is a boxed type, and if so, extract its header kind and then
    // compare; otherwise the check is trivially false
    auto boxTagConst = createIsizeConstant(rewriter, loc, boxTag());
    Value isBox = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq,
                                                tag, boxTagConst);
    rewriter.replaceOpWithNewOp<scf::IfOp>(
        op, TypeRange({i1Ty}), isBox,
        // isBox==true
        [&](OpBuilder &builder, Location l) {
          auto typeOfBoxOp = builder.create<cir::TypeOfBoxOp>(l, value);
          Value boxTy = typeOfBoxOp.kind();
          Value expectedTag = createIsizeConstant(rewriter, l, headerTag(kind));
          Value isEq = builder.create<LLVM::ICmpOp>(l, LLVM::ICmpPredicate::eq,
                                                    boxTy, expectedTag);
          builder.create<scf::YieldOp>(l, ValueRange({isEq}));
        },
        // isBox==false
        [&](OpBuilder &builder, Location l) {
          Value constFalse = createBoolConstant(builder, l, false);
          builder.create<scf::YieldOp>(l, ValueRange({constFalse}));
        });
    return success();
  }
};

//===---------===//
// IsTaggedTupleOp
//===---------===//
struct IsTaggedTupleOpLowering
    : public ConvertCIROpToLLVMPattern<cir::IsTaggedTupleOp> {
  using ConvertCIROpToLLVMPattern<
      cir::IsTaggedTupleOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::IsTaggedTupleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto input = adaptor.value();
    auto module = op->getParentOfType<ModuleOp>();
    AtomAttr atom = adaptor.tag();

    auto i1Ty = rewriter.getI1Type();
    auto i32Ty = rewriter.getI32Type();

    // Get the type of the input and its arity
    auto typeOfOp = rewriter.create<cir::TypeOfOp>(loc, input);
    Value kind = typeOfOp.kind();
    Value arity = typeOfOp.arity();
    // Check if the input value is a tuple
    auto expectedKind =
        createIsizeConstant(rewriter, loc, headerTag(lumen::TermKind::Tuple));
    Value isTuple = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq,
                                                  kind, expectedKind);
    // Check if the input value arity is at least 1
    auto minArity = createIsizeConstant(rewriter, loc, 1);
    Value hasArity = rewriter.create<LLVM::ICmpOp>(
        loc, LLVM::ICmpPredicate::uge, arity, minArity);
    // Compose the two booleans into a single value, and:
    // if true, extract the first element of the tuple and compare it to the
    // expected atom if false, we have our answer.
    Value checkTag = rewriter.create<LLVM::AndOp>(loc, isTuple, hasArity);
    rewriter.replaceOpWithNewOp<scf::IfOp>(
        op, TypeRange({i1Ty}), checkTag,
        // checkTag==true, i.e. we should examine the first element of the tuple
        [&](OpBuilder &builder, Location l) {
          Type tupleTy = getTupleType(1);
          Value tuplePtr = decodePtr(builder, l, tupleTy, input);
          SmallVector<Value> indices;
          // This first index refers to the base of the tuple
          // This second index refers to the second field of the tuple struct
          // (i.e. the data) This third index refers to the first element of the
          // tuple data
          Value zero = createIsizeConstant(builder, l, 0);
          Value one = createIndexAttrConstant(builder, l, i32Ty, 1);
          Value elemPtr = builder.create<LLVM::GEPOp>(
              l, tupleTy, tuplePtr, ValueRange({zero, one, zero}));
          Value elem = builder.create<LLVM::LoadOp>(l, elemPtr);
          Value expectedAtom = createAtom(builder, l, atom.getName(), module);
          Value isEq = builder.create<LLVM::ICmpOp>(l, LLVM::ICmpPredicate::eq,
                                                    elem, expectedAtom);
          builder.create<scf::YieldOp>(l, ValueRange({isEq}));
        },
        // checkTag==false, i.e. this tuple can't possibly match
        [&](OpBuilder &builder, Location l) {
          Value constFalse = builder.create<LLVM::ConstantOp>(
              l, i1Ty, builder.getIntegerAttr(i1Ty, 0));
          builder.create<scf::YieldOp>(l, ValueRange({constFalse}));
        });
    return success();
  }
};

//===---------===//
// MallocOp
//===---------===//
struct MallocOpLowering : public ConvertCIROpToLLVMPattern<cir::MallocOp> {
  using ConvertCIROpToLLVMPattern<cir::MallocOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::MallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto allocType = adaptor.allocType();
    auto i8Ty = rewriter.getI8Type();
    auto i32Ty = rewriter.getI32Type();
    auto i8PtrTy = LLVM::LLVMPointerType::get(i8Ty);

    auto boxedTy = allocType.dyn_cast<BoxedType>();
    auto kind = lumen::TermKind::None;
    if (!boxedTy && !allocType.isa<TupleType>())
      return rewriter.notifyMatchFailure(
          op, "failed to lower malloc op, unsupported alloc type");
    if (!boxedTy)
      kind = lumen::TermKind::Tuple;
    else
      kind = allocType.cast<TermType>().getTermKind();

    // We only support malloc for a limited subset of types
    uint64_t arity;
    switch (kind) {
    case lumen::TermKind::Tuple:
      arity = allocType.cast<TupleType>().size();
      break;
    case lumen::TermKind::Fun:
      arity = allocType.cast<CIRFunType>().getEnvArity();
      break;
    case lumen::TermKind::Cons:
      arity = 0;
      break;
    default:
      return rewriter.notifyMatchFailure(
          op, "failed to lower malloc op, unsupported alloc type");
    }

    auto kindArg = createIndexAttrConstant(rewriter, loc, i32Ty, kind);
    auto arityArg = createIsizeConstant(rewriter, loc, arity);
    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, TypeRange({i8PtrTy}), "__lumen_builtin_malloc",
        ValueRange({kindArg, arityArg}));

    auto ptrTy = LLVM::LLVMPointerType::get(convertType(allocType));
    rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(op, ptrTy,
                                                 callOp->getResult(0));
    return success();
  }
};

//===---------===//
// ConsOp
//===---------===//
struct ConsOpLowering : public ConvertCIROpToLLVMPattern<cir::ConsOp> {
  using ConvertCIROpToLLVMPattern<cir::ConsOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::ConsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // Allocate space on the process heap for the cell
    auto i32Ty = rewriter.getI32Type();
    auto consTy = rewriter.getType<CIRConsType>();
    auto consBoxTy = CIRBoxType::get(consTy);
    Value ptr = rewriter.create<cir::MallocOp>(loc, consBoxTy, consTy);
    auto cellTy = getConsType();
    Value base = createIsizeConstant(rewriter, loc, 0);
    Value zero = createIndexAttrConstant(rewriter, loc, i32Ty, 0);
    Value one = createIndexAttrConstant(rewriter, loc, i32Ty, 1);
    // Get a pointer to the head and tail and store their values
    auto headPtr = rewriter.create<LLVM::GEPOp>(loc, cellTy, ptr,
                                                ValueRange({base, zero}));
    rewriter.create<LLVM::StoreOp>(loc, headPtr, adaptor.head());
    auto tailPtr =
        rewriter.create<LLVM::GEPOp>(loc, cellTy, ptr, ValueRange({base, one}));
    rewriter.create<LLVM::StoreOp>(loc, tailPtr, adaptor.tail());
    // Box the pointer to the allocation as a list
    Value box = encodeListPtr(rewriter, loc, ptr);

    // Return the boxed value
    rewriter.replaceOp(op, ValueRange({box}));
    return success();
  }
};

//===---------===//
// HeadOp
//===---------===//
struct HeadOpLowering : public ConvertCIROpToLLVMPattern<cir::HeadOp> {
  using ConvertCIROpToLLVMPattern<cir::HeadOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::HeadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // NOTE: We should generate assertions that the input is of the correct type
    // during debug builds, but for now we lower with the assumption that the
    // compiler has already generated those checks

    // First, unbox the pointer as a pointer to a term (the pointer is actually
    // to a cons cell, but we're not accessing the tail, so we can skip the
    // unnecessary getelementptr instruction by using the struct type)
    auto isizeTy = getIsizeType();
    Value ptr = decodePtr(rewriter, loc, isizeTy, adaptor.cell());

    // Then return the result of loading the value from the pointer
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, ptr);
    return success();
  }
};

//===---------===//
// TailOp
//===---------===//
struct TailOpLowering : public ConvertCIROpToLLVMPattern<cir::TailOp> {
  using ConvertCIROpToLLVMPattern<cir::TailOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::TailOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // NOTE: We should generate assertions that the input is of the correct type
    // during debug builds, but for now we lower with the assumption that the
    // compiler has already generated those checks

    // First, unbox the pointer as a pointer to a cons cell
    auto i32Ty = rewriter.getI32Type();
    auto cellTy = getConsType();
    Value ptr = decodeListPtr(rewriter, loc, adaptor.cell());

    // Then, calculate the pointer to the tail cell
    Value base = createIsizeConstant(rewriter, loc, 0);
    Value one = createIndexAttrConstant(rewriter, loc, i32Ty, 1);
    auto tailPtr =
        rewriter.create<LLVM::GEPOp>(loc, cellTy, ptr, ValueRange({base, one}));

    // Then return the result of loading the value from the calculated pointer
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, tailPtr);
    return success();
  }
};

//===---------===//
// TupleOp
//===---------===//
struct TupleOpLowering : public ConvertCIROpToLLVMPattern<cir::TupleOp> {
  using ConvertCIROpToLLVMPattern<cir::TupleOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::TupleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto arity = adaptor.arity();

    // Allocate space on the process heap for the tuple
    auto termTy = getTermType();
    SmallVector<Type> elementTypes;
    elementTypes.reserve(arity);
    for (uint32_t i = 0; i < arity; i++)
      elementTypes.push_back(termTy);

    auto tupleTy = rewriter.getTupleType(elementTypes);
    auto boxedTupleTy = CIRBoxType::get(tupleTy);
    Value ptr = rewriter.create<cir::MallocOp>(loc, boxedTupleTy, tupleTy);

    // Box the pointer and return it
    Value box = encodePtr(rewriter, loc, ptr);
    rewriter.replaceOp(op, ValueRange({box}));
    return success();
  }
};

//===---------===//
// GetElementOp
//===---------===//
struct GetElementOpLowering
    : public ConvertCIROpToLLVMPattern<cir::GetElementOp> {
  using ConvertCIROpToLLVMPattern<cir::GetElementOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::GetElementOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto tuple = adaptor.tuple();
    auto index = adaptor.index();

    // NOTE: We should generate assertions that the input is of the correct
    // shape during debug builds, but for now we lower with the assumption that
    // the compiler has already generated those checks

    // First, unbox the pointer as a pointer to a tuple
    auto i32Ty = rewriter.getI32Type();
    auto isizeTy = getIsizeType();
    auto tupleTy = getTupleType(1);
    auto tuplePtrTy = LLVM::LLVMPointerType::get(tupleTy);
    Value ptr = decodePtr(rewriter, loc, isizeTy, tuple);
    Value tuplePtr = rewriter.create<LLVM::BitcastOp>(loc, tuplePtrTy, ptr);
    // Then, calculate the pointer to the <index>th element
    Value base = createIsizeConstant(rewriter, loc, 0);
    Value one = createIndexAttrConstant(rewriter, loc, i32Ty, 1);
    Value elemPtr = rewriter.create<LLVM::GEPOp>(
        loc, tupleTy, tuplePtr, ValueRange({base, one, index}));
    // Then return the result of loading the value from the calculated pointer
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, elemPtr);
    return success();
  }
};

//===---------===//
// SetElementOp
//===---------===//
struct SetElementOpLowering
    : public ConvertCIROpToLLVMPattern<cir::SetElementOp> {
  using ConvertCIROpToLLVMPattern<cir::SetElementOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::SetElementOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto tuple = adaptor.tuple();
    auto index = adaptor.index();
    auto value = adaptor.value();

    // NOTE: We should generate assertions that the input is of the correct
    // shape during debug builds, but for now we lower with the assumption that
    // the compiler has already generated those checks

    // First, unbox the pointer as a pointer to a tuple
    auto i32Ty = rewriter.getI32Type();
    auto isizeTy = getIsizeType();
    auto tupleTy = getTupleType(1);
    auto tuplePtrTy = LLVM::LLVMPointerType::get(tupleTy);
    Value ptr = decodePtr(rewriter, loc, isizeTy, tuple);
    Value tuplePtr = rewriter.create<LLVM::BitcastOp>(loc, tuplePtrTy, ptr);
    // Then, calculate the pointer to the <index>th element
    Value base = createIsizeConstant(rewriter, loc, 0);
    Value one = createIndexAttrConstant(rewriter, loc, i32Ty, 1);
    Value elemPtr = rewriter.create<LLVM::GEPOp>(
        loc, tupleTy, tuplePtr, ValueRange({base, one, index}));
    // Then store the input value at the calculated pointer
    rewriter.create<LLVM::StoreOp>(loc, elemPtr, value);
    rewriter.replaceOp(op, ValueRange({tuple}));
    return success();
  }
};

//===---------===//
// RaiseOp
//===---------===//
struct RaiseOpLowering : public ConvertCIROpToLLVMPattern<cir::RaiseOp> {
  using ConvertCIROpToLLVMPattern<cir::RaiseOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::RaiseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto klass = adaptor.exceptionClass();
    auto reason = adaptor.exceptionReason();
    auto trace = adaptor.exceptionTrace();

    // Get a reference to the process exception pointer
    auto module = op->getParentOfType<ModuleOp>();
    auto exceptionTls =
        module.lookupSymbol<LLVM::GlobalOp>("__lumen_process_exception");
    if (!exceptionTls)
      exceptionTls =
          insertProcessExceptionThreadLocal(rewriter, module.getLoc(), module);

    // Get a reference to the process signal enum
    auto signalTls =
        module.lookupSymbol<LLVM::GlobalOp>("__lumen_process_signal");
    if (!signalTls)
      signalTls =
          insertProcessSignalThreadLocal(rewriter, module.getLoc(), module);

    auto exceptionTy = getExceptionType();
    auto exceptionPtrTy = LLVM::LLVMPointerType::get(exceptionTy);
    auto klassTy = getTermType();
    auto reasonTy = getTermType();
    auto traceTy = getTraceType();
    Operation *callee = module.lookupSymbol("__lumen_builtin_raise/3");
    if (!callee) {
      auto calleeType = LLVM::LLVMFunctionType::get(
          exceptionPtrTy, ArrayRef<Type>{klassTy, reasonTy, traceTy});
      insertFunctionDeclaration(rewriter, loc, module,
                                "__lumen_builtin_raise/3", calleeType);
    }

    // Create the raw exception value using __lumen_builtin_raise/3
    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, TypeRange({exceptionPtrTy}), "__lumen_builtin_raise/3",
        ValueRange({klass, reason, trace}));

    // Then set the value of the process exception pointer and process signal
    // globals
    auto exceptionPtr = callOp.getResult(0);
    Value exceptionTlsPtr =
        rewriter.create<LLVM::AddressOfOp>(loc, exceptionTls);
    rewriter.create<LLVM::StoreOp>(loc, exceptionPtr, exceptionTlsPtr);

    auto i8Ty = rewriter.getI8Type();
    Value signalTlsPtr = rewriter.create<LLVM::AddressOfOp>(loc, signalTls);
    Value errorSignal = rewriter.create<LLVM::ConstantOp>(
        loc, i8Ty, rewriter.getI8IntegerAttr(/*ProcessSignal::Error*/ 3));
    rewriter.create<LLVM::StoreOp>(loc, errorSignal, signalTlsPtr);

    // Lastly, convert this op to a multi-value return where the result is
    // None and the error flag is set to true
    // auto errorFlag = createIsizeConstant(rewriter, loc, 1);
    auto none =
        createIsizeConstant(rewriter, loc, immediateTag(lumen::TermKind::None));
    rewriter.replaceOpWithNewOp<func::ReturnOp>(
        op, ValueRange({none, exceptionPtr}));
    return success();
  }
};

//===------------===//
// BuildStacktraceOp
//===------------===//
struct BuildStacktraceOpLowering
    : public ConvertCIROpToLLVMPattern<cir::BuildStacktraceOp> {
  using ConvertCIROpToLLVMPattern<
      cir::BuildStacktraceOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::BuildStacktraceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto traceTy = getTraceType();
    auto module = op->getParentOfType<ModuleOp>();
    Operation *callee = module.lookupSymbol("__lumen_build_stacktrace");
    if (!callee) {
      auto calleeType = LLVM::LLVMFunctionType::get(traceTy, ArrayRef<Type>{});
      insertFunctionDeclaration(rewriter, loc, module,
                                "__lumen_build_stacktrace", calleeType);
    }
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, TypeRange({traceTy}), "__lumen_build_stacktrace", ValueRange());
    return success();
  }
};

//===------------===//
// ExceptionClassOp
//===------------===//
struct ExceptionClassOpLowering
    : public ConvertCIROpToLLVMPattern<cir::ExceptionClassOp> {
  using ConvertCIROpToLLVMPattern<
      cir::ExceptionClassOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::ExceptionClassOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto exceptionPtr = adaptor.exception();

    auto i32Ty = getI32Type();
    auto exceptionTy = getExceptionType();
    Value zero = createIsizeConstant(rewriter, loc, 0);
    Value one = createIndexAttrConstant(rewriter, loc, i32Ty, 1);
    auto classAddr = rewriter.create<LLVM::GEPOp>(
        loc, exceptionTy, exceptionPtr, ValueRange({zero, one}));

    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, classAddr);
    return success();
  }
};

//===------------===//
// ExceptionReasonOp
//===------------===//
struct ExceptionReasonOpLowering
    : public ConvertCIROpToLLVMPattern<cir::ExceptionReasonOp> {
  using ConvertCIROpToLLVMPattern<
      cir::ExceptionReasonOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::ExceptionReasonOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto exceptionPtr = adaptor.exception();

    auto i32Ty = getI32Type();
    auto exceptionTy = getExceptionType();
    Value zero = createIsizeConstant(rewriter, loc, 0);
    Value two = createIndexAttrConstant(rewriter, loc, i32Ty, 2);
    auto reasonAddr = rewriter.create<LLVM::GEPOp>(
        loc, exceptionTy, exceptionPtr, ValueRange({zero, two}));

    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, reasonAddr);
    return success();
  }
};

//===------------===//
// ExceptionTraceOp
//===------------===//
struct ExceptionTraceOpLowering
    : public ConvertCIROpToLLVMPattern<cir::ExceptionTraceOp> {
  using ConvertCIROpToLLVMPattern<
      cir::ExceptionTraceOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::ExceptionTraceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto exceptionPtr = adaptor.exception();

    auto i32Ty = getI32Type();
    auto exceptionTy = getExceptionType();
    Value zero = createIsizeConstant(rewriter, loc, 0);
    Value three = createIndexAttrConstant(rewriter, loc, i32Ty, 3);
    auto traceAddr = rewriter.create<LLVM::GEPOp>(
        loc, exceptionTy, exceptionPtr, ValueRange({zero, three}));
    auto tracePtr = rewriter.create<LLVM::LoadOp>(loc, traceAddr);

    auto termTy = getTermType();
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange({termTy}),
                                              "__lumen_stacktrace_to_term",
                                              ValueRange({tracePtr}));
    return success();
  }
};

//===------------===//
// YieldOp
//===------------===//
struct YieldOpLowering : public ConvertCIROpToLLVMPattern<cir::YieldOp> {
  using ConvertCIROpToLLVMPattern<cir::YieldOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto voidTy = getVoidType();

    // If this op was not stripped by a pass, we're on a target which supports
    // stack switching, so lower this to a call to the yield intrinsic
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, TypeRange({voidTy}), "__lumen_builtin_yield", ValueRange());
    return success();
  }
};

//===------------===//
// RecvStartOp
//===------------===//
struct RecvStartOpLowering
    : public ConvertCIROpToLLVMPattern<cir::RecvStartOp> {
  using ConvertCIROpToLLVMPattern<cir::RecvStartOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::RecvStartOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto recvCtxTy = getRecvContextType();
    auto timeout = adaptor.timeout();

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange({recvCtxTy}),
                                              "__lumen_builtin_receive_start",
                                              ValueRange({timeout}));
    return success();
  }
};

//===------------===//
// RecvNextOp
//===------------===//
struct RecvNextOpLowering : public ConvertCIROpToLLVMPattern<cir::RecvNextOp> {
  using ConvertCIROpToLLVMPattern<cir::RecvNextOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::RecvNextOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto i8Ty = getI8Type();
    auto context = adaptor.context();

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange({i8Ty}),
                                              "__lumen_builtin_receive_start",
                                              ValueRange({context}));
    return success();
  }
};

//===------------===//
// RecvPeekOp
//===------------===//
struct RecvPeekOpLowering : public ConvertCIROpToLLVMPattern<cir::RecvPeekOp> {
  using ConvertCIROpToLLVMPattern<cir::RecvPeekOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::RecvPeekOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto i32Ty = getI32Type();
    auto recvCtxTy = getRecvContextType();
    auto messageTy = getMessageType();
    auto context = adaptor.context();

    Value zero = createIsizeConstant(rewriter, loc, 0);
    Value zero32 = createIndexAttrConstant(rewriter, loc, i32Ty, 0);
    Value one = createIndexAttrConstant(rewriter, loc, i32Ty, 1);
    auto two = rewriter.getI32ArrayAttr({2});
    Value messagePtr =
        rewriter.create<LLVM::ExtractValueOp>(loc, recvCtxTy, context, two);
    Value dataAddr;
    if (getPointerBitwidth() == 64) {
      dataAddr = rewriter.create<LLVM::GEPOp>(
          loc, messageTy, messagePtr, ValueRange({zero, one, one, one}));
    } else {
      dataAddr = rewriter.create<LLVM::GEPOp>(
          loc, messageTy, messagePtr, ValueRange({zero, one, one, zero32}));
    }
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, dataAddr);
    return success();
  }
};

//===------------===//
// RecvPopOp
//===------------===//
struct RecvPopOpLowering : public ConvertCIROpToLLVMPattern<cir::RecvPopOp> {
  using ConvertCIROpToLLVMPattern<cir::RecvPopOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::RecvPopOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto voidTy = getVoidType();
    auto context = adaptor.context();

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange({voidTy}),
                                              "__lumen_builtin_receive_pop",
                                              ValueRange({context}));
    return success();
  }
};

//===------------===//
// RecvDoneOp
//===------------===//
struct RecvDoneOpLowering : public ConvertCIROpToLLVMPattern<cir::RecvDoneOp> {
  using ConvertCIROpToLLVMPattern<cir::RecvDoneOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::RecvDoneOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto voidTy = getVoidType();
    auto context = adaptor.context();

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange({voidTy}),
                                              "__lumen_builtin_receive_done",
                                              ValueRange({context}));
    return success();
  }
};

//===------------===//
// DispatchTableOp
//===------------===//
struct DispatchTableOpLowering
    : public ConvertCIROpToLLVMPattern<cir::DispatchTableOp> {
  using ConvertCIROpToLLVMPattern<
      cir::DispatchTableOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::DispatchTableOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // All that is necessary here is that we inline the dispatch entries in the
    // module body
    auto &region = op.getRegion();
    auto &block = region.back();
    auto module = op.getModule();

    auto mod = op->getParentOfType<ModuleOp>();
    auto moduleBody = mod.getBody();
    auto i8ty = getI8Type();
    auto dispatchEntryTy = getDispatchEntryType();

    for (DispatchEntryOp entryOp : block.getOps<DispatchEntryOp>()) {
      PatternRewriter::InsertionGuard insertGuard(rewriter);
      rewriter.setInsertionPointToEnd(moduleBody);

      auto loc = entryOp->getLoc();
      auto function = entryOp.getFunction();
      auto arity = entryOp.getArity();
      auto symbol = entryOp.getSymbol();

      llvm::SHA1 hasher;
      hasher.update(module);
      hasher.update(function);
      auto arityStr = std::to_string(arity);
      hasher.update(StringRef(arityStr));

      auto globalName =
          std::string("lumen_dispatch_") + llvm::toHex(hasher.result(), true);
      std::string sectionName;
      if (isMachO())
        sectionName = std::string("__TEXT,__lumen_dispatch");
      else
        sectionName = std::string("__") + globalName;
      auto sectionAttr =
          rewriter.getNamedAttr("section", rewriter.getStringAttr(sectionName));
      auto entryConst = rewriter.create<LLVM::GlobalOp>(
          loc, dispatchEntryTy, /*isConstant=*/true, LLVM::Linkage::LinkonceODR,
          LLVM::ThreadLocalMode::NotThreadLocal, globalName, Attribute(),
          /*alignment=*/0, /*addrspace=*/0, /*dso_local=*/false,
          ArrayRef<NamedAttribute>{sectionAttr});

      auto &initRegion = entryConst.getInitializerRegion();
      auto entryBlock = rewriter.createBlock(&initRegion);

      rewriter.setInsertionPointToStart(entryBlock);

      Value entry = rewriter.create<LLVM::UndefOp>(loc, dispatchEntryTy);

      // Store the module name
      auto moduleNamePtr = createAtomStringGlobal(rewriter, loc, mod, module);
      entry = rewriter.create<LLVM::InsertValueOp>(loc, entry, moduleNamePtr,
                                                   rewriter.getI64ArrayAttr(0));

      // Store the function name
      auto functionNamePtr =
          createAtomStringGlobal(rewriter, loc, mod, function);
      entry = rewriter.create<LLVM::InsertValueOp>(loc, entry, functionNamePtr,
                                                   rewriter.getI64ArrayAttr(1));

      // Store the arity
      auto arityVal = rewriter.create<LLVM::ConstantOp>(
          loc, i8ty, rewriter.getI8IntegerAttr(arity));
      entry = rewriter.create<LLVM::InsertValueOp>(loc, entry, arityVal,
                                                   rewriter.getI64ArrayAttr(2));

      // Get the LLVM type of the function referenced by the symbol
      Operation *fun = mod.lookupSymbol(symbol.getValue());
      Type funTy;
      if (isa<LLVM::LLVMFuncOp>(fun))
        funTy = cast<LLVM::LLVMFuncOp>(fun).getFunctionType();
      else
        funTy = convertType(cast<FuncOp>(fun).getFunctionType());
      auto funPtr = rewriter.create<LLVM::AddressOfOp>(loc, funTy, symbol);
      // Cast the address of the function to an opaque function pointer (i.e.
      // `*const ()`)
      auto opaqueFunTy =
          LLVM::LLVMFunctionType::get(getVoidType(), ArrayRef<Type>{});
      auto opaqueFunPtrTy = LLVM::LLVMPointerType::get(opaqueFunTy);
      auto opaqueFunPtr =
          rewriter.create<LLVM::BitcastOp>(loc, opaqueFunPtrTy, funPtr);
      // Store the function pointer
      entry = rewriter.create<LLVM::InsertValueOp>(loc, entry, opaqueFunPtr,
                                                   rewriter.getI64ArrayAttr(3));

      rewriter.create<LLVM::ReturnOp>(loc, entry);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// ConvertCIRToLLVMPass
//
// This is the actual pass that applies all of our conversions and produces a
// module containing only LLVM dialect operations.
//===----------------------------------------------------------------------===//
namespace {
struct ConvertCIRToLLVMPass
    : public ConvertCIRToLLVMBase<ConvertCIRToLLVMPass> {
  ConvertCIRToLLVMPass() = default;
  ConvertCIRToLLVMPass(bool enableNanboxing) {
    this->enableNanboxing = enableNanboxing;
  }

  void runOnOperation() override;
};
} // namespace

void ConvertCIRToLLVMPass::runOnOperation() {
  ModuleOp module = getOperation();

  const auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();
  auto mlirDataLayout = dataLayoutAnalysis.getAtOrAbove(module);

  LowerToLLVMOptions options(&getContext(), mlirDataLayout);
  // Verify options
  bool isMachO = false;
  if (auto layoutAttr = module->getAttrOfType<StringAttr>(
          LLVM::LLVMDialect::getDataLayoutAttrName())) {
    if (failed(LLVM::LLVMDialect::verifyDataLayoutString(
            layoutAttr.getValue(), [this](const Twine &message) {
              getOperation().emitError() << message.str();
            }))) {
      signalPassFailure();
      return;
    }
    auto layout = layoutAttr.getValue();
    isMachO = layout.contains("m:o"); // mach-o mangling scheme
    options.dataLayout = llvm::DataLayout(layout);
  }

  // Define the conversion target for this lowering, which is the LLVM dialect
  LLVMConversionTarget target(getContext());
  target.addLegalOp<ModuleOp>();

  // We need to lower our custom types to their LLVM representation, so we
  // use a custom TypeConverter for this. We also use this type converter with
  // other dialects which we use and delegate handling of their types to an
  // internal type converter provided by MLIR
  CIRTypeConverter typeConverter(&getContext(), enableNanboxing, isMachO,
                                 options, &dataLayoutAnalysis);

  // We need to provide the set of rewrite patterns which will lower CIR -
  // as well as ops from other dialects we use - to the LLVM dialect.
  // Specifically, we expect to have a combination of `cir`, `func`, `cf`
  // and `scf` ops. We populate the rewriter with patterns provided by those
  // dialects for lowering to LLVM, and extend the set with our own rewrite
  // patterns.
  //
  // In some cases we may rely on transitive lowerings, e.g. cir -> cf ->
  // llvm, or cir -> scf -> cf -> llvm, in order to transfer all illegal ops
  // to legal ones in the LLVM dialect.
  RewritePatternSet patterns(&getContext());
  populateSCFToControlFlowConversionPatterns(patterns);
  cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
  populateFuncToLLVMConversionPatterns(typeConverter, patterns);

  // We apply our pattern rewrites first
  populateGeneratedPDLLPatterns(patterns);
  // These are the conversion patterns for CIR ops
  patterns.add<DispatchTableOpLowering>(typeConverter);
  patterns.add<ConstantOpLowering>(typeConverter);
  patterns.add<ConstantNullOpLowering>(typeConverter);
  patterns.add<CastOpLowering>(typeConverter);
  patterns.add<CallOpLowering>(typeConverter);
  patterns.add<IsNullOpLowering>(typeConverter);
  patterns.add<AndOpLowering>(typeConverter);
  patterns.add<AndAlsoOpLowering>(typeConverter);
  patterns.add<OrOpLowering>(typeConverter);
  patterns.add<OrElseOpLowering>(typeConverter);
  patterns.add<XorOpLowering>(typeConverter);
  patterns.add<NotOpLowering>(typeConverter);
  patterns.add<TypeOfImmediateOpLowering>(typeConverter);
  patterns.add<TypeOfBoxOpLowering>(typeConverter);
  patterns.add<TypeOfOpLowering>(typeConverter);
  patterns.add<IsTypeOpLowering>(typeConverter);
  patterns.add<IsTaggedTupleOpLowering>(typeConverter);
  patterns.add<MallocOpLowering>(typeConverter);
  // patterns.add<CaptureFunOpLowering>(typeConverter);
  patterns.add<ConsOpLowering>(typeConverter);
  patterns.add<HeadOpLowering>(typeConverter);
  patterns.add<TailOpLowering>(typeConverter);
  patterns.add<TupleOpLowering>(typeConverter);
  patterns.add<SetElementOpLowering>(typeConverter);
  patterns.add<GetElementOpLowering>(typeConverter);
  patterns.add<RaiseOpLowering>(typeConverter);
  patterns.add<BuildStacktraceOpLowering>(typeConverter);
  patterns.add<ExceptionClassOpLowering>(typeConverter);
  patterns.add<ExceptionReasonOpLowering>(typeConverter);
  patterns.add<ExceptionTraceOpLowering>(typeConverter);
  patterns.add<YieldOpLowering>(typeConverter);
  patterns.add<RecvStartOpLowering>(typeConverter);
  patterns.add<RecvNextOpLowering>(typeConverter);
  patterns.add<RecvPeekOpLowering>(typeConverter);
  patterns.add<RecvPopOpLowering>(typeConverter);
  patterns.add<RecvDoneOpLowering>(typeConverter);
  // patterns.add<BinaryStartOpLowering>(typeConverter);
  // patterns.add<BinaryFinishOpLowering>(typeConverter);
  // patterns.add<BinaryPushIntegerOpLowering>(typeConverter);
  // patterns.add<BinaryPushFloatOpLowering>(typeConverter);
  // patterns.add<BinaryPushUtf8OpLowering>(typeConverter);
  // patterns.add<BinaryPushUtf16OpLowering>(typeConverter);
  // patterns.add<BinaryPushBitsOpLowering>(typeConverter);

  // When complete, we want to be lowered completely to LLVM dialect, so we're
  // applying this as a full conversion. Doing so means that when this pass
  // completes, only legal LLVM dialect ops will remain. If we forget to lower
  // an op that is illegal in LLVM, the pass will fail with an error pointing to
  // the guilty op.
  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

/// Create the conversion passs
std::unique_ptr<OperationPass<ModuleOp>>
mlir::cir::createConvertCIRToLLVMPass() {
  return std::make_unique<ConvertCIRToLLVMPass>();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::cir::createConvertCIRToLLVMPass(bool enableNanboxing) {
  return std::make_unique<ConvertCIRToLLVMPass>(enableNanboxing);
}
