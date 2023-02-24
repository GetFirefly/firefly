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
#include "mlir/Dialect/SCF/IR/SCF.h"
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

const uint64_t NANBOX_SIGN_BIT = 0x01ull << 63;
// Also the sign bit for Integer
const uint64_t NANBOX_SIGNAL_BIT = 0x01ull << 51;
// Also the tag for Integer
const uint64_t NANBOX_NEG_INFINITY = 0xFFFull << 52;
// Also the tag for Nil
const uint64_t NANBOX_INFINITY = NANBOX_NEG_INFINITY & ~NANBOX_SIGN_BIT;
// Infinity is also the minimal NaN value, non-canonical
const uint64_t NANBOX_NAN = NANBOX_INFINITY;
// Also the tag for None
const uint64_t NANBOX_CANONICAL_NAN = NANBOX_NEG_INFINITY >> 1;

const uint64_t NANBOX_INTEGER_MASK = ~NANBOX_NEG_INFINITY;
const uint64_t NANBOX_INTEGER_NEG = NANBOX_NEG_INFINITY | NANBOX_SIGNAL_BIT;
const uint64_t NANBOX_TAG_MASK = 0xFull;
const uint64_t NANBOX_PTR_MASK =
    ~(NANBOX_SIGN_BIT | NANBOX_CANONICAL_NAN | NANBOX_TAG_MASK);
// const uint64_t NANBOX_MANTISSA_MASK = !(NANBOX_CANONICAL_NAN |
// NANBOX_SIGN_BIT);
const uint64_t NANBOX_LITERAL_TAG = 0x1ull;
const uint64_t CONS_TAG = 0x4ull;
const uint64_t CONS_LITERAL_TAG = CONS_TAG | NANBOX_LITERAL_TAG;
const uint64_t IS_CONS = NANBOX_INFINITY | CONS_TAG;
const uint64_t IS_CONS_LITERAL = NANBOX_INFINITY | CONS_LITERAL_TAG;
const uint64_t TAG_MASK =
    NANBOX_CANONICAL_NAN | NANBOX_SIGN_BIT | NANBOX_TAG_MASK;

namespace TermKind {
enum Kind {
  Invalid = 0,
  None,
  Nil,
  Bool,
  Atom,
  Int,
  Float,
  Cons,
  Tuple,
  Map,
  Closure,
  Pid,
  Port,
  Reference,
  Binary,
};
}

//===----------------------------------------------------------------------===//
// Pattern Rewrites
//===----------------------------------------------------------------------===//
//#include "CIR/Patterns.cpp.inc"

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
      : LLVMTypeConverter(ctx, options, analysis), useMachOMangling(isMachO) {
    // All immediates are lowered to opaque term type, as they must be encoded
    addConversion([&](CIRNoneType) { return getTermType(); });
    addConversion([&](CIROpaqueTermType) { return getTermType(); });
    addConversion([&](CIRNumberType) { return getTermType(); });
    addConversion([&](CIRIntegerType) { return getTermType(); });
    addConversion([&](CIRFloatType) { return getTermType(); });
    addConversion([&](CIRAtomType) { return getTermType(); });
    addConversion([&](CIRBoolType) { return getTermType(); });
    addConversion([&](CIRIsizeType) { return getTermType(); });
    addConversion([&](CIRNilType) { return getTermType(); });
    // All boxed types are lowered to their pointee representations, but boxes
    // are opaque terms as we must encode pointers when lowering boxed types.
    // The boxed types themselves should _not_ be lowered to pointer types,
    // that's what PtrType/BoxType are for
    addConversion([&](CIRBoxType) { return getTermType(); });
    addConversion([&](CIRBigIntType) { return getBigIntType(); });
    addConversion([&](CIRConsType) { return getConsType(); });
    addConversion([&](CIRMapType) { return getMapType(); });
    addConversion([&](CIRBitsType) { return getBinaryDataType(); });
    addConversion([&](CIRBinaryType) { return getBinaryDataType(); });
    addConversion([&](CIRPidType) { return getTermType(); });
    addConversion([&](CIRPortType) { return getTermType(); });
    addConversion([&](CIRReferenceType) { return getTermType(); });
    addConversion([&](CIRFunType type) { return convertFunType(type); });
    addConversion([&](CIRExceptionType) { return getExceptionType(); });
    addConversion([&](CIRTraceType) { return getTraceType(); });
    addConversion([&](CIRRecvContextType) { return getRecvContextType(); });
    addConversion([&](CIRBinaryBuilderType) { return getBinaryBuilderType(); });
    addConversion([&](CIRMatchContextType) { return getMatchContextType(); });
    addConversion([&](CIRProcessType) { return getProcessType(); });
    // For times where we want to reify an unencoded pointer type, we use
    // PtrType not BoxType
    addConversion([&](PtrType type) {
      return LLVM::LLVMPointerType::get(convertType(type.getElementType()));
    });
    // The MLIR tuple type has no lowering out of the box, so we handle it
    addConversion([&](TupleType type) { return convertTupleType(type); });
  }

  bool isMachO() { return useMachOMangling; }

  bool isNanboxingEnabled() { return true; }

  Type getVoidType() { return LLVM::LLVMVoidType::get(&getContext()); }

  Type getI8Type() { return IntegerType::get(&getContext(), 8); }

  // The following get*Type functions are all used to get the LLVM
  // representation of either a built-in type or a CIR type, _not_ the named
  // type.

  Type getIsizeType() {
    return IntegerType::get(&getContext(), getPointerBitwidth());
  }

  Type getI64Type() { return IntegerType::get(&getContext(), 64); }

  // NOTE: This is likely to change, so do not depend on this representation.
  // To get stackmaps working for garbage collection, we're likely going to
  // switch to a pointer type in a non-zero address space, and when that
  // happens, we can't have terms somtimes represented as integers and sometimes
  // as pointers, or it will cause gc roots to be missed.
  Type getTermType() { return getI64Type(); }

  // Floats are immediates on nanboxed platforms, boxed types everywhere else
  Type getFloatType() { return Float64Type::get(&getContext()); }

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
    // *BigIntTerm { BigInt { val: BigUint { digits: Vec<isize> {
    // { *isize, isize }, isize } }, sign: i8, padding: [N x i8]}}
    auto isizeTy = getIsizeType();
    auto i8Ty = IntegerType::get(context, 8);
    auto bitwidth = getPointerBitwidth();
    auto paddingTy = LLVM::LLVMArrayType::get(i8Ty, (bitwidth / 8) - 1);
    auto isizePtrTy = LLVM::LLVMPointerType::get(getIsizeType());
    auto digitsInnerTy =
        LLVM::LLVMStructType::getLiteral(context, {isizePtrTy, isizeTy});
    auto digitsTy =
        LLVM::LLVMStructType::getLiteral(context, {digitsInnerTy, isizeTy});
    assert(succeeded(bigIntTy.setBody({digitsTy, i8Ty, paddingTy},
                                      /*packed=*/false)) &&
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
    MLIRContext *context = &getContext();
    auto isizeTy = getIsizeType();
    auto termTy = getTermType();
    auto elemsTy = LLVM::LLVMArrayType::get(termTy, arity);
    return LLVM::LLVMStructType::getLiteral(context, {isizeTy, elemsTy});
  }

  // This layout is intentionally incomplete, as we don't control the layout of
  // our map implementation, and currently all operations on them are done via
  // the runtime only
  Type getMapType() {
    MLIRContext *context = &getContext();
    auto mapTy = LLVM::LLVMStructType::getIdentified(context, "erlang::Map");
    if (mapTy.isInitialized())
      return mapTy;
    // *MapTerm { internal: opaque }
    auto opaqueTy =
        LLVM::LLVMStructType::getOpaque("rpds::HashTrieMap", context);
    assert(succeeded(mapTy.setBody({opaqueTy}, /*packed=*/false)) &&
           "failed to set body of map struct!");
    return mapTy;
  }

  // This layout matches what our runtime produces/expects for binaries
  Type getBinaryDataType() {
    MLIRContext *context = &getContext();
    auto bitsTy =
        LLVM::LLVMStructType::getIdentified(context, "erlang::BinaryData");
    if (bitsTy.isInitialized())
      return bitsTy;
    // *BinaryData { flags: isize, data: [? x i8] }
    auto isizeTy = getIsizeType();
    auto dataTy = LLVM::LLVMArrayType::get(IntegerType::get(context, 8), 0);
    assert(succeeded(bitsTy.setBody({isizeTy, dataTy}, /*packed=*/false)) &&
           "failed to set body of binarydata struct!");
    return bitsTy;
  }

  // This layout matches what our runtime produces/expects for bitstring slices
  Type getBitSliceType() {
    MLIRContext *context = &getContext();
    auto bitsTy =
        LLVM::LLVMStructType::getIdentified(context, "erlang::BitSlice");
    if (bitsTy.isInitialized())
      return bitsTy;
    // *BitSlice { owner: OpaqueTerm, data: { [0 x i8]*, isize }, offset: i8,
    // bit_size: isize }
    auto termTy = getTermType();
    auto isizeTy = getIsizeType();
    auto i8Ty = getI8Type();
    auto bytesTy = LLVM::LLVMArrayType::get(i8Ty, 0);
    auto bytesPtrTy = LLVM::LLVMPointerType::get(bytesTy);
    auto sliceTy =
        LLVM::LLVMStructType::getLiteral(context, {bytesPtrTy, isizeTy});
    auto slicePtrTy = LLVM::LLVMPointerType::get(sliceTy);
    auto bitwidth = getPointerBitwidth();
    auto paddingTy = LLVM::LLVMArrayType::get(i8Ty, (bitwidth / 8) - 1);
    assert(
        succeeded(bitsTy.setBody({termTy, slicePtrTy, i8Ty, paddingTy, isizeTy},
                                 /*packed=*/false)) &&
        "failed to set body of bitslice struct!");
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

    auto atomTy = LLVM::LLVMPointerType::get(getAtomDataType());
    auto arityTy = getIsizeType();
    auto bareFunTy = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(context), {}, /*vararg=*/false);
    auto funPtrTy = LLVM::LLVMPointerType::get(bareFunTy);
    auto envTy = LLVM::LLVMArrayType::get(getTermType(), 1);

    assert(
        succeeded(closureTy.setBody({atomTy, atomTy, arityTy, funPtrTy, envTy},
                                    /*packed=*/false)) &&
        "failed to set body of closure struct!");
    return closureTy;
  }

  // This function returns a type equivalent in representation to GcBox<T>
  Type getGcBoxType(Type innerTy) {
    MLIRContext *context = &getContext();
    auto gcBoxTy =
        LLVM::LLVMStructType::getIdentified(context, "firefly_alloc::GcBox");
    if (gcBoxTy.isInitialized())
      return gcBoxTy;

    auto metadataTy =
        LLVM::LLVMStructType::getIdentified(context, "firefly_alloc::Metadata");
    auto typeIdTy = getIsizeType();
    auto ptrMetadataTy = getIsizeType();
    assert(succeeded(metadataTy.setBody({typeIdTy, ptrMetadataTy},
                                        /*packed=*/false)) &&
           "failed to set body of ptr metadata struct!");
    assert(
        succeeded(gcBoxTy.setBody({metadataTy, innerTy}, /*packed=*/false)) &&
        "failed to set body of closure struct!");
    return gcBoxTy;
  }

  // This function represnts `firefly_rt::function::ErlangResult`, which is
  // used as the general return type of runtime functions that are fallible.
  //
  // NOTE: It is essential to understand the ABI implications of the types you
  // build with this; as the mechanism by which values of the resulting type get
  // passed/returned from functions changes.
  Type getResultType(Type okTy) {
    MLIRContext *context = &getContext();
    auto isizeTy = getIsizeType();
    return LLVM::LLVMStructType::getLiteral(context, {isizeTy, okTy});
  }

  // This function returns the target-specific exception representation
  Type getExceptionType() {
    MLIRContext *context = &getContext();
    auto exceptionTy =
        LLVM::LLVMStructType::getIdentified(context, "erlang::Exception");
    if (exceptionTy.isInitialized())
      return exceptionTy;

    // Corresponds to ErlangException in firefly_rt
    // { class: term, reason: term, trace: *mut Trace, fragment: *const
    // HeapFragment }
    Type termTy = getTermType();
    Type traceTy = LLVM::LLVMPointerType::get(getTraceType());
    Type i8Ty = getI8Type();
    Type i8PtrTy = LLVM::LLVMPointerType::get(i8Ty);
    assert(succeeded(exceptionTy.setBody({termTy, termTy, traceTy, i8PtrTy},
                                         /*packed=*/false)) &&
           "failed to set body of exception struct!");
    return exceptionTy;
  }

  // This function returns the type of handle used for the raw exception trace
  // reprsentation
  Type getTraceType() {
    MLIRContext *context = &getContext();
    auto traceTy =
        LLVM::LLVMStructType::getIdentified(context, "erlang::Trace");
    if (traceTy.isInitialized())
      return traceTy;

    auto isizeTy = getIsizeType();
    assert(succeeded(traceTy.setBody({isizeTy}, /*packed=*/false)) &&
           "failed to set body of trace struct!");
    return traceTy;
  }

  // Corresponds to Message in firefly_alloc
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

  // Corresponds to *mut BitVec in firefly_binary
  Type getBinaryBuilderType() {
    MLIRContext *context = &getContext();
    return LLVM::LLVMStructType::getOpaque("erlang::BitVec", context);
  }

  // Corresponds to *mut Matcher<'static>  in firefly_binary
  Type getMatchContextType() {
    MLIRContext *context = &getContext();
    auto matchCtxTy =
        LLVM::LLVMStructType::getIdentified(context, "erlang::Matcher");
    if (matchCtxTy.isInitialized())
      return matchCtxTy;
    auto isizeTy = getIsizeType();
    assert(succeeded(matchCtxTy.setBody({isizeTy}, /*packed=*/false)) &&
           "failed to set body of match context struct!");
    return matchCtxTy;
  }

  // Represents the result produced by bs_match
  Type getMatchResultType() {
    MLIRContext *context = &getContext();
    auto matchResultTy =
        LLVM::LLVMStructType::getIdentified(context, "erlang::MatchResult");
    if (matchResultTy.isInitialized())
      return matchResultTy;
    auto termTy = getTermType();
    auto matchCtxTy = LLVM::LLVMPointerType::get(getMatchContextType());
    assert(succeeded(
        matchResultTy.setBody({termTy, matchCtxTy}, /*packed=*/false)));
    return matchResultTy;
  }

  // Corresponds to ReceiveContext in firefly_rt_minimal
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

  // Corresponds to Process in firefly_rt
  Type getProcessType() {
    MLIRContext *context = &getContext();
    auto processTy =
        LLVM::LLVMStructType::getIdentified(context, "erlang::Process");
    if (processTy.isInitialized())
      return processTy;

    auto isizeTy = getIsizeType();
    assert(succeeded(processTy.setBody({isizeTy}, /*packed=*/false)) &&
           "failed to set body of process struct!");
    return processTy;
  }

  // Corresponds to AtomData in firefly_rt
  Type getAtomDataType() {
    MLIRContext *context = &getContext();
    auto atomDataTy =
        LLVM::LLVMStructType::getIdentified(context, "erlang::AtomData");
    if (atomDataTy.isInitialized())
      return atomDataTy;

    auto isizeTy = getIsizeType();
    auto i8Ty = getI8Type();
    auto i8PtrTy = LLVM::LLVMPointerType::get(i8Ty);
    assert(succeeded(atomDataTy.setBody({isizeTy, i8PtrTy}, /*packed=*/false)));
    return atomDataTy;
  }

  // Corresponds to FunctionSymbol in firefly_rt
  Type getDispatchEntryType() {
    MLIRContext *context = &getContext();
    auto dispatchEntryTy =
        LLVM::LLVMStructType::getIdentified(context, "erlang::FunctionSymbol");
    if (dispatchEntryTy.isInitialized())
      return dispatchEntryTy;

    // Corresponds to FunctionSymbol in firefly_rt
    // { module: *AtomData, function: *AtomData, arity: u8, fun: *const () }
    auto atomDataTy = getAtomDataType();
    auto atomDataPtrTy = LLVM::LLVMPointerType::get(atomDataTy);
    auto i8Ty = getI8Type();
    auto opaqueFnTy =
        LLVM::LLVMFunctionType::get(getVoidType(), ArrayRef<Type>{});
    auto opaqueFnPtrTy = LLVM::LLVMPointerType::get(opaqueFnTy);
    auto bitwidth = getPointerBitwidth();
    auto paddingTy = LLVM::LLVMArrayType::get(i8Ty, (bitwidth / 8) - 1);
    assert(succeeded(dispatchEntryTy.setBody(
        {atomDataPtrTy, atomDataPtrTy, i8Ty, paddingTy, opaqueFnPtrTy},
        /*packed=*/false)));
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
    auto lastIdx = calleeTy.getNumInputs();
    if (lastIdx > 0)
      lastIdx--;
    return convertType(
        calleeTy.getWithArgsAndResults({lastIdx}, {closurePtrTy}, {}, {}));
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
      : ConversionPattern(typeConverter, rootOpName, benefit, context){};

  using Pattern::getContext;

protected:
  // This is necessary in order to access our type converter
  CIRTypeConverter *getTypeConverter() const {
    return static_cast<CIRTypeConverter *>(
        ConversionPattern::getTypeConverter());
  }

  // The following functions Provide target-specific term encoding details

  unsigned getPointerBitwidth() const {
    return getTypeConverter()->getPointerBitwidth();
  }
  bool isNanboxingEnabled() const {
    return getTypeConverter()->isNanboxingEnabled();
  }
  bool isMachO() const { return getTypeConverter()->isMachO(); }

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
  Type getBinaryDataType() const {
    return getTypeConverter()->getBinaryDataType();
  }
  Type getBitSliceType() const { return getTypeConverter()->getBitSliceType(); }
  Type getClosureType() const { return getTypeConverter()->getClosureType(); }
  Type getResultType(Type okTy) const {
    return getTypeConverter()->getResultType(okTy);
  }
  Type getExceptionType() const {
    return getTypeConverter()->getExceptionType();
  }
  Type getGcBoxType(Type innerTy) const {
    return getTypeConverter()->getGcBoxType(innerTy);
  }
  Type getTraceType() const { return getTypeConverter()->getTraceType(); }
  Type getBinaryBuilderType() const {
    return getTypeConverter()->getBinaryBuilderType();
  }
  Type getMatchContextType() const {
    return getTypeConverter()->getMatchContextType();
  }
  Type getMatchResultType() const {
    return getTypeConverter()->getMatchResultType();
  }
  Type getRecvContextType() const {
    return getTypeConverter()->getRecvContextType();
  }
  Type getProcessType() const { return getTypeConverter()->getProcessType(); }
  Type getMessageType() const { return getTypeConverter()->getMessageType(); }
  Type getAtomDataType() const { return getTypeConverter()->getAtomDataType(); }
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

  // This function builds an LLVM::ConstantOp with an i8 value and result
  // type
  Value createI8Constant(OpBuilder &builder, Location loc, int8_t value) const {
    return builder.create<LLVM::ConstantOp>(loc, getI8Type(),
                                            builder.getI8IntegerAttr(value));
  }

  // This function builds an LLVM::ConstantOp with an isize value and result
  // type
  Value createI32Constant(OpBuilder &builder, Location loc,
                          int32_t value) const {
    return builder.create<LLVM::ConstantOp>(loc, getI32Type(),
                                            builder.getI32IntegerAttr(value));
  }

  // This function builds an LLVM::ConstantOp with an i64 value and result
  // type
  Value createI64Constant(OpBuilder &builder, Location loc,
                          int64_t value) const {
    return builder.create<LLVM::ConstantOp>(loc, getI64Type(),
                                            builder.getI64IntegerAttr(value));
  }

  // This function builds an LLVM::ConstantOp with an isize value and result
  // type
  Value createIsizeConstant(OpBuilder &builder, Location loc,
                            uint64_t value) const {
    auto isizeTy = getIsizeType();
    return builder.create<LLVM::ConstantOp>(
        loc, isizeTy, builder.getIntegerAttr(isizeTy, value));
  }

  // This function builds an LLVM::ConstantOp with a value of term type
  Value createTermConstant(OpBuilder &builder, Location loc,
                           uint64_t value) const {
    auto termTy = getTermType();
    return builder.create<LLVM::ConstantOp>(
        loc, termTy, builder.getIntegerAttr(termTy, value));
  }

  // This function builds an LLVM::GlobalOp representing a constnat value of the
  // given type with the provided name.
  //
  // It defaults to internal linkage/non-thread-local, with an empty
  // initializer.
  //
  // NOTE: It is the callers responsibility to ensure a global with the given
  // name isn't already defined.
  LLVM::GlobalOp insertGlobalConstantOp(
      OpBuilder &builder, Location loc, std::string name, Type ty,
      LLVM::Linkage linkage = LLVM::Linkage::Internal) const {
    return builder.create<LLVM::GlobalOp>(loc, ty, /*isConstant=*/true, linkage,
                                          name, Attribute());
  }

  // This function is used to construct an LLVM constant for Erlang integer
  // terms
  Value createIntegerConstant(OpBuilder &builder, Location loc,
                              uint64_t value) const {
    uint64_t neg = value & NANBOX_INTEGER_NEG;
    switch (neg) {
    case 0:
      return createTermConstant(builder, loc, value | NANBOX_NEG_INFINITY);
    case NANBOX_INTEGER_NEG:
      return createTermConstant(builder, loc, value);
    default:
      llvm::outs() << neg;
      llvm::outs() << "\n";
      assert(false && "invalid immediate integer constant, out of range");
    }
  }

  // This function is used to construct an LLVM constant for an Erlang float
  // term
  Value createFloatConstant(OpBuilder &builder, Location loc, APFloat value,
                            ModuleOp &module) const {
    // When nanboxed, floats have no tag, but cannot be NaN, as NaN bits are
    // used for term tagging
    assert(!value.isNaN() && "invalid floating point constant for target, "
                             "floats must not be NaN!");
    return createIntegerConstant(builder, loc,
                                 value.bitcastToAPInt().getZExtValue());
  }

  Value createBigIntConstant(OpBuilder &builder, Location loc, Sign sign,
                             StringRef digits, ModuleOp &module) const {
    llvm::SHA1 hasher;
    hasher.update((unsigned)sign);
    hasher.update(digits);
    auto hash = llvm::toHex(hasher.result(), true);
    auto globalName = std::string("bigint_") + hash;

    auto i8Ty = builder.getI8Type();
    auto i32Ty = builder.getI32Type();
    auto isizeTy = getIsizeType();
    auto termTy = getTermType();
    auto digitsTy = LLVM::LLVMArrayType::get(i8Ty, digits.size());
    auto emptyArrayTy = LLVM::LLVMArrayType::get(i8Ty, 0);
    auto dataTy = LLVM::LLVMStructType::getLiteral(builder.getContext(),
                                                   {i32Ty, digitsTy});
    auto genericDataTy =
        LLVM::LLVMPointerType::get(LLVM::LLVMStructType::getLiteral(
            builder.getContext(), {i32Ty, emptyArrayTy}));
    auto fatPtrTy = LLVM::LLVMStructType::getLiteral(builder.getContext(),
                                                     {genericDataTy, isizeTy});

    auto dataConst = module.lookupSymbol<LLVM::GlobalOp>(globalName);
    if (!dataConst) {
      PatternRewriter::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());

      dataConst = builder.create<LLVM::GlobalOp>(
          loc, dataTy, /*isConstant=*/true, LLVM::Linkage::LinkonceODR,
          globalName, Attribute(),
          /*alignment=*/8, /*addrspace=*/0);

      auto &initRegion = dataConst.getInitializerRegion();
      builder.createBlock(&initRegion);

      Value dataSign = createI32Constant(builder, loc, (unsigned)sign);
      Value dataRaw = builder.create<LLVM::ConstantOp>(
          loc, digitsTy, builder.getStringAttr(digits.str()));

      Value data = builder.create<LLVM::UndefOp>(loc, dataTy);
      data = builder.create<LLVM::InsertValueOp>(
          loc, data, dataSign, builder.getDenseI64ArrayAttr(0));
      data = builder.create<LLVM::InsertValueOp>(
          loc, data, dataRaw, builder.getDenseI64ArrayAttr(1));
      builder.create<LLVM::ReturnOp>(loc, data);
    }

    Value ptr = builder.create<LLVM::AddressOfOp>(loc, dataConst);
    Value genericPtr = builder.create<LLVM::BitcastOp>(loc, genericDataTy, ptr);
    Value fatPtr = builder.create<LLVM::UndefOp>(loc, fatPtrTy);
    Value dataSize = createIsizeConstant(builder, loc, digits.size());
    fatPtr = builder.create<LLVM::InsertValueOp>(
        loc, fatPtr, genericPtr, builder.getDenseI64ArrayAttr(0));
    fatPtr = builder.create<LLVM::InsertValueOp>(
        loc, fatPtr, dataSize, builder.getDenseI64ArrayAttr(1));

    Operation *callee = module.lookupSymbol("__firefly_bigint_from_digits");
    if (!callee) {
      auto calleeType =
          LLVM::LLVMFunctionType::get(termTy, ArrayRef<Type>{fatPtrTy});
      insertFunctionDeclaration(builder, loc, module,
                                "__firefly_bigint_from_digits", calleeType);
    }

    Operation *call = builder.create<LLVM::CallOp>(
        loc, TypeRange({termTy}), "__firefly_bigint_from_digits",
        ValueRange({fatPtr}));
    return call->getResult(0);
  }

  Value createBinaryDataConstant(OpBuilder &builder, Location loc,
                                 StringRef value, bool isUtf8,
                                 ModuleOp &module) const {
    llvm::SHA1 hasher;
    hasher.update(value);
    auto hash = llvm::toHex(hasher.result(), true);
    auto globalName = std::string("binary_") + hash;

    auto isizeTy = getIsizeType();
    auto charsTy = LLVM::LLVMArrayType::get(builder.getI8Type(), value.size());
    auto dataTy = LLVM::LLVMStructType::getLiteral(builder.getContext(),
                                                   {isizeTy, charsTy});

    auto dataConst = module.lookupSymbol<LLVM::GlobalOp>(globalName);
    if (!dataConst) {
      PatternRewriter::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());

      dataConst = builder.create<LLVM::GlobalOp>(
          loc, dataTy, /*isConstant=*/false, LLVM::Linkage::LinkonceODR,
          globalName, Attribute(),
          /*alignment=*/8, /*addrspace=*/0);

      auto &initRegion = dataConst.getInitializerRegion();
      builder.createBlock(&initRegion);

      uint64_t size = value.size();
      uint64_t flags = (size << 4) | 0x08;
      if (isUtf8) {
        flags |= 0x04;
      } else {
        flags |= 0x01;
      }

      Value dataFlags = createIsizeConstant(builder, loc, flags);
      Value dataRaw = builder.create<LLVM::ConstantOp>(
          loc, charsTy, builder.getStringAttr(value.str()));

      Value data = builder.create<LLVM::UndefOp>(loc, dataTy);
      data = builder.create<LLVM::InsertValueOp>(
          loc, data, dataFlags, builder.getDenseI64ArrayAttr(0));
      data = builder.create<LLVM::InsertValueOp>(
          loc, data, dataRaw, builder.getDenseI64ArrayAttr(1));
      builder.create<LLVM::ReturnOp>(loc, data);
    }

    Value ptr = builder.create<LLVM::AddressOfOp>(loc, dataConst);
    return ptr;
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
      return createTermConstant(builder, loc, NANBOX_CANONICAL_NAN | 0x02);
    else if (name == "true")
      return createTermConstant(builder, loc, NANBOX_CANONICAL_NAN | 0x03);

    auto ptr = createAtomDataGlobal(builder, loc, module, name);
    return encodeAtomPtr(builder, loc, ptr);
  }

  // Used like `createAtom`, but for situations in which we are not encoding as
  // a term
  Value createAtomData(OpBuilder &builder, Location loc, StringRef name,
                       ModuleOp &module) const {
    if (name == "false") {
      auto dataTy = LLVM::LLVMPointerType::get(getAtomDataType());
      Value zero = createIsizeConstant(builder, loc, 0);
      return builder.create<LLVM::IntToPtrOp>(loc, dataTy, zero);
    } else if (name == "true") {
      auto dataTy = LLVM::LLVMPointerType::get(getAtomDataType());
      Value one = createIsizeConstant(builder, loc, 1);
      return builder.create<LLVM::IntToPtrOp>(loc, dataTy, one);
    }

    return createAtomDataGlobal(builder, loc, module, name);
  }

  // This function constructs an AtomData record as a global constant,
  // referencing the atom data as a separate global containing a null-terminated
  // string. Returns the address to the data record
  Value createAtomDataGlobal(OpBuilder &builder, Location loc, ModuleOp &module,
                             StringRef value) const {
    // Hash the atom if it is not a bare atom, as the atom value may not be
    // representable as a symbol, but we want them to nevertheless have a
    // consistent unique id. By allowing bare atoms to use their value as part
    // of the symbol name, we can also manually define atoms in firefly_rt
    // which are used by the runtime for comparisons.
    auto notfound = ~size_t(0);
    bool isBareAtom = value.find_if_not([](char c) {
      return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
             (c >= '0' && c <= '9') || c == '_' || c == '@';
    }) == notfound;

    std::string globalName;
    std::string hash;
    if (!isBareAtom) {
      llvm::SHA1 hasher;
      hasher.update(value);
      hash = llvm::toHex(hasher.result(), true);
      globalName = std::string("atom_") + hash;
    } else {
      hash = value.str();
      globalName = std::string("atom_") + value.str();
    }

    auto dataConst = module.lookupSymbol<LLVM::GlobalOp>(globalName);
    if (!dataConst) {
      PatternRewriter::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());

      auto dataTy = getAtomDataType();

      std::string sectionName;
      if (isMachO())
        sectionName = std::string("__DATA,__atoms");
      else
        sectionName = std::string("__atoms");
      auto sectionAttr =
          builder.getNamedAttr("section", builder.getStringAttr(sectionName));

      dataConst = builder.create<LLVM::GlobalOp>(
          loc, dataTy, /*isConstant=*/false, LLVM::Linkage::LinkonceODR,
          globalName, Attribute(),
          /*alignment=*/8, /*addrspace=*/0, /*dso_local=*/false,
          /*thread_local=*/false, ArrayRef<NamedAttribute>{sectionAttr});

      auto &initRegion = dataConst.getInitializerRegion();
      builder.createBlock(&initRegion);

      Value dataSize = createIsizeConstant(builder, loc, value.size());
      Value dataPtr =
          createCStringGlobal(builder, loc, module, hash, value, {});

      Value data = builder.create<LLVM::UndefOp>(loc, dataTy);
      data = builder.create<LLVM::InsertValueOp>(
          loc, data, dataSize, builder.getDenseI64ArrayAttr(0));
      data = builder.create<LLVM::InsertValueOp>(
          loc, data, dataPtr, builder.getDenseI64ArrayAttr(1));
      builder.create<LLVM::ReturnOp>(loc, data);
    }

    return builder.create<LLVM::AddressOfOp>(loc, dataConst);
  }

  // This function constructs a global string constant with a
  // given name and value.
  //
  // The name is optional, as a name will be generated if not provided.
  // The value will not be null-terminated.
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
    auto data = value.str();

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
          globalName, valueAttr,
          /*alignment=*/0, /*addrspace=*/0, /*dso_local=*/false,
          /*thread_local=*/false, attrs);
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
    Value tag =
        createTermConstant(builder, loc, NANBOX_INFINITY | (uint64_t)0x01);
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
    Value tag = createTermConstant(
        builder, loc, NANBOX_INFINITY | (uint64_t)(isLiteral ? 0x05 : 0x04));
    Value valueAsInt = builder.create<LLVM::PtrToIntOp>(loc, termTy, value);
    return builder.create<LLVM::OrOp>(loc, valueAsInt, tag);
  }

  /// This function encodes a pointer as a GcBox<T>, which requires that it was
  /// originally allocated via GcBox, and is non-null. It is up to the caller to
  /// guarantee these properties
  Value encodeGcBoxPtr(OpBuilder &builder, Location loc, Value value) const {
    // TODO: Possibly use ptrmask intrinsic
    auto termTy = getTermType();
    Value tag = createTermConstant(builder, loc, NANBOX_INFINITY);
    Value valueAsInt = builder.create<LLVM::PtrToIntOp>(loc, termTy, value);
    return builder.create<LLVM::OrOp>(loc, valueAsInt, tag);
  }

  /// Same as encodeListPtr, but for tuples
  Value encodeTuplePtr(OpBuilder &builder, Location loc, Value value,
                       bool isLiteral = false) const {
    // TODO: Possibly use ptrmask intrinsic
    auto termTy = getTermType();
    Value tag = createTermConstant(
        builder, loc, NANBOX_INFINITY | (uint64_t)(isLiteral ? 0x07 : 0x06));
    Value valueAsInt = builder.create<LLVM::PtrToIntOp>(loc, termTy, value);
    return builder.create<LLVM::OrOp>(loc, valueAsInt, tag);
  }

  /// This function handles encoding a pointer to AtomData as an atom immediate
  Value encodeAtomPtr(OpBuilder &builder, Location loc, Value value) const {
    auto termTy = getTermType();
    Value valueAsInt = builder.create<LLVM::PtrToIntOp>(loc, termTy, value);
    Value tag =
        createTermConstant(builder, loc, NANBOX_CANONICAL_NAN | (uint64_t)0x02);
    return builder.create<LLVM::OrOp>(loc, valueAsInt, tag);
  }

  /// Floats are always encoded by a simple bitcast
  ///
  /// NOTE: This is only valid if the float is not NaN, or one of the infinities
  Value encodeFloat(OpBuilder &builder, Location loc, Value value) const {
    auto termTy = getTermType();
    return builder.create<LLVM::BitcastOp>(loc, termTy, value);
  }

  /// Integers must fit in the mantissa bits of a 64-bit float and are tagged
  /// by setting the sign + exponent bits to 1
  Value encodeInteger(OpBuilder &builder, Location loc, Value value) const {
    Value zero = createTermConstant(builder, loc, 0);
    Value isNeg = builder.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::slt,
                                               value, zero);
    Value negTag = createTermConstant(builder, loc, NANBOX_INTEGER_NEG);
    Value posTag = createTermConstant(builder, loc, NANBOX_NEG_INFINITY);
    Value tag = builder.create<LLVM::SelectOp>(loc, isNeg, negTag, posTag);
    Value mask = createTermConstant(builder, loc, !NANBOX_INTEGER_NEG);
    Value masked = builder.create<LLVM::AndOp>(loc, value, mask);
    return builder.create<LLVM::OrOp>(loc, masked, tag);
  }

  Value decodeInteger(OpBuilder &builder, Location loc, Value value) const {
    // See opaque.rs in firefly_rt
    auto mask = createTermConstant(builder, loc, NANBOX_INTEGER_MASK);
    Value raw = builder.create<LLVM::AndOp>(loc, value, mask);
    auto signal = createTermConstant(builder, loc, NANBOX_SIGNAL_BIT);
    Value signExtract = builder.create<LLVM::AndOp>(loc, raw, signal);
    Value isNeg = builder.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq,
                                               signExtract, signal);
    Value neg = createTermConstant(builder, loc, NANBOX_INTEGER_NEG);
    Value sign = builder.create<LLVM::MulOp>(loc, isNeg, neg);

    return builder.create<LLVM::OrOp>(loc, raw, sign);
  }

  // This function is intended to be a general purpose pointer decoder
  // for term values, i.e. it masks out the bits of an opaque term that
  // are used solely for tagging in all pointer types currently supported
  //
  // The caller must guarantee the following:
  //
  // * The given value is a boxed term, i.e. an encoded pointer
  // * The pointer produced by decoding the box is a valid pointer
  // * The pointee value can be dereferenced as a value of type `pointee`
  //
  // NOTE: This function strips all box tags, so code that cares about whether
  // the value is literal or not must perform that check separately.
  Value decodePtr(OpBuilder &builder, Location loc, Type pointee,
                  Value box) const {
    // NOTE: Possibly use ptrmask intrinsic
    auto ptrTy = LLVM::LLVMPointerType::get(pointee);
    // Need to untag the pointer first
    auto mask = createTermConstant(builder, loc, NANBOX_PTR_MASK);
    Value untagged = builder.create<LLVM::AndOp>(loc, box, mask);
    return builder.create<LLVM::IntToPtrOp>(loc, ptrTy, untagged);
  }

  // An alias for decodePtr with a type representing the cons cell layout.
  Value decodeListPtr(OpBuilder &builder, Location loc, Value box) const {
    return decodePtr(builder, loc, getConsType(), box);
  }

  Value decodeGcBoxPtr(OpBuilder &builder, Location loc, Type pointee,
                       Value box) const {
    return decodePtr(builder, loc, pointee, box);
  }

  LLVM::LLVMFuncOp
  insertFunctionDeclaration(OpBuilder &builder, Location loc, ModuleOp module,
                            StringRef name, LLVM::LLVMFunctionType type,
                            ArrayRef<NamedAttribute> attrs = {},
                            ArrayRef<DictionaryAttr> argAttrs = {}) const {
    PatternRewriter::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToEnd(module.getBody());
    return builder.create<LLVM::LLVMFuncOp>(loc, name, type,
                                            LLVM::Linkage::External, false,
                                            LLVM::CConv::C, attrs, argAttrs);
  }
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

    // Determine what type of constant this is and lower appropriately
    auto loc = op.getLoc();
    auto attr = adaptor.getValue();
    auto module = op->getParentOfType<ModuleOp>();
    auto replacement =
        TypeSwitch<Type, Value>(resultType)
            .Case<IntegerType>([&](IntegerType ty) {
              return rewriter.create<LLVM::ConstantOp>(loc, ty, attr);
            })
            .Case<IndexType>([&](IndexType ty) {
              auto isizeTy = getIsizeType();
              auto value = attr.cast<IntegerAttr>().getValue();
              return rewriter.create<LLVM::ConstantOp>(
                  loc, isizeTy,
                  rewriter.getIntegerAttr(isizeTy, value.getLimitedValue()));
            })
            .Case<FloatType>([&](FloatType ty) {
              return rewriter.create<LLVM::ConstantOp>(loc, ty,
                                                       attr.cast<FloatAttr>());
            })
            .Case<CIRNoneType>([&](CIRNoneType) {
              return createTermConstant(rewriter, loc, NANBOX_CANONICAL_NAN);
            })
            .Case<CIRNilType>([&](CIRNilType) {
              assert(NANBOX_INFINITY != 0);
              return createTermConstant(rewriter, loc, NANBOX_INFINITY);
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
              bool isTrue = attr.cast<CIRBoolAttr>().getValue();
              return createTermConstant(rewriter, loc,
                                        NANBOX_CANONICAL_NAN |
                                            (uint64_t)(isTrue ? 0x03 : 0x02));
            })
            .Case<CIRBoxType>([&](CIRBoxType boxTy) {
              return TypeSwitch<Type, Value>(boxTy.getElementType())
                  .Case<CIRBitsType>([&](CIRBitsType) {
                    StringRef str = attr.cast<StringAttr>().getValue();
                    bool isUtf8 =
                        op->getAttrOfType<BoolAttr>("utf8").getValue();
                    return createBinaryDataConstant(rewriter, loc, str, isUtf8,
                                                    module);
                  })
                  .Case<CIRBigIntType>([&](CIRBigIntType) {
                    auto bigIntAttr = attr.cast<BigIntAttr>();
                    return createBigIntConstant(rewriter, loc,
                                                bigIntAttr.getSign(),
                                                bigIntAttr.getDigits(), module);
                  })
                  .Default([](Type) { return nullptr; });
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
      Value none = createTermConstant(rewriter, loc, NANBOX_CANONICAL_NAN);
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
    auto newOp = rewriter.replaceOpWithNewOp<func::CallOp>(
        op, adaptor.getCallee(), calleeType.getResults(),
        adaptor.getOperands());
    newOp->setAttrs(op->getAttrs());
    return success();
  }
};

//===---------===//
// EnterOp
//===---------===//
struct EnterOpLowering : public ConvertCIROpToLLVMPattern<cir::EnterOp> {
  using ConvertCIROpToLLVMPattern<cir::EnterOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::EnterOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    Operation *callee = module.lookupSymbol(adaptor.getCallee());

    Type packedResult = nullptr;
    if (isa<func::FuncOp>(callee)) {
      auto func = cast<func::FuncOp>(callee);
      auto calleeType = func.getFunctionType();
      unsigned numResults = func.getNumResults();
      auto resultTypes = llvm::to_vector<4>(calleeType.getResults());
      if (numResults != 0)
        if (!(packedResult =
                  this->getTypeConverter()->packFunctionResults(resultTypes)))
          return failure();
    } else {
      auto func = cast<LLVM::LLVMFuncOp>(callee);
      auto resultTypes = func.getResultTypes();
      if (resultTypes.size() > 0)
        packedResult = resultTypes.front();
    }

    auto promoted = this->getTypeConverter()->promoteOperands(
        loc, op->getOperands(), adaptor.getOperands(), rewriter);
    auto newOp = rewriter.create<LLVM::CallOp>(
        loc, packedResult ? TypeRange(packedResult) : TypeRange(), promoted,
        op->getAttrs());
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, newOp->getResult(0));
    return success();
  }
};

//===---------===//
// CallIndirectOp
//===---------===//
/*
struct CallIndirectOpLowering
    : public ConvertCIROpToLLVMPattern<cir::CallIndirectOp> {
  using ConvertCIROpToLLVMPattern<
      cir::CallIndirectOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::CallIndirectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto callee = adaptor.getCallee();
    auto funTy = getClosureType(0);
    auto funPtrTy = LLVM::LLVMPointerType::get(funTy);

    // 1. Cast the term value to a GcBox<Closure> pointer
    Value ptr = decodeGcBoxPtr(rewriter, loc, funTy, callee)
    // 2. Extract the env arity from the GcBox metadata
    Value zero = createIsizeConstant(builder, loc, 0);
    Value metadataPtr = rewriter.create<LLVM::GEPOp>(loc, isizeTy, ptr,
                                                     ValueRange({zero, one}));
    Value envArity = rewriter.create<LLVM::LoadOp>(loc, metadataPtr);
    // 3. Fetch the actual function pointer from the closure
    Value one = createIsizeConstant(builder, loc, 1);
    Value three = createIndexAttrConstant(builder, loc, i32Ty, 3);
    Value funPtr = builder.create<LLVM::GEPOp>(loc, bareFnPtrTy, ptr,
                                               ValueRange({one, three}));
    Value bareFun = builder.create<LLVM::LoadOp>(loc, funPtr);
    // 4. Fetch the env pointer
    Value four = createIndexAttrConstant(builder, loc, i32Ty, 4);
    Value envBasePtr = builder.create<LLVM::GEPOp>(loc, envPtrTy, ptr,
                                                   ValueRange({one, four}));
    Value envBase = builder.create<LLVM::LoadOp>(loc, envBasePtr);
    // 5. Extract the env values

    // 4. Cast the inner function pointer to the appropriate type
    // 5. Call the function passing all original operands + env

    rewriter.replaceOpWithNewOp<func::CallOp>(op, adaptor.getCallee(),
resultTypes, adaptor.getOperands()); return success();
  }
};
*/
//===---------===//
// CastOp
//===---------===//
struct CastOpLowering : public ConvertCIROpToLLVMPattern<cir::CastOp> {
  using ConvertCIROpToLLVMPattern<cir::CastOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::CastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto inputs = adaptor.getInputs();
    auto inputTypes = op.getInputTypes();
    auto outputTypes = op.getResultTypes();

    SmallVector<Value, 1> results;
    for (auto it : llvm::enumerate(inputTypes)) {
      Value input = inputs[it.index()];
      Type inputType = it.value();
      Type outputType = outputTypes[it.index()];

      if (inputType == outputType) {
        results.push_back(input);
        continue;
      }

      // Casts from concrete term type to opaque term type are no-ops
      if (inputType.isa<TermType>() && outputType.isa<CIROpaqueTermType>()) {
        results.push_back(input);
        continue;
      }

      // Casts from opaque term type to any term type are no-ops
      if (inputType.isa<CIROpaqueTermType>() && outputType.isa<TermType>()) {
        results.push_back(input);
        continue;
      }

      // Casts from small integer to generic integer or number type;
      // as well as casts from float to number are no-ops
      if (inputType.isa<CIRIsizeType>()) {
        if (outputType.isa<CIRIntegerType>() ||
            outputType.isa<CIRNumberType>()) {
          results.push_back(input);
          continue;
        }
      }
      if (inputType.isa<CIRFloatType>() && outputType.isa<CIRNumberType>()) {
        results.push_back(input);
        continue;
      }

      // Casts from opaque term type to certain types produced by primops
      // are simple direct bitcasts, as the value is not actually a term
      if (inputType.isa<CIROpaqueTermType>()) {
        // Casts from opaque term type to i64 are intended as bitcasts to allow
        // working with terms as native integers, but are no-ops since terms are
        // actually integers at the LLVM level
        if (outputType.isInteger(64)) {
          results.push_back(input);
          continue;
        }

        if (auto ptrTy = outputType.dyn_cast_or_null<PtrType>()) {
          auto innerTy = ptrTy.getElementType();
          if (innerTy.isa<CIRBinaryBuilderType>()) {
            auto bvTy = getBinaryBuilderType();
            auto bvPtrTy = LLVM::LLVMPointerType::get(bvTy);
            results.push_back(
                rewriter.create<LLVM::IntToPtrOp>(loc, bvPtrTy, input));
            continue;
          }

          if (innerTy.isa<CIRMatchContextType>()) {
            auto ctxTy = getMatchContextType();
            auto ctxPtrTy = LLVM::LLVMPointerType::get(ctxTy);
            results.push_back(
                rewriter.create<LLVM::IntToPtrOp>(loc, ctxPtrTy, input));
            continue;
          }

          if (innerTy.isa<CIRExceptionType>()) {
            auto excTy = getExceptionType();
            auto excPtrTy = LLVM::LLVMPointerType::get(excTy);
            results.push_back(
                rewriter.create<LLVM::IntToPtrOp>(loc, excPtrTy, input));
            continue;
          }
        }
      }

      // Casts from Erlang primitives to LLVM primitives
      if (inputType.isa<CIRBoolType>() && outputType.isInteger(1)) {
        // To cast from a boolean term to its value, we know that we can
        // truncate to i1 due to how true/false terms are encoded
        results.push_back(
            rewriter.create<LLVM::TruncOp>(loc, getI1Type(), input));
        continue;
      }

      // Casts from Erlang small integers to native integers imply decoding of
      // the integer value
      if (inputType.isa<CIRIsizeType>() && outputType.isa<IntegerType>()) {
        IntegerType intTy = outputType.cast<IntegerType>();
        Value integer = decodeInteger(rewriter, loc, input);
        if (intTy.getWidth() == 64) {
          results.push_back(integer);
        } else {
          results.push_back(
              rewriter.create<LLVM::TruncOp>(loc, outputType, input));
        }
        continue;
      }

      // Casts from LLVM primitives to Erlang primitives
      if (inputType.isInteger(1) &&
          outputType.isa<CIRBoolType, CIROpaqueTermType>()) {
        // To cast from i1 to a boolean term, we treat the value as the symbol
        // id, zext and encode as an atom
        Value symbol = rewriter.create<LLVM::ZExtOp>(loc, getTermType(), input);
        Value tag =
            createTermConstant(rewriter, loc, NANBOX_CANONICAL_NAN | 0x02);
        results.push_back(rewriter.create<LLVM::OrOp>(loc, symbol, tag));
        continue;
      }

      // No other casts are supported currently
      inputType.dump();
      llvm::outs() << "\n";
      outputType.dump();
      llvm::outs() << "\n";
      return rewriter.notifyMatchFailure(
          op,
          "failed to lower cast, unsupported source/target type combination");
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
    auto value = adaptor.getValue();
    Value nullValue = rewriter.create<LLVM::NullOp>(loc, value.getType());
    rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(op, LLVM::ICmpPredicate::eq,
                                              value, nullValue);
    return success();
  }
};

//===---------===//
// TruncOp
//===---------===//
struct TruncOpLowering : public ConvertCIROpToLLVMPattern<cir::TruncOp> {
  using ConvertCIROpToLLVMPattern<cir::TruncOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::TruncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ty = convertType(op.getResult().getType());
    auto value = adaptor.getValue();
    rewriter.replaceOpWithNewOp<LLVM::TruncOp>(op, ty, value);
    return success();
  }
};

//===---------===//
// ZExtOp
//===---------===//
struct ZExtOpLowering : public ConvertCIROpToLLVMPattern<cir::ZExtOp> {
  using ConvertCIROpToLLVMPattern<cir::ZExtOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::ZExtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ty = convertType(op.getResult().getType());
    auto value = adaptor.getValue();
    rewriter.replaceOpWithNewOp<LLVM::ZExtOp>(op, ty, value);
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
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();

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
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();

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
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();

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
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();

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
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();

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
    auto value = adaptor.getValue();

    auto i1Ty = rewriter.getI1Type();

    Value constTrue = createBoolConstant(rewriter, loc, true);
    rewriter.replaceOpWithNewOp<LLVM::XOrOp>(op, i1Ty,
                                             ValueRange({value, constTrue}));
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
    auto i32Ty = getI32Type();
    auto termTy = getTermType();
    auto module = op->getParentOfType<ModuleOp>();

    Operation *callee = module.lookupSymbol("__firefly_builtin_typeof");
    if (!callee) {
      auto calleeType =
          LLVM::LLVMFunctionType::get(i32Ty, ArrayRef<Type>{termTy});
      insertFunctionDeclaration(rewriter, loc, module,
                                "__firefly_builtin_typeof", calleeType);
    }

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange({i32Ty}),
                                              "__firefly_builtin_typeof",
                                              ValueRange({adaptor.getValue()}));
    return success();
  }
};

//===---------===//
// IsListOp
//===---------===//
struct IsListOpLowering : public ConvertCIROpToLLVMPattern<cir::IsListOp> {
  using ConvertCIROpToLLVMPattern<cir::IsListOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::IsListOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto i1Ty = getI1Type();

    auto input = adaptor.getValue();
    //#[inline]
    // pub fn is_list(self) -> bool {
    //  const IS_CONS: u64 = INFINITY | CONS_TAG;
    //  const IS_CONS_LITERAL: u64 = INFINITY | CONS_LITERAL_TAG;
    //
    //    match self.0 & (NAN | SIGN_BIT | TAG_MASK) {
    //        IS_CONS | IS_CONS_LITERAL => true,
    //        _ => self.0 == NIL,
    //    }
    //}

    auto mask = createTermConstant(rewriter, loc, TAG_MASK);
    Value masked = rewriter.create<LLVM::AndOp>(loc, input, mask);
    // Is the masked value a boxed cons cell
    Value consTag = createTermConstant(rewriter, loc, IS_CONS);
    Value isCons = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq,
                                                 masked, consTag);
    // Is the masked value a literal cons cell
    Value consLiteralTag = createTermConstant(rewriter, loc, IS_CONS_LITERAL);
    Value isConsLiteral = rewriter.create<LLVM::ICmpOp>(
        loc, LLVM::ICmpPredicate::eq, masked, consLiteralTag);
    // Is the untagged value nil
    Value nilValue = createTermConstant(rewriter, loc, NANBOX_INFINITY);
    Value isNil = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq,
                                                input, nilValue);
    Value isNonEmpty = rewriter.create<LLVM::OrOp>(loc, isCons, isConsLiteral);
    Value isList = rewriter.create<LLVM::OrOp>(loc, isNonEmpty, isNil);

    rewriter.replaceOpWithNewOp<LLVM::TruncOp>(op, i1Ty, isList);
    return success();
  }
};

//===---------===//
// IsNonEmptyListOp
//===---------===//
struct IsNonEmptyListOpLowering
    : public ConvertCIROpToLLVMPattern<cir::IsNonEmptyListOp> {
  using ConvertCIROpToLLVMPattern<
      cir::IsNonEmptyListOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::IsNonEmptyListOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // Same as IsListOp, but without the check for nil
    auto mask = createTermConstant(rewriter, loc, TAG_MASK);
    Value masked = rewriter.create<LLVM::AndOp>(loc, adaptor.getValue(), mask);
    // Is the tag value a boxed cons cell
    Value consTag = createTermConstant(rewriter, loc, IS_CONS);
    Value isCons = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq,
                                                 masked, consTag);
    // Is the tag value a literal cons cell
    Value consLiteralTag = createTermConstant(rewriter, loc, IS_CONS_LITERAL);
    Value isConsLiteral = rewriter.create<LLVM::ICmpOp>(
        loc, LLVM::ICmpPredicate::eq, masked, consLiteralTag);

    rewriter.replaceOpWithNewOp<LLVM::OrOp>(op, isCons, isConsLiteral);
    return success();
  }
};

//===---------===//
// IsTupleOp
//===---------===//
struct IsTupleOpLowering : public ConvertCIROpToLLVMPattern<cir::IsTupleOp> {
  using ConvertCIROpToLLVMPattern<cir::IsTupleOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::IsTupleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto i1Ty = getI1Type();
    auto i8Ty = getI8Type();
    auto i32Ty = getI32Type();
    auto termTy = getTermType();
    auto module = op->getParentOfType<ModuleOp>();

    auto resultTy =
        LLVM::LLVMStructType::getLiteral(getContext(), {i8Ty, i32Ty});

    Operation *callee = module.lookupSymbol("__firefly_builtin_is_tuple");
    if (!callee) {
      auto calleeType =
          LLVM::LLVMFunctionType::get(resultTy, ArrayRef<Type>{termTy});
      insertFunctionDeclaration(rewriter, loc, module,
                                "__firefly_builtin_is_tuple", calleeType);
    }

    Operation *call = rewriter.create<LLVM::CallOp>(
        loc, TypeRange({resultTy}), "__firefly_builtin_is_tuple",
        ValueRange({adaptor.getValue()}));
    Value result = call->getResult(0);

    Value isTupleWide = rewriter.create<LLVM::ExtractValueOp>(
        loc, result, ArrayRef<int64_t>{0});
    Value arity = rewriter.create<LLVM::ExtractValueOp>(loc, result,
                                                        ArrayRef<int64_t>{1});
    Value isTuple = rewriter.create<LLVM::TruncOp>(loc, i1Ty, isTupleWide);

    rewriter.replaceOp(op, ValueRange({isTuple, arity}));
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
    auto i1Ty = getI1Type();
    auto module = op->getParentOfType<ModuleOp>();
    auto input = adaptor.getValue();
    AtomAttr atom = adaptor.getTag();

    auto isTupleOp = rewriter.create<cir::IsTupleOp>(loc, input);
    Value isTuple = isTupleOp.getResult();
    Value arity = isTupleOp.getArity();
    Value one = createI32Constant(rewriter, loc, 1);
    Value withAtLeastOneElement = rewriter.create<LLVM::ICmpOp>(
        loc, LLVM::ICmpPredicate::uge, arity, one);
    Value isCandidate =
        rewriter.create<LLVM::AndOp>(loc, isTuple, withAtLeastOneElement);

    rewriter.replaceOpWithNewOp<scf::IfOp>(
        op, TypeRange({i1Ty}), isCandidate,
        // isCandidate==true, i.e. we should examine the first element of the
        // tuple
        [&](OpBuilder &builder, Location l) {
          auto tupleTy = getTupleType(1);
          auto termPtrTy = LLVM::LLVMPointerType::get(getTermType());
          auto i32Ty = getI32Type();
          Value tuplePtr = decodePtr(builder, l, tupleTy, input);
          SmallVector<Value> indices;
          // This first index refers to the base of the tuple struct
          // This second index refers to the base of the elements array
          // This third index refers to the first element of the array
          Value zero = createIsizeConstant(builder, l, 0);
          Value first = createIndexAttrConstant(builder, l, i32Ty, 0);
          Value elemPtr = builder.create<LLVM::GEPOp>(
              l, termPtrTy, tuplePtr, ValueRange({zero, first, first}));
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
// IsNumberOp
//===---------===//
struct IsNumberOpLowering : public ConvertCIROpToLLVMPattern<cir::IsNumberOp> {
  using ConvertCIROpToLLVMPattern<cir::IsNumberOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::IsNumberOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto i1Ty = getI1Type();
    auto termTy = getTermType();
    auto module = op->getParentOfType<ModuleOp>();

    Operation *callee = module.lookupSymbol("__firefly_builtin_is_number");
    if (!callee) {
      auto calleeType =
          LLVM::LLVMFunctionType::get(i1Ty, ArrayRef<Type>{termTy});
      insertFunctionDeclaration(rewriter, loc, module,
                                "__firefly_builtin_is_number", calleeType);
    }

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange({i1Ty}),
                                              "__firefly_builtin_is_number",
                                              ValueRange({adaptor.getValue()}));
    return success();
  }
};

//===---------===//
// IsFloatOp
//===---------===//
struct IsFloatOpLowering : public ConvertCIROpToLLVMPattern<cir::IsFloatOp> {
  using ConvertCIROpToLLVMPattern<cir::IsFloatOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::IsFloatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto value = adaptor.getValue();

    // The value is a float if its NaN bits (ignoring the quiet bit) are not set
    Value nan = createIsizeConstant(rewriter, loc, NANBOX_NAN);
    Value masked = rewriter.create<LLVM::AndOp>(loc, value, nan);
    rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(op, LLVM::ICmpPredicate::ne,
                                              value, masked);
    return success();
  }
};

//===---------===//
// IsIntegerOp
//===---------===//
struct IsIntegerOpLowering
    : public ConvertCIROpToLLVMPattern<cir::IsIntegerOp> {
  using ConvertCIROpToLLVMPattern<cir::IsIntegerOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::IsIntegerOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto value = adaptor.getValue();

    // _firefly_builtin_typeof == TermKind::Int
    Value ty = rewriter.create<cir::TypeOfOp>(loc, value);
    Value expected = createI32Constant(rewriter, loc, TermKind::Int);
    rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(op, LLVM::ICmpPredicate::eq, ty,
                                              expected);
    return success();
  }
};

//===---------===//
// IsIsizeOp
//===---------===//
struct IsIsizeOpLowering : public ConvertCIROpToLLVMPattern<cir::IsIsizeOp> {
  using ConvertCIROpToLLVMPattern<cir::IsIsizeOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::IsIsizeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto value = adaptor.getValue();

    // value & NEG_INFINITY == NEG_INFINITY
    Value mask = createIsizeConstant(rewriter, loc, NANBOX_NEG_INFINITY);
    Value masked = rewriter.create<LLVM::AndOp>(loc, value, mask);
    rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(op, LLVM::ICmpPredicate::eq,
                                              masked, mask);
    return success();
  }
};

//===---------===//
// IsBigIntOp
//===---------===//
struct IsBigIntOpLowering : public ConvertCIROpToLLVMPattern<cir::IsBigIntOp> {
  using ConvertCIROpToLLVMPattern<cir::IsBigIntOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::IsBigIntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto value = adaptor.getValue();

    // value & NEG_INFINITY != NEG_INFINITY && __firefly_builtin_typeof ==
    // TermKind::Int
    Value ty = rewriter.create<cir::TypeOfOp>(loc, value);
    Value expected = createI32Constant(rewriter, loc, TermKind::Int);
    Value isInt = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq,
                                                ty, expected);
    Value mask = createIsizeConstant(rewriter, loc, NANBOX_NEG_INFINITY);
    Value masked = rewriter.create<LLVM::AndOp>(loc, value, mask);
    Value notSmallInt = rewriter.create<LLVM::ICmpOp>(
        loc, LLVM::ICmpPredicate::ne, masked, mask);
    rewriter.replaceOpWithNewOp<LLVM::AndOp>(op, isInt, notSmallInt);
    return success();
  }
};

//===---------===//
// IsAtomOp
//===---------===//
struct IsAtomOpLowering : public ConvertCIROpToLLVMPattern<cir::IsAtomOp> {
  using ConvertCIROpToLLVMPattern<cir::IsAtomOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::IsAtomOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto input = adaptor.getValue();

    // Extract the tag bits
    Value tagMask = createTermConstant(rewriter, loc, TAG_MASK);
    Value masked = rewriter.create<LLVM::AndOp>(loc, input, tagMask);
    // The masked value must meet one of two criteria:
    //
    // 1. Have it's tag bits be equal to the 'false' value, which is the same as
    // the atom tag scheme
    // 2. Be equal to the 'true' value (NOTE: The entire thing, not just the tag
    // bits, as 'true' overlaps with the tag scheme for Rc<T>)
    //
    Value atomTag =
        createTermConstant(rewriter, loc, NANBOX_CANONICAL_NAN | 0x02);
    Value isAtom = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq,
                                                 masked, atomTag);
    Value trueTag =
        createTermConstant(rewriter, loc, NANBOX_CANONICAL_NAN | 0x03);
    Value isTrue = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq,
                                                 input, trueTag);
    // Combine the two checks to give us our result
    rewriter.replaceOpWithNewOp<LLVM::OrOp>(op, isAtom, isTrue);
    return success();
  }
};

//===---------===//
// IsBoolOp
//===---------===//
struct IsBoolOpLowering : public ConvertCIROpToLLVMPattern<cir::IsBoolOp> {
  using ConvertCIROpToLLVMPattern<cir::IsBoolOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::IsBoolOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto value = adaptor.getValue();

    Value constFalse =
        createIsizeConstant(rewriter, loc, NANBOX_CANONICAL_NAN | 0x02);
    Value isFalse = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq,
                                                  value, constFalse);
    Value constTrue =
        createIsizeConstant(rewriter, loc, NANBOX_CANONICAL_NAN | 0x03);
    Value isTrue = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq,
                                                 value, constTrue);
    rewriter.replaceOpWithNewOp<LLVM::OrOp>(op, isFalse, isTrue);
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
    auto expectedType = adaptor.getExpected();
    auto value = adaptor.getValue();

    if (auto boxTy = expectedType.dyn_cast<CIRBoxType>()) {
      TermKind::Kind expectedTermKind = TermKind::Invalid;
      if (boxTy.isa<CIRMapType>()) {
        expectedTermKind = TermKind::Map;
      } else if (boxTy.isa<CIRBitsType>() || boxTy.isa<CIRBinaryType>()) {
        expectedTermKind = TermKind::Binary;
      } else if (boxTy.isa<CIRPidType>()) {
        expectedTermKind = TermKind::Pid;
      } else if (boxTy.isa<CIRPortType>()) {
        expectedTermKind = TermKind::Port;
      } else if (boxTy.isa<CIRReferenceType>()) {
        expectedTermKind = TermKind::Reference;
      } else if (boxTy.isa<CIRFunType>()) {
        expectedTermKind = TermKind::Closure;
      } else {
        return rewriter.notifyMatchFailure(
            op, "failed to lower is_type op, unsupported boxed match type");
      }

      // __firefly_builtin_typeof == expectedTermKind
      Value ty = rewriter.create<cir::TypeOfOp>(loc, value);
      Value expected = createI32Constant(rewriter, loc, expectedTermKind);
      rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(op, LLVM::ICmpPredicate::eq, ty,
                                                expected);
      return success();
    } else {
      return rewriter.notifyMatchFailure(
          op, "failed to lower is_type op, unsupported match type");
    }
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
    auto module = op->getParentOfType<ModuleOp>();
    auto process = adaptor.getProcess();
    auto allocType = adaptor.getAllocType();
    auto i8Ty = rewriter.getI8Type();
    auto i32Ty = rewriter.getI32Type();
    auto isizeTy = getIsizeType();
    auto i8PtrTy = LLVM::LLVMPointerType::get(i8Ty);
    auto procPtrTy = LLVM::LLVMPointerType::get(getProcessType());

    Operation *callee = module.lookupSymbol("__firefly_builtin_malloc");
    if (!callee) {
      auto calleeType = LLVM::LLVMFunctionType::get(
          i8PtrTy, ArrayRef<Type>{procPtrTy, i32Ty, isizeTy});
      insertFunctionDeclaration(rewriter, loc, module,
                                "__firefly_builtin_malloc", calleeType);
    }

    Type outType;
    TermKind::Kind mallocType = TermKind::Invalid;
    unsigned size = 0;

    if (allocType.isa<CIRConsType>()) {
      mallocType = TermKind::Cons;
      outType = convertType(allocType);
    } else if (allocType.isa<TupleType>()) {
      mallocType = TermKind::Tuple;
      outType = convertType(allocType);
      size = allocType.cast<TupleType>().size();
    } else if (allocType.isa<CIRFunType>()) {
      mallocType = TermKind::Closure;
      size = allocType.cast<CIRFunType>().getEnvArity();
      outType = getClosureType();
    } else {
      return rewriter.notifyMatchFailure(
          op, "failed to lower malloc op, unsupported boxed type");
    }

    auto kindArg = createIndexAttrConstant(rewriter, loc, i32Ty, mallocType);
    auto arityArg = createIsizeConstant(rewriter, loc, size);
    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, TypeRange({i8PtrTy}), "__firefly_builtin_malloc",
        ValueRange({process, kindArg, arityArg}));

    auto ptrTy = LLVM::LLVMPointerType::get(outType);
    rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(op, ptrTy,
                                                 callOp->getResult(0));
    return success();
  }
};

//===---------===//
// UnpackEnvOp
//===---------===//
struct UnpackEnvOpLowering
    : public ConvertCIROpToLLVMPattern<cir::UnpackEnvOp> {
  using ConvertCIROpToLLVMPattern<cir::UnpackEnvOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::UnpackEnvOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // First, unbox the term as a GcBox<Closure>
    auto closureTy = getClosureType();
    auto termTy = getTermType();
    auto termPtrTy = LLVM::LLVMPointerType::get(termTy);
    Value ptr = decodeGcBoxPtr(rewriter, loc, closureTy, adaptor.getFun());

    // Then obtain the address of the specific env item
    Value base = createI32Constant(rewriter, loc, 0);
    Value env = createI32Constant(rewriter, loc, 4);
    Value index =
        createI32Constant(rewriter, loc, adaptor.getIndex().getLimitedValue());
    auto itemAddr = rewriter.create<LLVM::GEPOp>(
        loc, termPtrTy, ptr, ValueRange({base, env, index}));

    // Then store the input value at the calculated pointer
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, itemAddr);
    return success();
  }
};

//===---------===//
// MakeFunOp
//===---------===//
struct MakeFunOpLowering : public ConvertCIROpToLLVMPattern<cir::MakeFunOp> {
  using ConvertCIROpToLLVMPattern<cir::MakeFunOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::MakeFunOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();

    auto termTy = getTermType();
    auto process = adaptor.getProcess();
    auto calleeName = adaptor.getCallee();
    auto funTy = op->getAttrOfType<TypeAttr>("funType").getValue();
    // TODO: Make sure the callee type matches the type of the callee function,
    // NOT a closure type
    Type calleeType;
    Operation *calleeOp = module.lookupSymbol(calleeName);
    assert(calleeOp && "expected callee op to already be defined");
    if (auto calleeMlirFun = dyn_cast<func::FuncOp>(calleeOp)) {
      calleeType = convertType(calleeMlirFun.getFunctionType());
    } else if (auto calleeLlvmFun = dyn_cast<LLVM::LLVMFuncOp>(calleeOp)) {
      calleeType = calleeLlvmFun.getFunctionType();
    }
    auto callee =
        rewriter.create<LLVM::AddressOfOp>(loc, calleeType, calleeName);

    SmallVector<Type, 1> envTypes;
    for (auto env : adaptor.getEnv()) {
      envTypes.push_back(env.getType());
    }

    // Allocate space on the process heap for the closure
    auto closureTy = getClosureType();
    auto closurePtrTy = LLVM::LLVMPointerType::get(closureTy);
    auto termPtrTy = LLVM::LLVMPointerType::get(termTy);
    auto atomDataPtrTy = LLVM::LLVMPointerType::get(getAtomDataType());
    auto atomDataPtrPtrTy = LLVM::LLVMPointerType::get(atomDataPtrTy);
    auto isizeTy = getIsizeType();
    auto isizePtrTy = LLVM::LLVMPointerType::get(isizeTy);

    Value mallocPtr =
        rewriter.create<cir::MallocOp>(loc, process, TypeAttr::get(funTy));
    Operation *ptrCast = rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange({closurePtrTy}), ValueRange({mallocPtr}));
    Value ptr = ptrCast->getResult(0);

    // Write the various bits of metadata to the allocated closure

    // Use the callee name to obtain the module/function/arity metadata
    //
    // NOTE: It is invalid for a callee to not be a valid erlang function name
    auto moduleSplit = calleeName.split(':');
    auto moduleName = std::get<0>(moduleSplit);
    auto functionArity = std::get<1>(moduleSplit);
    if (functionArity.empty()) {
      op.emitError("invalid callee for fun, must be an erlang function with "
                   "name 'module:fun/arity'");
      return failure();
    }
    auto functionSplit = functionArity.rsplit('/');
    auto functionName = std::get<0>(functionSplit);
    auto arityStr = std::get<1>(functionSplit);
    if (arityStr.empty()) {
      op.emitError("invalid callee for fun, must be an erlang function with "
                   "name 'module:fun/arity'");
      return failure();
    }
    int8_t arity = 0;
    if (arityStr.getAsInteger(10, arity)) {
      op.emitError("invalid arity in fun name, expected valid integer");
      return failure();
    }

    // Store the module atom
    auto moduleAtom = createAtomData(rewriter, loc, moduleName, module);
    Value zero = createI32Constant(rewriter, loc, 0);
    auto modulePtr = rewriter.create<LLVM::GEPOp>(loc, atomDataPtrPtrTy, ptr,
                                                  ValueRange({zero, zero}));
    rewriter.create<LLVM::StoreOp>(loc, moduleAtom, modulePtr);

    // Store the function atom
    auto functionAtom = createAtomData(rewriter, loc, functionName, module);
    Value one = createI32Constant(rewriter, loc, 1);
    auto functionPtr = rewriter.create<LLVM::GEPOp>(loc, atomDataPtrPtrTy, ptr,
                                                    ValueRange({zero, one}));
    rewriter.create<LLVM::StoreOp>(loc, functionAtom, functionPtr);

    // Store the arity (isize)
    auto arityInt = createIsizeConstant(rewriter, loc, arity);
    Value two = createI32Constant(rewriter, loc, 2);
    auto arityPtr = rewriter.create<LLVM::GEPOp>(loc, isizePtrTy, ptr,
                                                 ValueRange({zero, two}));
    rewriter.create<LLVM::StoreOp>(loc, arityInt, arityPtr);

    // Store the callee pointer
    auto opaqueFunTy =
        LLVM::LLVMFunctionType::get(getVoidType(), ArrayRef<Type>{});
    auto opaqueFunPtrTy = LLVM::LLVMPointerType::get(opaqueFunTy);
    auto opaqueFunPtrPtrTy = LLVM::LLVMPointerType::get(opaqueFunPtrTy);
    Value three = createI32Constant(rewriter, loc, 3);
    auto calleePtr = rewriter.create<LLVM::GEPOp>(loc, opaqueFunPtrPtrTy, ptr,
                                                  ValueRange({zero, three}));
    auto calleeRaw =
        rewriter.create<LLVM::BitcastOp>(loc, opaqueFunPtrTy, callee);
    rewriter.create<LLVM::StoreOp>(loc, calleeRaw, calleePtr);

    // Store the env in the closure
    Value four = createI32Constant(rewriter, loc, 4);
    uint64_t envIdx = 0;
    for (auto env : adaptor.getEnv()) {
      Value envIdxConst;
      switch (envIdx) {
      case 0:
        envIdxConst = zero;
        break;
      case 1:
        envIdxConst = one;
        break;
      case 2:
        envIdxConst = two;
        break;
      case 3:
        envIdxConst = three;
        break;
      case 4:
        envIdxConst = four;
        break;
      default:
        envIdxConst = createI32Constant(rewriter, loc, envIdx);
        break;
      }
      auto envPtr = rewriter.create<LLVM::GEPOp>(
          loc, termPtrTy, ptr, ValueRange({zero, four, envIdxConst}));
      rewriter.create<LLVM::StoreOp>(loc, env, envPtr);
      envIdx++;
    }

    // Box the pointer to the allocation
    Value box = encodeGcBoxPtr(rewriter, loc, ptr);
    Value isErr = createBoolConstant(rewriter, loc, false);

    // Return the boxed value
    rewriter.replaceOp(op, ValueRange({isErr, box}));
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
    auto process = adaptor.getProcess();

    // Allocate space on the process heap for the cell
    auto termTy = getTermType();
    auto consTy = rewriter.getType<CIRConsType>();
    auto cellTy = getConsType();
    auto cellPtrTy = LLVM::LLVMPointerType::get(cellTy);
    auto termPtrTy = LLVM::LLVMPointerType::get(termTy);

    Value mallocPtr =
        rewriter.create<cir::MallocOp>(loc, process, TypeAttr::get(consTy));
    Operation *ptrCast = rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange({cellPtrTy}), ValueRange({mallocPtr}));
    Value ptr = ptrCast->getResult(0);
    Value base = createI32Constant(rewriter, loc, 0);
    Value zero = createI32Constant(rewriter, loc, 0);
    Value one = createI32Constant(rewriter, loc, 1);
    // Get a pointer to the head and tail and store their values
    auto headPtr = rewriter.create<LLVM::GEPOp>(loc, termPtrTy, ptr,
                                                ValueRange({base, zero}));
    rewriter.create<LLVM::StoreOp>(loc, adaptor.getHead(), headPtr);
    auto tailPtr = rewriter.create<LLVM::GEPOp>(loc, termPtrTy, ptr,
                                                ValueRange({base, one}));
    rewriter.create<LLVM::StoreOp>(loc, adaptor.getTail(), tailPtr);
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
    auto termTy = getTermType();
    Value ptr = decodePtr(rewriter, loc, termTy, adaptor.getCell());

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
    auto termPtrTy = LLVM::LLVMPointerType::get(getTermType());
    Value ptr = decodeListPtr(rewriter, loc, adaptor.getCell());

    // Then, calculate the pointer to the tail cell
    Value base = createI32Constant(rewriter, loc, 0);
    Value one = createI32Constant(rewriter, loc, 1);
    Value tailPtr = rewriter.create<LLVM::GEPOp>(loc, termPtrTy, ptr,
                                                 ValueRange({base, one}));

    // Then return the result of loading the value from the calculated pointer
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, tailPtr);
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
    auto tuple = adaptor.getTuple();

    // NOTE: We should generate assertions that the input is of the correct
    // shape during debug builds, but for now we lower with the assumption that
    // the compiler has already generated those checks

    // First, unbox the pointer as a pointer to a tuple
    auto tupleTy = getTupleType(1);
    auto termPtrTy = LLVM::LLVMPointerType::get(getTermType());
    Value ptr = decodePtr(rewriter, loc, tupleTy, tuple);
    // Then, calculate the pointer to the <index>th element
    Value base = createI32Constant(rewriter, loc, 0);
    Value one = createI32Constant(rewriter, loc, 1);
    Value index =
        createI32Constant(rewriter, loc, adaptor.getIndex().getLimitedValue());
    Value elemPtr = rewriter.create<LLVM::GEPOp>(
        loc, termPtrTy, ptr, ValueRange({base, one, index}));
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
    auto tuple = adaptor.getTuple();
    auto value = adaptor.getValue();
    auto updateInPlace = adaptor.getInPlace();

    // NOTE: We should generate assertions that the input is of the correct
    // shape during debug builds, but for now we lower with the assumption that
    // the compiler has already generated those checks

    // First, unbox the pointer as a pointer to a tuple
    auto tupleTy = getTupleType(1);
    auto tuplePtrTy = LLVM::LLVMPointerType::get(tupleTy);
    Value ptr = decodePtr(rewriter, loc, tupleTy, tuple);
    Value index = createIsizeConstant(rewriter, loc,
                                      adaptor.getIndex().getLimitedValue());
    if (!updateInPlace) {
      // If we can't mutate the tuple in-place, transform this op into
      // a call to a specialized builtin equivalent to erlang:setelement/3
      auto termTy = getTermType();
      auto isizeTy = getIsizeType();
      auto module = op->getParentOfType<ModuleOp>();
      Operation *callee = module.lookupSymbol("__firefly_set_element");
      if (!callee) {
        auto tuplePtrTy = LLVM::LLVMPointerType::get(tupleTy);
        auto calleeType = LLVM::LLVMFunctionType::get(
            termTy, ArrayRef<Type>{tuplePtrTy, isizeTy, termTy});
        insertFunctionDeclaration(rewriter, loc, module,
                                  "__firefly_set_element", calleeType);
      }
      rewriter.replaceOpWithNewOp<LLVM::CallOp>(
          op, TypeRange({termTy}), "__firefly_set_element",
          ValueRange({ptr, index, value}));
      return success();
    }

    // If we reach here, we are allowed to mutate the original tuple in-place;
    // Then, calculate the pointer to the <index>th element
    Value base = createI32Constant(rewriter, loc, 0);
    Value one = createI32Constant(rewriter, loc, 1);
    Value index32 =
        createI32Constant(rewriter, loc, adaptor.getIndex().getLimitedValue());
    Value elemPtr = rewriter.create<LLVM::GEPOp>(
        loc, tuplePtrTy, ptr, ValueRange({base, one, index32}));
    // Then store the input value at the calculated pointer
    rewriter.create<LLVM::StoreOp>(loc, value, elemPtr);
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
    auto klass = adaptor.getExceptionClass();
    auto reason = adaptor.getExceptionReason();
    auto trace = adaptor.getExceptionTrace();

    // Get a reference to the process exception pointer
    auto module = op->getParentOfType<ModuleOp>();

    auto exceptionTy = getExceptionType();
    auto exceptionPtrTy = LLVM::LLVMPointerType::get(exceptionTy);
    auto klassTy = getTermType();
    auto reasonTy = getTermType();
    auto traceTy = LLVM::LLVMPointerType::get(getTraceType());
    Operation *callee = module.lookupSymbol("__firefly_builtin_raise/3");
    if (!callee) {
      auto calleeType = LLVM::LLVMFunctionType::get(
          exceptionPtrTy, ArrayRef<Type>{klassTy, reasonTy, traceTy});
      insertFunctionDeclaration(rewriter, loc, module,
                                "__firefly_builtin_raise/3", calleeType);
    }

    // Create the raw exception value using __firefly_builtin_raise/3
    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, TypeRange({exceptionPtrTy}), "__firefly_builtin_raise/3",
        ValueRange({klass, reason, trace}));
    auto exceptionPtr = callOp.getResult();

    // Lastly, convert this op to our multi-value return convention
    Value isError = createBoolConstant(rewriter, loc, true);
    Value exceptionTerm =
        rewriter.create<LLVM::PtrToIntOp>(loc, getTermType(), exceptionPtr);
    rewriter.replaceOpWithNewOp<func::ReturnOp>(
        op, ValueRange({isError, exceptionTerm}));
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
    auto exceptionPtr = adaptor.getException();

    auto termPtrTy = LLVM::LLVMPointerType::get(getTermType());
    Value zero = createI32Constant(rewriter, loc, 0);
    Value one = createI32Constant(rewriter, loc, 0);
    auto classAddr = rewriter.create<LLVM::GEPOp>(loc, termPtrTy, exceptionPtr,
                                                  ValueRange({zero, one}));

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
    auto exceptionPtr = adaptor.getException();

    auto termPtrTy = LLVM::LLVMPointerType::get(getTermType());
    Value zero = createI32Constant(rewriter, loc, 0);
    Value two = createI32Constant(rewriter, loc, 1);
    auto reasonAddr = rewriter.create<LLVM::GEPOp>(loc, termPtrTy, exceptionPtr,
                                                   ValueRange({zero, two}));

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
    auto exceptionPtr = adaptor.getException();

    auto tracePtrTy =
        LLVM::LLVMPointerType::get(LLVM::LLVMPointerType::get(getTraceType()));
    Value zero = createI32Constant(rewriter, loc, 0);
    Value three = createI32Constant(rewriter, loc, 2);
    auto traceAddr = rewriter.create<LLVM::GEPOp>(loc, tracePtrTy, exceptionPtr,
                                                  ValueRange({zero, three}));

    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, traceAddr);
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
        op, TypeRange({voidTy}), "__firefly_builtin_yield", ValueRange());
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
    auto timeout = adaptor.getTimeout();

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange({recvCtxTy}),
                                              "__firefly_builtin_receive_start",
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
    auto context = adaptor.getContext();

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange({i8Ty}),
                                              "__firefly_builtin_receive_start",
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
    auto termPtrTy = LLVM::LLVMPointerType::get(getTermType());
    auto context = adaptor.getContext();

    Value zero = createIsizeConstant(rewriter, loc, 0);
    Value zero32 = createIndexAttrConstant(rewriter, loc, i32Ty, 0);
    Value one = createIndexAttrConstant(rewriter, loc, i32Ty, 1);
    Value messagePtr = rewriter.create<LLVM::ExtractValueOp>(
        loc, context, ArrayRef<int64_t>{2});
    Value dataAddr;
    if (getPointerBitwidth() == 64) {
      dataAddr = rewriter.create<LLVM::GEPOp>(
          loc, termPtrTy, messagePtr, ValueRange({zero, one, one, one}));
    } else {
      dataAddr = rewriter.create<LLVM::GEPOp>(
          loc, termPtrTy, messagePtr, ValueRange({zero, one, one, zero32}));
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
    auto context = adaptor.getContext();

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange({voidTy}),
                                              "__firefly_builtin_receive_pop",
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
    auto context = adaptor.getContext();

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange({voidTy}),
                                              "__firefly_builtin_receive_done",
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
          std::string("firefly_dispatch_") + llvm::toHex(hasher.result(), true);
      std::string sectionName;
      if (isMachO())
        sectionName = std::string("__DATA,__dispatch");
      else
        sectionName = std::string("__dispatch");
      auto sectionAttr =
          rewriter.getNamedAttr("section", rewriter.getStringAttr(sectionName));
      auto entryConst = rewriter.create<LLVM::GlobalOp>(
          loc, dispatchEntryTy, /*isConstant=*/true, LLVM::Linkage::LinkonceODR,
          globalName, Attribute(),
          /*alignment=*/8, /*addrspace=*/0, /*dso_local=*/false,
          /*thread_local=*/false, ArrayRef<NamedAttribute>{sectionAttr});

      auto &initRegion = entryConst.getInitializerRegion();
      auto entryBlock = rewriter.createBlock(&initRegion);

      rewriter.setInsertionPointToStart(entryBlock);

      Value entry = rewriter.create<LLVM::UndefOp>(loc, dispatchEntryTy);

      // Store the module name
      auto moduleNamePtr = createAtomDataGlobal(rewriter, loc, mod, module);
      entry = rewriter.create<LLVM::InsertValueOp>(
          loc, entry, moduleNamePtr, rewriter.getDenseI64ArrayAttr(0));

      // Store the function name
      auto functionNamePtr = createAtomDataGlobal(rewriter, loc, mod, function);
      entry = rewriter.create<LLVM::InsertValueOp>(
          loc, entry, functionNamePtr, rewriter.getDenseI64ArrayAttr(1));

      // Store the arity
      auto arityVal = rewriter.create<LLVM::ConstantOp>(
          loc, i8ty, rewriter.getI8IntegerAttr(arity));
      entry = rewriter.create<LLVM::InsertValueOp>(
          loc, entry, arityVal, rewriter.getDenseI64ArrayAttr(2));

      // Get the LLVM type of the function referenced by the symbol
      Operation *fun = mod.lookupSymbol(symbol);
      Type funTy;
      if (isa<LLVM::LLVMFuncOp>(fun))
        funTy = cast<LLVM::LLVMFuncOp>(fun).getFunctionType();
      else
        funTy = convertType(cast<func::FuncOp>(fun).getFunctionType());
      auto funPtr = rewriter.create<LLVM::AddressOfOp>(loc, funTy, symbol);
      // Cast the address of the function to an opaque function pointer (i.e.
      // `*const ()`)
      auto opaqueFunTy =
          LLVM::LLVMFunctionType::get(getVoidType(), ArrayRef<Type>{});
      auto opaqueFunPtrTy = LLVM::LLVMPointerType::get(opaqueFunTy);
      auto opaqueFunPtr =
          rewriter.create<LLVM::BitcastOp>(loc, opaqueFunPtrTy, funPtr);
      // Store the function pointer
      entry = rewriter.create<LLVM::InsertValueOp>(
          loc, entry, opaqueFunPtr, rewriter.getDenseI64ArrayAttr(4));

      rewriter.create<LLVM::ReturnOp>(loc, entry);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

//===------------===//
// BinaryMatchStartOp
//===------------===//
struct BinaryMatchStartOpLowering
    : public ConvertCIROpToLLVMPattern<cir::BinaryMatchStartOp> {
  using ConvertCIROpToLLVMPattern<
      cir::BinaryMatchStartOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::BinaryMatchStartOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto termTy = getTermType();
    auto resultTy = getResultType(termTy);
    auto i1Ty = getI1Type();
    auto bin = adaptor.getBin();

    auto module = op->getParentOfType<ModuleOp>();
    Operation *callee = module.lookupSymbol("__firefly_bs_match_start");
    if (!callee) {
      auto calleeType =
          LLVM::LLVMFunctionType::get(resultTy, ArrayRef<Type>{termTy});
      insertFunctionDeclaration(rewriter, loc, module,
                                "__firefly_bs_match_start", calleeType);
    }
    auto callOp = rewriter.create<LLVM::CallOp>(loc, TypeRange({resultTy}),
                                                "__firefly_bs_match_start",
                                                ValueRange({bin}));
    Value callResult = callOp->getResult(0);
    Value isErrWide = rewriter.create<LLVM::ExtractValueOp>(
        loc, callResult, ArrayRef<int64_t>{0});
    Value isErr = rewriter.create<LLVM::TruncOp>(loc, i1Ty, isErrWide);
    Value result = rewriter.create<LLVM::ExtractValueOp>(loc, callResult,
                                                         ArrayRef<int64_t>{1});
    rewriter.replaceOp(op, ValueRange({isErr, result}));
    return success();
  }
};

//===------------===//
// BinaryMatchOp
//===------------===//
struct BinaryMatchOpLowering
    : public ConvertCIROpToLLVMPattern<cir::BinaryMatchOp> {
  using ConvertCIROpToLLVMPattern<
      cir::BinaryMatchOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::BinaryMatchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto voidTy = getVoidType();
    auto matchTy = getMatchResultType();
    auto resultTy = getResultType(matchTy);
    auto resultPtrTy = LLVM::LLVMPointerType::get(resultTy);
    auto termTy = getTermType();
    auto termPtrTy = LLVM::LLVMPointerType::get(termTy);
    auto matchContextTy = LLVM::LLVMPointerType::get(getMatchContextType());
    auto matchContextPtrTy = LLVM::LLVMPointerType::get(matchContextTy);
    auto isizeTy = getIsizeType();
    auto isizePtrTy = LLVM::LLVMPointerType::get(isizeTy);
    auto i1Ty = getI1Type();
    auto i64Ty = getI64Type();

    auto matchContext = adaptor.getMatchContext();
    BinarySpecAttr specAttr = adaptor.getSpec();
    BinaryEntrySpecifier spec = specAttr.getValue();

    auto module = op->getParentOfType<ModuleOp>();
    Operation *callee = module.lookupSymbol("__firefly_bs_match");
    // Because of the size of the result type, a pointer to the
    // zero-initialized structure is passed in as an implicit first argument,
    // i.e. the sret attribute must be applied
    ArrayRef<NamedAttribute> attrs = {};
    SmallVector<DictionaryAttr> argAttrs;
    SmallVector<NamedAttribute> resultPtrAttrs;
    resultPtrAttrs.push_back(
        rewriter.getNamedAttr("llvm.noalias", rewriter.getUnitAttr()));
    resultPtrAttrs.push_back(
        rewriter.getNamedAttr("llvm.sret", rewriter.getUnitAttr()));
    argAttrs.push_back(rewriter.getDictionaryAttr(resultPtrAttrs));
    argAttrs.push_back(rewriter.getDictionaryAttr({}));
    argAttrs.push_back(rewriter.getDictionaryAttr({}));
    argAttrs.push_back(rewriter.getDictionaryAttr({}));
    if (!callee) {
      auto calleeType = LLVM::LLVMFunctionType::get(
          voidTy, ArrayRef<Type>{resultPtrTy, matchContextTy, i64Ty, termTy});
      callee =
          insertFunctionDeclaration(rewriter, loc, module, "__firefly_bs_match",
                                    calleeType, attrs, argAttrs);
    }

    uint64_t specRawInt = 0;
    specRawInt |= ((uint64_t)(spec.data.raw)) << 32;
    specRawInt |= (uint64_t)(spec.tag);

    Value specRaw = createI64Constant(rewriter, loc, specRawInt);
    Value size = adaptor.getSize();
    if (!size) {
      size = createTermConstant(rewriter, loc, NANBOX_CANONICAL_NAN);
    }

    Value one = createI32Constant(rewriter, loc, 1);
    Value resultPtr = rewriter.create<LLVM::AllocaOp>(loc, resultPtrTy, one);

    rewriter.create<LLVM::CallOp>(
        loc, cast<LLVM::LLVMFuncOp>(callee),
        ValueRange({resultPtr, matchContext, specRaw, size}));

    Value zero = createI32Constant(rewriter, loc, 0);

    Value isErrPtr = rewriter.create<LLVM::GEPOp>(loc, isizePtrTy, resultPtr,
                                                  ValueRange({zero, zero}));
    Value isErrWide = rewriter.create<LLVM::LoadOp>(loc, isErrPtr);
    Value isErr = rewriter.create<LLVM::TruncOp>(loc, i1Ty, isErrWide);

    Value extractedPtr = rewriter.create<LLVM::GEPOp>(
        loc, termPtrTy, resultPtr, ValueRange({zero, one, zero}));
    Value extracted = rewriter.create<LLVM::LoadOp>(loc, extractedPtr);

    Value updatedMatchCtxPtr = rewriter.create<LLVM::GEPOp>(
        loc, matchContextPtrTy, resultPtr, ValueRange({zero, one, one}));
    Value updatedMatchContext =
        rewriter.create<LLVM::LoadOp>(loc, updatedMatchCtxPtr);
    rewriter.replaceOp(op, ValueRange({isErr, extracted, updatedMatchContext}));
    return success();
  }
};

//===------------===//
// BinaryTestTailOp
//===------------===//
struct BinaryTestTailOpLowering
    : public ConvertCIROpToLLVMPattern<cir::BinaryTestTailOp> {
  using ConvertCIROpToLLVMPattern<
      cir::BinaryTestTailOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::BinaryTestTailOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto matchContextTy = LLVM::LLVMPointerType::get(getMatchContextType());
    auto isizeTy = getIsizeType();
    auto i1Ty = getI1Type();
    auto matchCtx = adaptor.getMatchContext();
    auto size =
        createIsizeConstant(rewriter, loc, adaptor.getSize().getLimitedValue());

    auto module = op->getParentOfType<ModuleOp>();
    Operation *callee = module.lookupSymbol("__firefly_bs_test_tail");
    if (!callee) {
      auto calleeType = LLVM::LLVMFunctionType::get(
          i1Ty, ArrayRef<Type>{matchContextTy, isizeTy});
      insertFunctionDeclaration(rewriter, loc, module, "__firefly_bs_test_tail",
                                calleeType);
    }
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange({i1Ty}),
                                              "__firefly_bs_test_tail",
                                              ValueRange({matchCtx, size}));
    return success();
  }
};

//===------------===//
// BinaryMatchSkipOp
//===------------===//
struct BinaryMatchSkipOpLowering
    : public ConvertCIROpToLLVMPattern<cir::BinaryMatchSkipOp> {
  using ConvertCIROpToLLVMPattern<
      cir::BinaryMatchSkipOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::BinaryMatchSkipOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto termTy = getTermType();
    auto matchContextTy = LLVM::LLVMPointerType::get(getMatchContextType());
    auto resultTy = getResultType(matchContextTy);
    auto i1Ty = getI1Type();
    auto i64Ty = getI64Type();

    auto matchContext = adaptor.getMatchContext();
    BinarySpecAttr specAttr = adaptor.getSpec();
    BinaryEntrySpecifier spec = specAttr.getValue();

    auto module = op->getParentOfType<ModuleOp>();
    Operation *callee = module.lookupSymbol("__firefly_bs_match_skip");
    if (!callee) {
      auto calleeType = LLVM::LLVMFunctionType::get(
          resultTy, ArrayRef<Type>{matchContextTy, i64Ty, termTy, i64Ty});
      callee = insertFunctionDeclaration(rewriter, loc, module,
                                         "__firefly_bs_match_skip", calleeType);
    }

    uint64_t specRawInt = 0;
    specRawInt |= ((uint64_t)(spec.data.raw)) << 32;
    specRawInt |= (uint64_t)(spec.tag);

    Value specRaw = createI64Constant(rewriter, loc, specRawInt);
    Value size = adaptor.getSize();
    Value matchValue = adaptor.getValue();

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, cast<LLVM::LLVMFuncOp>(callee),
        ValueRange({matchContext, specRaw, size, matchValue}));

    Value callResult = callOp->getResult(0);
    Value isErrWide = rewriter.create<LLVM::ExtractValueOp>(
        loc, callResult, ArrayRef<int64_t>{0});
    Value isErr = rewriter.create<LLVM::TruncOp>(loc, i1Ty, isErrWide);
    Value updatedMatchCtx = rewriter.create<LLVM::ExtractValueOp>(
        loc, callResult, ArrayRef<int64_t>{1});
    rewriter.replaceOp(op, ValueRange({isErr, updatedMatchCtx}));
    return success();
  }
};

//===------------===//
// BinaryPushOp
//===------------===//
struct BinaryPushOpLowering
    : public ConvertCIROpToLLVMPattern<cir::BinaryPushOp> {
  using ConvertCIROpToLLVMPattern<cir::BinaryPushOp>::ConvertCIROpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cir::BinaryPushOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto binaryBuilderTy = getBinaryBuilderType();
    auto binaryBuilderPtrTy = LLVM::LLVMPointerType::get(binaryBuilderTy);
    auto termTy = getTermType();
    auto resultTy = getResultType(termTy);
    auto i1Ty = getI1Type();
    auto i64Ty = getI64Type();

    auto bin = adaptor.getBin();
    BinarySpecAttr specAttr = adaptor.getSpec();
    BinaryEntrySpecifier spec = specAttr.getValue();

    auto module = op->getParentOfType<ModuleOp>();
    Operation *callee = module.lookupSymbol("__firefly_bs_push");
    if (!callee) {
      auto calleeType = LLVM::LLVMFunctionType::get(
          resultTy, ArrayRef<Type>{binaryBuilderPtrTy, i64Ty, termTy, termTy});
      insertFunctionDeclaration(rewriter, loc, module, "__firefly_bs_push",
                                calleeType);
    }

    uint64_t specRawInt = 0;
    specRawInt |= ((uint64_t)(spec.data.raw)) << 32;
    specRawInt |= (uint64_t)(spec.tag);

    Value specRaw = createI64Constant(rewriter, loc, specRawInt);
    Value size = adaptor.getSize();
    if (!size) {
      size = createTermConstant(rewriter, loc, NANBOX_CANONICAL_NAN);
    }

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, TypeRange({resultTy}), "__firefly_bs_push",
        ValueRange({bin, specRaw, adaptor.getValue(), size}));
    Value callResult = callOp->getResult(0);
    Value isErrWide = rewriter.create<LLVM::ExtractValueOp>(
        loc, callResult, ArrayRef<int64_t>{0});
    Value isErr = rewriter.create<LLVM::TruncOp>(loc, i1Ty, isErrWide);
    Value newBin = rewriter.create<LLVM::ExtractValueOp>(loc, callResult,
                                                         ArrayRef<int64_t>{1});
    rewriter.replaceOp(op, ValueRange({isErr, newBin}));
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
  // populateGeneratedPDLLPatterns(patterns);
  // These are the conversion patterns for CIR ops
  patterns.add<DispatchTableOpLowering>(typeConverter);
  patterns.add<ConstantOpLowering>(typeConverter);
  patterns.add<ConstantNullOpLowering>(typeConverter);
  patterns.add<CastOpLowering>(typeConverter);
  patterns.add<CallOpLowering>(typeConverter);
  patterns.add<EnterOpLowering>(typeConverter);
  // patterns.add<CallIndirectOpLowering>(typeConverter);
  patterns.add<IsNullOpLowering>(typeConverter);
  patterns.add<TruncOpLowering>(typeConverter);
  patterns.add<ZExtOpLowering>(typeConverter);
  patterns.add<AndOpLowering>(typeConverter);
  patterns.add<AndAlsoOpLowering>(typeConverter);
  patterns.add<OrOpLowering>(typeConverter);
  patterns.add<OrElseOpLowering>(typeConverter);
  patterns.add<XorOpLowering>(typeConverter);
  patterns.add<NotOpLowering>(typeConverter);
  patterns.add<TypeOfOpLowering>(typeConverter);
  patterns.add<IsTypeOpLowering>(typeConverter);
  patterns.add<IsAtomOpLowering>(typeConverter);
  patterns.add<IsBoolOpLowering>(typeConverter);
  patterns.add<IsNumberOpLowering>(typeConverter);
  patterns.add<IsIntegerOpLowering>(typeConverter);
  patterns.add<IsIsizeOpLowering>(typeConverter);
  patterns.add<IsBigIntOpLowering>(typeConverter);
  patterns.add<IsListOpLowering>(typeConverter);
  patterns.add<IsNonEmptyListOpLowering>(typeConverter);
  patterns.add<IsTupleOpLowering>(typeConverter);
  patterns.add<IsTaggedTupleOpLowering>(typeConverter);
  patterns.add<MallocOpLowering>(typeConverter);
  patterns.add<MakeFunOpLowering>(typeConverter);
  patterns.add<UnpackEnvOpLowering>(typeConverter);
  patterns.add<ConsOpLowering>(typeConverter);
  patterns.add<HeadOpLowering>(typeConverter);
  patterns.add<TailOpLowering>(typeConverter);
  patterns.add<SetElementOpLowering>(typeConverter);
  patterns.add<GetElementOpLowering>(typeConverter);
  patterns.add<RaiseOpLowering>(typeConverter);
  patterns.add<ExceptionClassOpLowering>(typeConverter);
  patterns.add<ExceptionReasonOpLowering>(typeConverter);
  patterns.add<ExceptionTraceOpLowering>(typeConverter);
  patterns.add<YieldOpLowering>(typeConverter);
  patterns.add<RecvStartOpLowering>(typeConverter);
  patterns.add<RecvNextOpLowering>(typeConverter);
  patterns.add<RecvPeekOpLowering>(typeConverter);
  patterns.add<RecvPopOpLowering>(typeConverter);
  patterns.add<RecvDoneOpLowering>(typeConverter);
  patterns.add<BinaryMatchStartOpLowering>(typeConverter);
  patterns.add<BinaryMatchOpLowering>(typeConverter);
  patterns.add<BinaryMatchSkipOpLowering>(typeConverter);
  patterns.add<BinaryTestTailOpLowering>(typeConverter);
  patterns.add<BinaryPushOpLowering>(typeConverter);

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
