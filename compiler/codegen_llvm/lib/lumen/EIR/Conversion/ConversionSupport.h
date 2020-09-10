#ifndef LUMEN_EIR_CONVERSION_CONVERSION_SUPPORT_H
#define LUMEN_EIR_CONVERSION_CONVERSION_SUPPORT_H

#include "lumen/EIR/Conversion/TargetInfo.h"
#include "lumen/EIR/IR/EIRAttributes.h"
#include "lumen/EIR/IR/EIROps.h"
#include "lumen/EIR/IR/EIRTypes.h"

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/Casting.h"

using ::llvm::SmallVectorImpl;
using ::llvm::TargetMachine;
using ::llvm::StringSwitch;
using ::llvm::dyn_cast_or_null;
using ::llvm::cast;
using ::llvm::isa;
using ::mlir::ConversionPatternRewriter;
using ::mlir::LLVMTypeConverter;
using ::mlir::LogicalResult;
using ::mlir::PatternRewriter;
using ::mlir::success;
using ::mlir::SymbolTable;
using ::mlir::edsc::OperationBuilder;
using ::mlir::edsc::ScopedContext;
using ::mlir::edsc::ValueBuilder;
using ::mlir::LLVM::LLVMType;
using ::mlir::LLVM::LLVMIntegerType;

namespace LLVM = ::mlir::LLVM;

using std_call = OperationBuilder<::mlir::CallOp>;
using llvm_null = ValueBuilder<LLVM::NullOp>;
using llvm_add = ValueBuilder<LLVM::AddOp>;
using llvm_and = ValueBuilder<LLVM::AndOp>;
using llvm_or = ValueBuilder<LLVM::OrOp>;
using llvm_xor = ValueBuilder<LLVM::XOrOp>;
using llvm_shl = ValueBuilder<LLVM::ShlOp>;
using llvm_shr = ValueBuilder<LLVM::LShrOp>;
using llvm_bitcast = ValueBuilder<LLVM::BitcastOp>;
using llvm_zext = ValueBuilder<LLVM::ZExtOp>;
using llvm_sext = ValueBuilder<LLVM::SExtOp>;
using llvm_trunc = ValueBuilder<LLVM::TruncOp>;
using llvm_constant = ValueBuilder<LLVM::ConstantOp>;
using llvm_extractvalue = ValueBuilder<LLVM::ExtractValueOp>;
using llvm_gep = ValueBuilder<LLVM::GEPOp>;
using llvm_addressof = ValueBuilder<LLVM::AddressOfOp>;
using llvm_insertvalue = ValueBuilder<LLVM::InsertValueOp>;
using llvm_condbr = OperationBuilder<LLVM::CondBrOp>;
using llvm_br = OperationBuilder<LLVM::BrOp>;
using llvm_call = OperationBuilder<LLVM::CallOp>;
using llvm_invoke = OperationBuilder<LLVM::InvokeOp>;
using llvm_icmp = ValueBuilder<LLVM::ICmpOp>;
using llvm_load = ValueBuilder<LLVM::LoadOp>;
using llvm_store = OperationBuilder<LLVM::StoreOp>;
using llvm_atomicrmw = OperationBuilder<LLVM::AtomicRMWOp>;
using llvm_select = ValueBuilder<LLVM::SelectOp>;
using llvm_mul = ValueBuilder<LLVM::MulOp>;
using llvm_ptrtoint = ValueBuilder<LLVM::PtrToIntOp>;
using llvm_inttoptr = ValueBuilder<LLVM::IntToPtrOp>;
using llvm_sub = ValueBuilder<LLVM::SubOp>;
using llvm_undef = ValueBuilder<LLVM::UndefOp>;
using llvm_urem = ValueBuilder<LLVM::URemOp>;
using llvm_alloca = ValueBuilder<LLVM::AllocaOp>;
using llvm_return = OperationBuilder<LLVM::ReturnOp>;
using llvm_landingpad = ValueBuilder<LLVM::LandingpadOp>;
using eir_cast = ValueBuilder<::lumen::eir::CastOp>;
using eir_gep = ValueBuilder<::lumen::eir::GetElementPtrOp>;
using eir_malloc = ValueBuilder<::lumen::eir::MallocOp>;
using eir_cons = ValueBuilder<::lumen::eir::ConsOp>;
using eir_list = ValueBuilder<::lumen::eir::ListOp>;
using eir_tuple = ValueBuilder<::lumen::eir::TupleOp>;
using eir_map = OperationBuilder<::lumen::eir::ConstructMapOp>;
using eir_nil = ValueBuilder<::lumen::eir::ConstantNilOp>;
using eir_none = ValueBuilder<::lumen::eir::ConstantNoneOp>;
using eir_constant_float = ValueBuilder<::lumen::eir::ConstantFloatOp>;
using eir_constant_binary = ValueBuilder<::lumen::eir::ConstantBinaryOp>;
using eir_constant_tuple = ValueBuilder<::lumen::eir::ConstantTupleOp>;
using eir_constant_list = ValueBuilder<::lumen::eir::ConstantListOp>;
using eir_trace_construct = ValueBuilder<::lumen::eir::TraceConstructOp>;

namespace lumen {
namespace eir {

bool isa_eir_type(Type t);
bool isa_std_type(Type t);
bool isa_llvm_type(Type t);

using BuildCastFnT = std::function<Optional<Type>(OpBuilder &)>;

struct EirTypeConverter : public mlir::TypeConverter {
  using TypeConverter::TypeConverter;

  EirTypeConverter(unsigned pointerSizeInBits, LLVMTypeConverter &tc)
      : pointerSizeInBits(pointerSizeInBits), typeConverter(tc) {
    addTargetMaterialization(materializeCast);
  }

  static Optional<Value> materializeCast(OpBuilder &builder, Type resultType,
                                         ValueRange inputs, Location loc) {
    if (inputs.size() != 1) return llvm::None;
    return builder.create<CastOp>(loc, inputs[0], resultType).getResult();
  }

  /// This function is used to determine which type to cast to when operating
  /// on values of the given input types. If the types can be operated on directly
  /// in LLVM IR, then this will return Some(Type) which the caller can then use
  /// to insert casts where appropriate. If the types cannot be operated on directly
  /// either due to incomplete type information, or because the types must use
  /// runtime-provided functionality to operate on, then this will return llvm::None,
  /// and the caller should cast the types to term type and use an appropriate runtime
  /// function for whatever operation it is lowering
  Optional<Type> coalesceOperandTypes(Type lhs, Type rhs);

  Type packFunctionResults(TargetInfo &targetInfo, ArrayRef<Type> types);

 private:
  unsigned pointerSizeInBits;
  LLVMTypeConverter &typeConverter;
};

Optional<Type> convertType(Type type, EirTypeConverter &converter,
                           TargetInfo &targetInfo);

class ConversionContext {
 public:
  explicit ConversionContext(MLIRContext *ctx, EirTypeConverter &tc,
                             TargetInfo &ti)
      : targetInfo(ti), typeConverter(tc), context(ctx) {}

  ConversionContext(const ConversionContext &ctx)
      : targetInfo(ctx.targetInfo),
        typeConverter(ctx.typeConverter),
        context(ctx.context) {}

  TargetInfo &targetInfo;
  EirTypeConverter &typeConverter;
  MLIRContext *context;

  LLVMType getUsizeType() const { return targetInfo.getUsizeType(); }
  LLVMType getI1Type() const { return targetInfo.getI1Type(); }
  LLVMType getI8Type() const { return targetInfo.getI8Type(); }
  LLVMType getI32Type() const { return targetInfo.getI32Type(); }
  LLVMType getI64Type() const { return targetInfo.getI64Type(); }
  LLVMType getDoubleType() const { return targetInfo.getDoubleType(); }
  LLVMType getTupleType(unsigned arity) const {
    return targetInfo.makeTupleType(arity);
  }
  LLVMType getTupleType(ArrayRef<LLVMType> elementTypes) const {
    return targetInfo.makeTupleType(elementTypes);
  }

  APInt encodeImmediateConstant(uint32_t type, uint64_t value);
  APInt encodeHeaderConstant(uint32_t type, uint64_t arity);

  APInt &getNilValue() const { return targetInfo.getNilValue(); }
  APInt &getNoneValue() const { return targetInfo.getNoneValue(); }
};

class OpConversionContext : public ConversionContext {
 public:
  explicit OpConversionContext(const ConversionContext &ctx,
                               ConversionPatternRewriter &cpr)
      : ConversionContext(ctx), rewriter(cpr) {}

  OpConversionContext(const OpConversionContext &ctx)
      : ConversionContext(ctx), rewriter(ctx.rewriter) {}

  ConversionPatternRewriter &rewriter;

  using ConversionContext::context;
  using ConversionContext::encodeHeaderConstant;
  using ConversionContext::encodeImmediateConstant;
  using ConversionContext::getDoubleType;
  using ConversionContext::getI1Type;
  using ConversionContext::getI32Type;
  using ConversionContext::getI64Type;
  using ConversionContext::getI8Type;
  using ConversionContext::getNilValue;
  using ConversionContext::getNoneValue;
  using ConversionContext::getTupleType;
  using ConversionContext::getUsizeType;
  using ConversionContext::targetInfo;
  using ConversionContext::typeConverter;

  Type getIntegerType(unsigned bitWidth = 0) const {
    if (bitWidth == 0)
      return rewriter.getIntegerType(targetInfo.pointerSizeInBits);
    else
      return rewriter.getIntegerType(bitWidth);
  }

  inline IntegerAttr getIntegerAttr(int64_t i) const {
    return rewriter.getIntegerAttr(getIntegerType(), i);
  }

  inline IntegerAttr getIntegerAttr(APInt &i) const {
    return rewriter.getIntegerAttr(getIntegerType(), i);
  }

  inline IntegerAttr getI1Attr(int64_t i) const {
    return rewriter.getIntegerAttr(rewriter.getI1Type(), i);
  }

  inline IntegerAttr getI8Attr(int64_t i) const {
    return rewriter.getIntegerAttr(getIntegerType(8), i);
  }

  inline IntegerAttr getI32Attr(int32_t i) const {
    return rewriter.getIntegerAttr(getIntegerType(32), i);
  }

  inline IntegerAttr getU32Attr(int32_t i) const {
    return rewriter.getIntegerAttr(getIntegerType(32),
                                   APInt(32, i, /*signed=*/false));
  }

  inline ArrayAttr getI64ArrayAttr(unsigned i) const {
    return rewriter.getI64ArrayAttr(i);
  }

  inline StringAttr getStringAttr(StringRef str) const {
    return rewriter.getStringAttr(str);
  }

  Operation *getOrInsertFunction(ModuleOp mod, StringRef symbol,
                                 LLVMType resultTy, ArrayRef<LLVMType> argTypes,
                                 ArrayRef<NamedAttribute> attrs = {}) const;

  Value getOrInsertGlobal(ModuleOp mod, StringRef name, LLVMType valueType,
                          Attribute value = Attribute(),
                          LLVM::Linkage linkage = LLVM::Linkage::Internal,
                          LLVM::ThreadLocalMode tlsMode =
                              LLVM::ThreadLocalMode::NotThreadLocal) const {
    return getOrInsertGlobal(mod, name, valueType, value, linkage, tlsMode,
                             /*isConstant=*/false);
  }
  Value getOrInsertGlobal(ModuleOp mod, StringRef name, LLVMType valueType,
                          Attribute value, LLVM::Linkage linkage,
                          LLVM::ThreadLocalMode tlsMode,
                          bool isConstant) const {
    auto savePoint = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(mod.getBody());
    auto global = getOrInsertGlobalOp(mod, name, valueType, value, linkage,
                                      tlsMode, isConstant);
    rewriter.restoreInsertionPoint(savePoint);
    return llvm_addressof(global);
  }
  Value getOrInsertGlobalConstant(
      ModuleOp mod, StringRef name, LLVMType valueType,
      Attribute value = Attribute(),
      LLVM::Linkage linkage = LLVM::Linkage::Internal,
      LLVM::ThreadLocalMode tlsMode =
          LLVM::ThreadLocalMode::NotThreadLocal) const {
    return getOrInsertGlobal(mod, name, valueType, value, linkage, tlsMode,
                             /*isConstant=*/true);
  }
  LLVM::GlobalOp getOrInsertGlobalOp(
      ModuleOp mod, StringRef name, LLVMType valueType,
      Attribute value = Attribute(),
      LLVM::Linkage linkage = LLVM::Linkage::Internal,
      LLVM::ThreadLocalMode tlsMode =
          LLVM::ThreadLocalMode::NotThreadLocal) const {
    return getOrInsertGlobalOp(mod, name, valueType, value, linkage, tlsMode,
                               /*isConstant=*/false);
  }
  LLVM::GlobalOp getOrInsertGlobalConstantOp(
      ModuleOp mod, StringRef name, LLVMType valueType,
      Attribute value = Attribute(),
      LLVM::Linkage linkage = LLVM::Linkage::Internal,
      LLVM::ThreadLocalMode tlsMode =
          LLVM::ThreadLocalMode::NotThreadLocal) const {
    return getOrInsertGlobalOp(mod, name, valueType, value, linkage, tlsMode,
                               /*isConstant=*/true);
  }
  LLVM::GlobalOp getOrInsertGlobalOp(ModuleOp mod, StringRef name,
                                     LLVMType valueTy, Attribute value,
                                     LLVM::Linkage linkage,
                                     LLVM::ThreadLocalMode tlsMode,
                                     bool isConstant) const {
    if (auto global = mod.lookupSymbol<LLVM::GlobalOp>(name)) return global;

    auto global = rewriter.create<LLVM::GlobalOp>(
        mod.getLoc(), valueTy, isConstant, linkage, tlsMode, name, value);
    return global;
  }
  LLVM::GlobalOp getOrInsertConstantString(ModuleOp mod, StringRef name,
                                           StringRef value) const {
    assert(!name.empty() && "cannot create unnamed global string!");

    // Create the global at the entry of the module.
    LLVM::GlobalOp global = mod.lookupSymbol<LLVM::GlobalOp>(name);
    if (!global) {
      auto strTy = LLVMType::getArrayTy(getI8Type(), value.size());

      PatternRewriter::InsertionGuard insertGuard(rewriter);
      rewriter.setInsertionPointToStart(mod.getBody());
      global =
          getOrInsertGlobalConstantOp(mod, name, strTy, getStringAttr(value));
    }

    return global;
  }
  LLVM::GlobalOp getOrInsertGlobalString(ModuleOp mod, StringRef name,
                                         StringRef value) const;

  Value buildMalloc(ModuleOp mod, LLVMType ty, unsigned allocTy,
                    Value arity) const;

  Value encodeList(Value cons, bool isLiteral = false) const;
  Value encodeBox(Value val) const;
  Value encodeLiteral(Value val) const;
  Value encodeImmediate(ModuleOp mod, Location loc, OpaqueTermType ty,
                        Value val) const;
  Value decodeBox(LLVMType innerTy, Value box) const;
  Value decodeList(Value box) const;
  Value decodeImmediate(Value val) const;
};

template <typename Op>
class RewritePatternContext : public OpConversionContext {
 public:
  explicit RewritePatternContext(const ConversionContext &ctx, Op &op,
                                 ConversionPatternRewriter &cpr)
      : OpConversionContext(ctx, cpr),
        op(op),
        parentModule(op.template getParentOfType<ModuleOp>()),
        scope(cpr, op.getLoc()) {}

  Op &op;
  ModuleOp parentModule;
  ScopedContext scope;

  using OpConversionContext::context;
  using OpConversionContext::decodeBox;
  using OpConversionContext::decodeImmediate;
  using OpConversionContext::decodeList;
  using OpConversionContext::encodeBox;
  using OpConversionContext::encodeHeaderConstant;
  using OpConversionContext::encodeImmediateConstant;
  using OpConversionContext::encodeList;
  using OpConversionContext::encodeLiteral;
  using OpConversionContext::getDoubleType;
  using OpConversionContext::getI1Attr;
  using OpConversionContext::getI1Type;
  using OpConversionContext::getI32Attr;
  using OpConversionContext::getI32Type;
  using OpConversionContext::getI64ArrayAttr;
  using OpConversionContext::getI64Type;
  using OpConversionContext::getI8Attr;
  using OpConversionContext::getI8Type;
  using OpConversionContext::getIntegerAttr;
  using OpConversionContext::getNilValue;
  using OpConversionContext::getNoneValue;
  using OpConversionContext::getStringAttr;
  using OpConversionContext::getTupleType;
  using OpConversionContext::getUsizeType;
  using OpConversionContext::rewriter;
  using OpConversionContext::targetInfo;
  using OpConversionContext::typeConverter;

  inline const ModuleOp &getModule() const { return parentModule; }

  Operation *getOrInsertFunction(StringRef symbol, LLVMType resultTy,
                                 ArrayRef<LLVMType> argTypes,
                                 ArrayRef<NamedAttribute> attrs = {}) const {
    ModuleOp mod = getModule();
    return OpConversionContext::getOrInsertFunction(mod, symbol, resultTy,
                                                    argTypes, attrs);
  }
  Value getOrInsertGlobal(StringRef name, LLVMType valueType, Attribute value,
                          LLVM::Linkage linkage, LLVM::ThreadLocalMode tlsMode,
                          bool isConstant) const {
    ModuleOp mod = getModule();
    return OpConversionContext::getOrInsertGlobal(mod, name, valueType, value,
                                                  linkage, tlsMode, isConstant);
  }
  Value getOrInsertGlobal(StringRef name, LLVMType valueType,
                          Attribute value = Attribute(),
                          LLVM::Linkage linkage = LLVM::Linkage::Internal,
                          LLVM::ThreadLocalMode tlsMode =
                              LLVM::ThreadLocalMode::NotThreadLocal) const {
    ModuleOp mod = getModule();
    return OpConversionContext::getOrInsertGlobal(mod, name, valueType, value,
                                                  linkage, tlsMode);
  }
  Value getOrInsertGlobalConstant(
      StringRef name, LLVMType valueType, Attribute value = Attribute(),
      LLVM::Linkage linkage = LLVM::Linkage::Internal,
      LLVM::ThreadLocalMode tlsMode =
          LLVM::ThreadLocalMode::NotThreadLocal) const {
    ModuleOp mod = getModule();
    return OpConversionContext::getOrInsertGlobal(mod, name, valueType, value,
                                                  linkage, tlsMode);
  }

  LLVM::GlobalOp getOrInsertGlobalOp(
      StringRef name, LLVMType valueType, Attribute value = Attribute(),
      LLVM::Linkage linkage = LLVM::Linkage::Internal,
      LLVM::ThreadLocalMode tlsMode =
          LLVM::ThreadLocalMode::NotThreadLocal) const {
    ModuleOp mod = getModule();
    return OpConversionContext::getOrInsertGlobalOp(
        mod, name, valueType, value, linkage, tlsMode, /*isConstant=*/false);
  }
  LLVM::GlobalOp getOrInsertGlobalConstantOp(
      StringRef name, LLVMType valueType, Attribute value = Attribute(),
      LLVM::Linkage linkage = LLVM::Linkage::Internal,
      LLVM::ThreadLocalMode tlsMode =
          LLVM::ThreadLocalMode::NotThreadLocal) const {
    ModuleOp mod = getModule();
    return OpConversionContext::getOrInsertGlobalOp(
        mod, name, valueType, value, linkage, tlsMode, /*isConstant=*/true);
  }
  LLVM::GlobalOp getOrInsertGlobalOp(StringRef name, LLVMType valueTy,
                                     Attribute value, LLVM::Linkage linkage,
                                     LLVM::ThreadLocalMode tlsMode,
                                     bool isConstant) const {
    ModuleOp mod = getModule();
    return OpConversionContext::getOrInsertGlobalOp(
        mod, name, valueTy, value, linkage, tlsMode, isConstant);
  }
  LLVM::GlobalOp getOrInsertConstantString(StringRef name,
                                           StringRef value) const {
    ModuleOp mod = getModule();
    return OpConversionContext::getOrInsertConstantString(mod, name, value);
  }
  LLVM::GlobalOp getOrInsertGlobalString(StringRef name,
                                         StringRef value) const {
    ModuleOp mod = getModule();
    return OpConversionContext::getOrInsertGlobalString(mod, name, value);
  }
  Value buildMalloc(LLVMType ty, unsigned allocTy, Value arity) const {
    ModuleOp mod = getModule();
    return OpConversionContext::buildMalloc(mod, ty, allocTy, arity);
  }
  Value encodeImmediate(OpaqueTermType ty, Value val) const {
    ModuleOp mod = getModule();
    return OpConversionContext::encodeImmediate(mod, val.getLoc(), ty, val);
  }
};

template <typename Op>
class EIROpConversion : public mlir::OpConversionPattern<Op> {
 public:
  explicit EIROpConversion(MLIRContext *context, EirTypeConverter &tc,
                           TargetInfo &ti, mlir::PatternBenefit benefit = 1)
      : mlir::OpConversionPattern<Op>::OpConversionPattern(context, benefit),
        ctx(context, tc, ti) {}

 private:
  ConversionContext ctx;

 protected:
  ConversionContext const &getContext() const { return ctx; }

  RewritePatternContext<Op> getRewriteContext(
      Op op, ConversionPatternRewriter &rewriter) const {
    return RewritePatternContext<Op>(ctx, op, rewriter);
  }
};

}  // namespace eir
}  // namespace lumen

#endif
