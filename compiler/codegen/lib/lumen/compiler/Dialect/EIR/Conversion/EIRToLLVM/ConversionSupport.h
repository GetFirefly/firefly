#ifndef LUMEN_EIR_CONVERSION_CONVERSION_SUPPORT_H
#define LUMEN_EIR_CONVERSION_CONVERSION_SUPPORT_H

#include "llvm/Target/TargetMachine.h"
#include "lumen/compiler/Dialect/EIR/IR/EIRAttributes.h"
#include "lumen/compiler/Dialect/EIR/IR/EIROps.h"
#include "lumen/compiler/Dialect/EIR/IR/EIRTypes.h"
#include "lumen/compiler/Target/TargetInfo.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/EDSC/Intrinsics.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"

using ::llvm::TargetMachine;
using ::mlir::ConversionPatternRewriter;
using ::mlir::LLVMTypeConverter;
using ::mlir::LogicalResult;
using ::mlir::edsc::intrinsics::OperationBuilder;
using ::mlir::edsc::intrinsics::ValueBuilder;
using ::mlir::LLVM::LLVMDialect;
using ::mlir::LLVM::LLVMType;
using ::mlir::success;

namespace LLVM = ::mlir::LLVM;

using std_call = OperationBuilder<::mlir::CallOp>;
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
using llvm_call = OperationBuilder<LLVM::CallOp>;
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
using eir_cast = ValueBuilder<::lumen::eir::CastOp>;
using eir_malloc = ValueBuilder<::lumen::eir::MallocOp>;
using eir_nil = ValueBuilder<::lumen::eir::ConstantNilOp>;
using eir_none = ValueBuilder<::lumen::eir::ConstantNoneOp>;
using eir_constant_float = ValueBuilder<::lumen::eir::ConstantFloatOp>;
using eir_constant_binary = ValueBuilder<::lumen::eir::ConstantBinaryOp>;
using eir_constant_tuple = ValueBuilder<::lumen::eir::ConstantTupleOp>;
using eir_constant_list = ValueBuilder<::lumen::eir::ConstantListOp>;

namespace lumen {
namespace eir {

Optional<Type> convertType(Type type, LLVMTypeConverter &converter,
                           TargetInfo &targetInfo);

class ConversionContext {
 public:
  explicit ConversionContext(MLIRContext *ctx, LLVMTypeConverter &tc,
                             TargetInfo &ti)
      : dialect(tc.getDialect()),
        targetInfo(ti),
        typeConverter(tc),
        context(ctx) {}

  ConversionContext(const ConversionContext &ctx)
      : dialect(ctx.dialect),
        targetInfo(ctx.targetInfo),
        typeConverter(ctx.typeConverter),
        context(ctx.context) {}

  LLVMDialect *dialect;
  TargetInfo &targetInfo;
  LLVMTypeConverter &typeConverter;
  MLIRContext *context;

  LLVMType getUsizeType() const { return targetInfo.getUsizeType(); }
  LLVMType getI1Type() const { return targetInfo.getI1Type(); }
  LLVMType getI8Type() const { return targetInfo.getI8Type(); }
  LLVMType getI32Type() const { return targetInfo.getI32Type(); }
  LLVMType getTupleType(unsigned arity) const {
    return targetInfo.makeTupleType(dialect, arity);
  }
  LLVMType getTupleType(ArrayRef<LLVMType> elementTypes) const {
    return targetInfo.makeTupleType(dialect, elementTypes);
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
  using ConversionContext::dialect;
  using ConversionContext::encodeHeaderConstant;
  using ConversionContext::encodeImmediateConstant;
  using ConversionContext::getI1Type;
  using ConversionContext::getI32Type;
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
                                 LLVMType resultTy,
                                 ArrayRef<LLVMType> argTypes = {}) const;

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
  edsc::ScopedContext scope;

  using OpConversionContext::context;
  using OpConversionContext::decodeBox;
  using OpConversionContext::decodeImmediate;
  using OpConversionContext::decodeList;
  using OpConversionContext::dialect;
  using OpConversionContext::encodeBox;
  using OpConversionContext::encodeHeaderConstant;
  using OpConversionContext::encodeImmediateConstant;
  using OpConversionContext::encodeList;
  using OpConversionContext::encodeLiteral;
  using OpConversionContext::getI1Attr;
  using OpConversionContext::getI1Type;
  using OpConversionContext::getI32Attr;
  using OpConversionContext::getI32Type;
  using OpConversionContext::getI64ArrayAttr;
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

  inline Location getLoc() const { return op.getLoc(); }

  inline const ModuleOp &getModule() const { return parentModule; }

  Operation *getOrInsertFunction(StringRef symbol, LLVMType resultTy,
                                 ArrayRef<LLVMType> argTypes = {}) const {
    ModuleOp mod = getModule();
    return OpConversionContext::getOrInsertFunction(mod, symbol, resultTy,
                                                    argTypes);
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
    return OpConversionContext::encodeImmediate(mod, getLoc(), ty, val);
  }
};

template <typename Op>
class EIROpConversion : public mlir::OpConversionPattern<Op> {
 public:
  explicit EIROpConversion(MLIRContext *context, LLVMTypeConverter &tc,
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
