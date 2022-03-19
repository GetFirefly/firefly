#ifndef LUMEN_EIR_CONVERSION_CONVERSION_SUPPORT_H
#define LUMEN_EIR_CONVERSION_CONVERSION_SUPPORT_H

#include "llvm/Support/Casting.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"

#include "lumen/EIR/Conversion/TargetPlatformBuilder.h"
#include "lumen/EIR/IR/EIRAttributes.h"
#include "lumen/EIR/IR/EIROps.h"
#include "lumen/EIR/IR/EIRTypes.h"

using ::llvm::cast;
using ::llvm::dyn_cast_or_null;
using ::llvm::isa;
using ::llvm::SmallVectorImpl;
using ::llvm::StringSwitch;
using ::mlir::CallableOpInterface;
using ::mlir::CallInterfaceCallable;
using ::mlir::ConversionPatternRewriter;
using ::mlir::Float64Type;
using ::mlir::LLVMTypeConverter;
using ::mlir::LogicalResult;
using ::mlir::PatternRewriter;
using ::mlir::success;
using ::mlir::SymbolTable;
using ::mlir::UnitAttr;
using ::mlir::edsc::OperationBuilder;
using ::mlir::edsc::ScopedContext;
using ::mlir::edsc::ValueBuilder;
using ::mlir::LLVM::LLVMArrayType;
using ::mlir::LLVM::LLVMPointerType;
using ::mlir::LLVM::LLVMStructType;
using ::mlir::LLVM::LLVMTokenType;
using ::mlir::LLVM::LLVMVoidType;

using SignednessSemantics = ::mlir::IntegerType::SignednessSemantics;

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
using llvm_addrspacecast = ValueBuilder<LLVM::AddrSpaceCastOp>;
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
using eir_map = OperationBuilder<::lumen::eir::MapOp>;
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
    /// on values of the given input types. If the types can be operated on
    /// directly in LLVM IR, then this will return Some(Type) which the caller
    /// can then use to insert casts where appropriate. If the types cannot be
    /// operated on directly either due to incomplete type information, or
    /// because the types must use runtime-provided functionality to operate on,
    /// then this will return llvm::None, and the caller should cast the types
    /// to term type and use an appropriate runtime function for whatever
    /// operation it is lowering
    Optional<Type> coalesceOperandTypes(Type lhs, Type rhs);

    Type packFunctionResults(TargetPlatform &platform, ArrayRef<Type> types);

    Optional<Type> deferTypeConversion(Type type) {
        return typeConverter.convertType(type);
    }

    MLIRContext &getContext() { return typeConverter.getContext(); }

    inline unsigned getPointerWidth() { return pointerSizeInBits; }

   private:
    unsigned pointerSizeInBits;
    LLVMTypeConverter &typeConverter;
};

Optional<Type> convertType(Type type, EirTypeConverter &converter,
                           TargetPlatform &platform);

class OpConversionContext : public TargetPlatformBuilder {
   public:
    explicit OpConversionContext(ConversionPatternRewriter &cpr,
                                 TargetPlatform &platform,
                                 EirTypeConverter &typeConverter)
        : TargetPlatformBuilder(cpr, platform), typeConverter(typeConverter) {}

    OpConversionContext(const OpConversionContext &ctx)
        : TargetPlatformBuilder(ctx), typeConverter(ctx.typeConverter) {}

    EirTypeConverter typeConverter;

    inline IntegerAttr getI1Attr(int64_t i) {
        return getIntegerAttr(getI1Type(), i);
    }

    inline IntegerAttr getI8Attr(int8_t i) { return getI8IntegerAttr(i); }

    inline IntegerAttr getI32Attr(int32_t i) { return getI32IntegerAttr(i); }

    inline ::mlir::StringAttr getStringAttr(StringRef str) {
        return getStringAttr(str);
    }

    Value getOrInsertGlobal(
        ModuleOp mod, StringRef name, Type valueType,
        Attribute value = Attribute(),
        LLVM::Linkage linkage = LLVM::Linkage::Internal,
        LLVM::ThreadLocalMode tlsMode = LLVM::ThreadLocalMode::NotThreadLocal) {
        return getOrInsertGlobal(mod, name, valueType, value, linkage, tlsMode,
                                 /*isConstant=*/false);
    }
    Value getOrInsertGlobal(ModuleOp mod, StringRef name, Type valueType,
                            Attribute value, LLVM::Linkage linkage,
                            LLVM::ThreadLocalMode tlsMode, bool isConstant) {
        auto savePoint = saveInsertionPoint();
        auto &body = mod.body();
        setInsertionPointToStart(&body.front());
        auto global = getOrInsertGlobalOp(mod, name, valueType, value, linkage,
                                          tlsMode, isConstant);
        restoreInsertionPoint(savePoint);
        return llvm_addressof(global);
    }
    Value getOrInsertGlobalConstant(
        ModuleOp mod, StringRef name, Type valueType,
        Attribute value = Attribute(),
        LLVM::Linkage linkage = LLVM::Linkage::Internal,
        LLVM::ThreadLocalMode tlsMode = LLVM::ThreadLocalMode::NotThreadLocal) {
        return getOrInsertGlobal(mod, name, valueType, value, linkage, tlsMode,
                                 /*isConstant=*/true);
    }
    LLVM::GlobalOp getOrInsertGlobalOp(
        ModuleOp mod, StringRef name, Type valueType,
        Attribute value = Attribute(),
        LLVM::Linkage linkage = LLVM::Linkage::Internal,
        LLVM::ThreadLocalMode tlsMode = LLVM::ThreadLocalMode::NotThreadLocal) {
        return getOrInsertGlobalOp(mod, name, valueType, value, linkage,
                                   tlsMode,
                                   /*isConstant=*/false);
    }
    LLVM::GlobalOp getOrInsertGlobalConstantOp(
        ModuleOp mod, StringRef name, Type valueType,
        Attribute value = Attribute(),
        LLVM::Linkage linkage = LLVM::Linkage::Internal,
        LLVM::ThreadLocalMode tlsMode = LLVM::ThreadLocalMode::NotThreadLocal) {
        return getOrInsertGlobalOp(mod, name, valueType, value, linkage,
                                   tlsMode,
                                   /*isConstant=*/true);
    }
    LLVM::GlobalOp getOrInsertGlobalOp(ModuleOp mod, StringRef name,
                                       Type valueTy, Attribute value,
                                       LLVM::Linkage linkage,
                                       LLVM::ThreadLocalMode tlsMode,
                                       bool isConstant) {
        if (auto global = mod.lookupSymbol<LLVM::GlobalOp>(name)) return global;

        auto global = create<LLVM::GlobalOp>(mod.getLoc(), valueTy, isConstant,
                                             linkage, tlsMode, name, value);
        return global;
    }
    LLVM::GlobalOp getOrInsertConstantString(ModuleOp mod, StringRef name,
                                             StringRef value) {
        assert(!name.empty() && "cannot create unnamed global string!");

        // Create the global at the entry of the module.
        LLVM::GlobalOp global = mod.lookupSymbol<LLVM::GlobalOp>(name);
        if (!global) {
            auto strTy = LLVMArrayType::get(getI8Type(), value.size());

            auto ip = saveInsertionPoint();
            auto &body = mod.body();
            setInsertionPointToStart(&body.front());
            global = getOrInsertGlobalConstantOp(mod, name, strTy,
                                                 getStringAttr(value));
            restoreInsertionPoint(ip);
        }

        return global;
    }
    LLVM::GlobalOp getOrInsertGlobalString(ModuleOp mod, StringRef name,
                                           StringRef value) const;
};

template <typename Op>
class RewritePatternContext : public OpConversionContext {
   public:
    explicit RewritePatternContext(Op &op, ConversionPatternRewriter &cpr,
                                   TargetPlatform &platform,
                                   EirTypeConverter &typeConverter)
        : OpConversionContext(cpr, platform, typeConverter),
          op(op),
          parentModule(op->template getParentOfType<ModuleOp>()),
          scope(cpr, op.getLoc()) {}

    Op &op;
    ModuleOp parentModule;
    ScopedContext scope;

    inline const ModuleOp &getModule() const { return parentModule; }

    Operation *getOrInsertFunction(StringRef symbol, Type resultTy,
                                   ArrayRef<Type> argTypes,
                                   ArrayRef<NamedAttribute> attrs = {}) {
        ModuleOp mod = getModule();
        return OpConversionContext::getOrInsertFunction(mod, symbol, resultTy,
                                                        argTypes, attrs);
    }
    Value getOrInsertGlobal(StringRef name, Type valueType, Attribute value,
                            LLVM::Linkage linkage,
                            LLVM::ThreadLocalMode tlsMode,
                            bool isConstant) const {
        ModuleOp mod = getModule();
        return OpConversionContext::getOrInsertGlobal(
            mod, name, valueType, value, linkage, tlsMode, isConstant);
    }
    Value getOrInsertGlobal(
        StringRef name, Type valueType, Attribute value = Attribute(),
        LLVM::Linkage linkage = LLVM::Linkage::Internal,
        LLVM::ThreadLocalMode tlsMode = LLVM::ThreadLocalMode::NotThreadLocal) {
        ModuleOp mod = getModule();
        return OpConversionContext::getOrInsertGlobal(mod, name, valueType,
                                                      value, linkage, tlsMode);
    }
    Value getOrInsertGlobalConstant(
        StringRef name, Type valueType, Attribute value = Attribute(),
        LLVM::Linkage linkage = LLVM::Linkage::Internal,
        LLVM::ThreadLocalMode tlsMode =
            LLVM::ThreadLocalMode::NotThreadLocal) const {
        ModuleOp mod = getModule();
        return OpConversionContext::getOrInsertGlobal(mod, name, valueType,
                                                      value, linkage, tlsMode);
    }

    LLVM::GlobalOp getOrInsertGlobalOp(
        StringRef name, Type valueType, Attribute value = Attribute(),
        LLVM::Linkage linkage = LLVM::Linkage::Internal,
        LLVM::ThreadLocalMode tlsMode = LLVM::ThreadLocalMode::NotThreadLocal) {
        ModuleOp mod = getModule();
        return OpConversionContext::getOrInsertGlobalOp(mod, name, valueType,
                                                        value, linkage, tlsMode,
                                                        /*isConstant=*/false);
    }
    LLVM::GlobalOp getOrInsertGlobalConstantOp(
        StringRef name, Type valueType, Attribute value = Attribute(),
        LLVM::Linkage linkage = LLVM::Linkage::Internal,
        LLVM::ThreadLocalMode tlsMode = LLVM::ThreadLocalMode::NotThreadLocal) {
        ModuleOp mod = getModule();
        return OpConversionContext::getOrInsertGlobalOp(
            mod, name, valueType, value, linkage, tlsMode, /*isConstant=*/true);
    }
    LLVM::GlobalOp getOrInsertGlobalOp(StringRef name, Type valueTy,
                                       Attribute value, LLVM::Linkage linkage,
                                       LLVM::ThreadLocalMode tlsMode,
                                       bool isConstant) {
        ModuleOp mod = getModule();
        return OpConversionContext::getOrInsertGlobalOp(
            mod, name, valueTy, value, linkage, tlsMode, isConstant);
    }
    LLVM::GlobalOp getOrInsertConstantString(StringRef name, StringRef value) {
        ModuleOp mod = getModule();
        return OpConversionContext::getOrInsertConstantString(mod, name, value);
    }
    LLVM::GlobalOp getOrInsertGlobalString(StringRef name, StringRef value) {
        ModuleOp mod = getModule();
        return OpConversionContext::getOrInsertGlobalString(mod, name, value);
    }
    Value buildMalloc(Type ty, unsigned allocTy, Value arity) {
        ModuleOp mod = getModule();
        return OpConversionContext::buildMalloc(mod, ty, allocTy, arity);
    }
    Value encodeImmediate(Type ty, Value val) {
        ModuleOp mod = getModule();
        return OpConversionContext::encodeImmediate(mod, val.getLoc(), ty, val);
    }
};

template <typename Op>
class EIROpConversion : public mlir::OpConversionPattern<Op> {
   public:
    explicit EIROpConversion(MLIRContext *context, EirTypeConverter &tc,
                             TargetPlatform &platform,
                             mlir::PatternBenefit benefit = 1)
        : mlir::OpConversionPattern<Op>::OpConversionPattern(context, benefit),
          platform(platform),
          typeConverter(tc) {}

   private:
    TargetPlatform platform;
    EirTypeConverter typeConverter;

   protected:
    const TargetPlatform &getPlatform() const { return platform; }
    const EirTypeConverter &getTypeConverter() const { return typeConverter; }

    RewritePatternContext<Op> getRewriteContext(
        Op op, ConversionPatternRewriter &rewriter) const {
        TargetPlatform p(platform);
        EirTypeConverter tc(typeConverter);
        return RewritePatternContext<Op>(op, rewriter, p, tc);
    }
};

}  // namespace eir
}  // namespace lumen

#endif
