#include "lumen/compiler/Dialect/EIR/Conversion/EIRToLLVM/ConvertEIRToLLVM.h"

#include "llvm/Target/TargetMachine.h"
#include "lumen/compiler/Dialect/EIR/IR/EIRAttributes.h"
#include "lumen/compiler/Dialect/EIR/IR/EIROps.h"
#include "lumen/compiler/Dialect/EIR/IR/EIRTypes.h"
#include "lumen/compiler/Target/TargetInfo.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/Dialect/StandardOps/Ops.h"
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
using ::mlir::PatternMatchResult;
using ::mlir::edsc::intrinsics::OperationBuilder;
using ::mlir::edsc::intrinsics::ValueBuilder;
using ::mlir::LLVM::LLVMDialect;
using ::mlir::LLVM::LLVMType;

namespace LLVM = ::mlir::LLVM;

using llvm_add = ValueBuilder<LLVM::AddOp>;
using llvm_and = ValueBuilder<LLVM::AndOp>;
using llvm_or = ValueBuilder<LLVM::OrOp>;
using llvm_xor = ValueBuilder<LLVM::XOrOp>;
using llvm_shl = ValueBuilder<LLVM::ShlOp>;
using llvm_bitcast = ValueBuilder<LLVM::BitcastOp>;
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
using llvm_select = ValueBuilder<LLVM::SelectOp>;
using llvm_mul = ValueBuilder<LLVM::MulOp>;
using llvm_ptrtoint = ValueBuilder<LLVM::PtrToIntOp>;
using llvm_inttoptr = ValueBuilder<LLVM::IntToPtrOp>;
using llvm_sub = ValueBuilder<LLVM::SubOp>;
using llvm_undef = ValueBuilder<LLVM::UndefOp>;
using llvm_urem = ValueBuilder<LLVM::URemOp>;
using llvm_alloca = ValueBuilder<LLVM::AllocaOp>;
using llvm_return = OperationBuilder<LLVM::ReturnOp>;

namespace lumen {
namespace eir {

static bool isa_eir_type(Type t) {
  return inbounds(t.getKind(), Type::Kind::FIRST_EIR_TYPE,
                  Type::Kind::LAST_EIR_TYPE);
}

static bool isa_std_type(Type t) {
  return inbounds(t.getKind(), Type::Kind::FIRST_STANDARD_TYPE,
                  Type::Kind::LAST_STANDARD_TYPE);
}

static Optional<Type> convertType(Type type, LLVMTypeConverter &converter,
                                  TargetInfo &targetInfo) {
  if (!isa_eir_type(type)) return Optional<Type>();

  MLIRContext *context = type.getContext();
  auto termTy = targetInfo.getTermType();

  if (auto refTy = type.dyn_cast_or_null<RefType>()) {
    auto innerTy = converter.convertType(refTy.getInnerType()).cast<LLVMType>();
    return innerTy.getPointerTo();
  }

  if (auto boxTy = type.dyn_cast_or_null<BoxType>()) {
    auto boxedTy = converter.convertType(boxTy.getBoxedType()).cast<LLVMType>();
    return boxedTy.getPointerTo();
  }

  OpaqueTermType ty = type.cast<OpaqueTermType>();
  if (ty.isOpaque() || ty.isImmediate()) return termTy;

  if (ty.isNonEmptyList()) return targetInfo.getConsType();

  llvm::outs() << "\ntype: ";
  type.dump();
  llvm::outs() << "\n";
  assert(false && "unimplemented type conversion");

  return llvm::None;
}

template <typename T>
static LLVMType getPtrToElementType(T containerType,
                                    LLVMTypeConverter &lowering) {
  return lowering.convertType(containerType.getElementType())
      .template cast<LLVMType>()
      .getPointerTo();
}

static FlatSymbolRefAttr getOrInsertFunction(PatternRewriter &rewriter,
                                             ModuleOp mod, LLVMDialect *dialect,
                                             TargetInfo &targetInfo,
                                             StringRef symbol,
                                             LLVMType resultType,
                                             ArrayRef<LLVMType> argTypes = {}) {
  auto *context = mod.getContext();

  if (mod.lookupSymbol<mlir::FuncOp>(symbol))
    return SymbolRefAttr::get(symbol, context);

  if (mod.lookupSymbol<FuncOp>(symbol))
    return SymbolRefAttr::get(symbol, context);

  if (mod.lookupSymbol<LLVM::LLVMFuncOp>(symbol))
    return SymbolRefAttr::get(symbol, context);

  // Create a function declaration for the symbol
  auto fnTy = LLVMType::getFunctionTy(resultType, argTypes, /*isVarArg=*/false);

  // Insert the function into the body of the parent module.
  PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(mod.getBody());
  rewriter.create<LLVM::LLVMFuncOp>(mod.getLoc(), symbol, fnTy);
  return SymbolRefAttr::get(symbol, context);
}

/// Return a symbol reference to the printf function, inserting it into the
/// module if necessary.
static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter,
                                           ModuleOp mod, LLVMDialect *dialect,
                                           TargetInfo &targetInfo) {
  auto termTy = targetInfo.getTermType();
  ArrayRef<LLVMType> argTypes({termTy});
  return getOrInsertFunction(rewriter, mod, dialect, targetInfo,
                             "__lumen_builtin_printf", termTy, argTypes);
}

/// Return a value representing an access into a global string with the given
/// name, creating the string if necessary.
static Value getOrCreateGlobalString(Location loc, OpBuilder &builder,
                                     StringRef name, StringRef value,
                                     ModuleOp mod, LLVMDialect *dialect) {
  assert(!name.empty() && "cannot create unnamed global string!");

  auto extendedName = name.str();
  extendedName.append({'_', 'g', 'l', 'o', 'b', 'a', 'l'});

  auto i8PtrTy = LLVMType::getInt8PtrTy(dialect);
  auto i64Ty = LLVMType::getInt64Ty(dialect);
  auto indexTy = builder.getIndexType();

  // Create the global at the entry of the module.
  LLVM::GlobalOp globalConst;
  LLVM::GlobalOp global;
  if (!(globalConst = mod.lookupSymbol<LLVM::GlobalOp>(name))) {
    OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(mod.getBody());
    auto type =
        LLVMType::getArrayTy(LLVMType::getInt8Ty(dialect), value.size());
    globalConst = builder.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/true,
                                                 LLVM::Linkage::Internal, name,
                                                 builder.getStringAttr(value));
    auto ptrType = LLVMType::getInt8PtrTy(dialect);
    global = builder.create<LLVM::GlobalOp>(
        loc, ptrType, /*isConstant=*/false, LLVM::Linkage::Internal,
        StringRef(extendedName), Attribute());
    auto &initRegion = global.getInitializerRegion();
    auto *initBlock = builder.createBlock(&initRegion);

    // Get the pointer to the first character in the global string.
    auto globalPtr = builder.create<LLVM::AddressOfOp>(loc, globalConst);
    Value cst0 = llvm_constant(i64Ty, builder.getIntegerAttr(indexTy, 0));
    auto gepPtr = builder.create<LLVM::GEPOp>(loc, i8PtrTy, globalPtr,
                                              ArrayRef<Value>({cst0, cst0}));
    builder.create<LLVM::ReturnOp>(loc, gepPtr.getResult());
  } else {
    global = mod.lookupSymbol<LLVM::GlobalOp>(StringRef(extendedName));
  }

  return llvm_addressof(global);
}

// Builds IR to construct a boxed term that points to the given header value,
// it is expected that the header value is a pointer value, not an immediate.
//
// The type of the resulting term is Term
static Value make_box(OpBuilder &builder, edsc::ScopedContext &context,
                      LLVMTypeConverter &converter, TargetInfo &targetInfo,
                      Value header) {
  auto headerTy = targetInfo.getHeaderType();
  Value headerPtrInt = llvm_ptrtoint(headerTy, header);
  auto boxTag = targetInfo.boxTag();
  auto tagAttr = builder.getIntegerAttr(
      builder.getIntegerType(targetInfo.pointerSizeInBits), boxTag);
  Value boxTagConst = llvm_constant(headerTy, tagAttr);
  return llvm_or(headerPtrInt, boxTagConst);
}

// Builds IR to construct a boxed list term
// it is expected that the cons cell value is a pointer value, not an immediate.
//
// The type of the resulting term is Term
static Value make_list(OpBuilder &builder, edsc::ScopedContext &context,
                       LLVMTypeConverter &converter, TargetInfo &targetInfo,
                       Value cons) {
  auto headerTy = targetInfo.getHeaderType();
  Value consPtrInt = llvm_ptrtoint(headerTy, cons);
  auto listTag = targetInfo.listTag();
  auto tagAttr = builder.getIntegerAttr(
      builder.getIntegerType(targetInfo.pointerSizeInBits), listTag);
  Value listTagConst = llvm_constant(headerTy, tagAttr);
  return llvm_or(consPtrInt, listTagConst);
}

static Value unbox(OpBuilder &builder, edsc::ScopedContext &context,
                   LLVMTypeConverter &converter, TargetInfo &targetInfo,
                   LLVMType innerTy, Value box) {
  auto intNTy = builder.getIntegerType(targetInfo.pointerSizeInBits);
  auto termTy = targetInfo.getTermType();
  auto boxTy = box.getType().cast<LLVMType>();
  assert(boxTy == termTy && "expected boxed pointer type");
  auto boxTag = targetInfo.boxTag();
  // No unboxing required, pointers are pointers
  if (boxTag == 0) {
    return llvm_inttoptr(innerTy, box);
  }
  auto tagAttr = builder.getIntegerAttr(intNTy, boxTag);
  Value boxTagConst = llvm_constant(termTy, tagAttr);
  auto neg1Attr = builder.getIntegerAttr(intNTy, -1);
  Value neg1Const = llvm_constant(termTy, neg1Attr);
  Value untagged = llvm_and(box, llvm_xor(boxTagConst, neg1Const));
  return llvm_inttoptr(innerTy, untagged);
}

static Value unbox_list(OpBuilder &builder, edsc::ScopedContext &context,
                        LLVMTypeConverter &converter, TargetInfo &targetInfo,
                        LLVMType innerTy, Value box) {
  auto intNTy = builder.getIntegerType(targetInfo.pointerSizeInBits);
  auto termTy = targetInfo.getTermType();
  auto listTag = targetInfo.listTag();
  auto listTagAttr = builder.getIntegerAttr(intNTy, listTag);
  Value listTagConst = llvm_constant(termTy, listTagAttr);
  auto listMask = targetInfo.listMask();
  auto listMaskAttr = builder.getIntegerAttr(intNTy, listMask);
  Value listMaskConst = llvm_constant(termTy, listMaskAttr);
  auto neg1Attr = builder.getIntegerAttr(intNTy, -1);
  Value neg1Const = llvm_constant(termTy, neg1Attr);
  Value untagged = llvm_and(box, llvm_xor(listMaskConst, neg1Const));
  return llvm_inttoptr(innerTy, untagged);
}

namespace {

/// Conversion Patterns

template <typename Op>
class EIROpConversion : public mlir::OpConversionPattern<Op> {
 public:
  explicit EIROpConversion(MLIRContext *context, LLVMTypeConverter &converter_,
                           TargetInfo &targetInfo_,
                           mlir::PatternBenefit benefit = 1)
      : mlir::OpConversionPattern<Op>::OpConversionPattern(context, benefit),
        typeConverter(converter_),
        targetInfo(targetInfo_) {}

 protected:
  LLVMTypeConverter &typeConverter;
  TargetInfo &targetInfo;
};

struct TraceConstructOpConversion : public EIROpConversion<TraceConstructOp> {
  using EIROpConversion::EIROpConversion;

  PatternMatchResult matchAndRewrite(
      TraceConstructOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    LLVMDialect *dialect = typeConverter.getDialect();
    ModuleOp parentModule = op.getParentOfType<ModuleOp>();
    auto termTy = targetInfo.getHeaderType();
    ArrayRef<LLVMType> argTypes({});
    auto callee = getOrInsertFunction(
        rewriter, parentModule, dialect, targetInfo,
        "__lumen_builtin_trace_construct", termTy, argTypes);

    rewriter.replaceOpWithNewOp<mlir::CallOp>(op, callee,
                                              ArrayRef<Type>{termTy}, operands);
    return matchSuccess();
  }
};

struct TraceCaptureOpConversion : public EIROpConversion<TraceCaptureOp> {
  using EIROpConversion::EIROpConversion;

  PatternMatchResult matchAndRewrite(
      TraceCaptureOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    LLVMDialect *dialect = typeConverter.getDialect();
    ModuleOp parentModule = op.getParentOfType<ModuleOp>();
    auto termTy = targetInfo.getHeaderType();
    auto callee =
        getOrInsertFunction(rewriter, parentModule, dialect, targetInfo,
                            "__lumen_builtin_trace_capture", termTy);

    rewriter.replaceOpWithNewOp<mlir::CallOp>(op, callee,
                                              ArrayRef<Type>{termTy});
    return matchSuccess();
  }
};

struct IsTypeOpConversion : public EIROpConversion<IsTypeOp> {
  using EIROpConversion::EIROpConversion;

  PatternMatchResult matchAndRewrite(
      IsTypeOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    edsc::ScopedContext context(rewriter, op.getLoc());
    IsTypeOpOperandAdaptor adaptor(operands);

    LLVMDialect *dialect = typeConverter.getDialect();
    ModuleOp parentModule = op.getParentOfType<ModuleOp>();
    auto termTy = targetInfo.getHeaderType();
    auto int1Ty = LLVMType::getInt1Ty(dialect);
    auto int32Ty = LLVMType::getIntNTy(dialect, 32);
    ArrayRef<LLVMType> argTypes({int32Ty, termTy});

    auto matchType = op.getMatchType().cast<OpaqueTermType>();
    if (matchType.isBox()) {
      auto boxType = matchType.cast<BoxType>();
      auto boxedType = boxType.getBoxedType();
      // Lists
      if (boxedType.isa<ConsType>()) {
        auto listTag = targetInfo.listTag();
        auto listTagAttr = rewriter.getIntegerAttr(
            rewriter.getIntegerType(targetInfo.pointerSizeInBits), listTag);
        Value listTagConst = llvm_constant(termTy, listTagAttr);
        auto listMask = targetInfo.listMask();
        auto listMaskAttr = rewriter.getIntegerAttr(
            rewriter.getIntegerType(targetInfo.pointerSizeInBits), listMask);
        Value listMaskConst = llvm_constant(termTy, listMaskAttr);
        Value masked = llvm_and(adaptor.value(), listMaskConst);
        rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(op, LLVM::ICmpPredicate::eq,
                                                  listTagConst, masked);
      } else {
        auto matchKind = boxedType.getForeignKind();
        auto matchAttr =
            rewriter.getIntegerAttr(rewriter.getIntegerType(32), matchKind);
        Value matchConst = llvm_constant(int32Ty, matchAttr);
        auto callee = getOrInsertFunction(
            rewriter, parentModule, dialect, targetInfo,
            "__lumen_builtin_is_boxed_type", int1Ty, argTypes);
        ArrayRef<Value> isTypeOperands({matchConst, adaptor.value()});
        auto isType = rewriter.create<mlir::CallOp>(
            op.getLoc(), callee, ArrayRef<Type>({int1Ty}), isTypeOperands);
        rewriter.replaceOp(op, isType.getResults());
      }
    } else {
      auto matchKind = matchType.getForeignKind();
      auto matchAttr =
          rewriter.getIntegerAttr(rewriter.getIntegerType(32), matchKind);
      Value matchConst = llvm_constant(int32Ty, matchAttr);
      auto callee =
          getOrInsertFunction(rewriter, parentModule, dialect, targetInfo,
                              "__lumen_builtin_is_type", int1Ty, argTypes);
      ArrayRef<Value> isTypeOperands({matchConst, adaptor.value()});
      auto isType = rewriter.create<mlir::CallOp>(
          op.getLoc(), callee, ArrayRef<Type>({int1Ty}), isTypeOperands);
      rewriter.replaceOp(op, isType.getResults());
    }

    return matchSuccess();
  }
};

struct YieldOpConversion : public EIROpConversion<YieldOp> {
  using EIROpConversion::EIROpConversion;

  PatternMatchResult matchAndRewrite(
      YieldOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    LLVMDialect *dialect = typeConverter.getDialect();
    ModuleOp parentModule = op.getParentOfType<ModuleOp>();
    auto termTy = targetInfo.getHeaderType();
    auto callee =
        getOrInsertFunction(rewriter, parentModule, dialect, targetInfo,
                            "__lumen_builtin_yield", termTy);

    rewriter.replaceOpWithNewOp<mlir::CallOp>(op, callee, ArrayRef<Type>({}));
    return matchSuccess();
  }
};

struct ReturnOpConversion : public EIROpConversion<ReturnOp> {
  using EIROpConversion::EIROpConversion;

  PatternMatchResult matchAndRewrite(
      ReturnOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::ReturnOp>(op, operands);
    return matchSuccess();
  }
};

struct BranchOpConversion : public EIROpConversion<eir::BranchOp> {
  using EIROpConversion::EIROpConversion;

  PatternMatchResult matchAndRewrite(
      eir::BranchOp op, ArrayRef<Value> _properOperands,
      ArrayRef<Block *> destinations, ArrayRef<ArrayRef<Value>> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto dest = destinations.front();
    auto destArgs = operands.front();
    rewriter.replaceOpWithNewOp<mlir::BranchOp>(op, dest, destArgs);
    return matchSuccess();
  }
};

// Need to lower condition to i1
struct CondBranchOpConversion : public EIROpConversion<eir::CondBranchOp> {
  using EIROpConversion::EIROpConversion;

  PatternMatchResult matchAndRewrite(
      eir::CondBranchOp op, ArrayRef<Value> properOperands,
      ArrayRef<Block *> destinations, ArrayRef<ArrayRef<Value>> operands,
      ConversionPatternRewriter &rewriter) const override {
    edsc::ScopedContext context(rewriter, op.getLoc());
    CondBranchOpOperandAdaptor adaptor(properOperands);

    auto cond = adaptor.condition();
    auto trueDest = op.getTrueDest();
    auto falseDest = op.getFalseDest();
    auto trueArgs = ValueRange(op.getTrueOperands());
    auto falseArgs = ValueRange(op.getFalseOperands());

    Value finalCond;
    bool requiresCondLowering = false;
    if (auto condTy = cond.getType().dyn_cast_or_null<LLVMType>()) {
      if (condTy.isIntegerTy(1)) {
        finalCond = cond;
      } else {
        requiresCondLowering = true;
      }
    } else {
      requiresCondLowering = true;
    }

    if (requiresCondLowering) {
      auto maskInfo = targetInfo.immediateMask();

      // We're building the equivalent of:
      //   (bool)(cond & IMMED_MASK)
      //
      //   or
      //
      //   (bool)((cond & IMMED_MASK) >> IMMED_SHIFT)
      //
      // This relies on the fact that 0 is false, and 1 is true,
      // both in the native representation and in our atom table
      auto headerTy = targetInfo.getHeaderType();
      auto maskAttr = rewriter.getIntegerAttr(
          rewriter.getIntegerType(targetInfo.pointerSizeInBits), maskInfo.mask);
      Value maskConst = llvm_constant(headerTy, maskAttr);
      Value maskedCond = llvm_and(cond, maskConst);
      if (maskInfo.requiresShift()) {
        auto shiftAttr = rewriter.getIntegerAttr(
            rewriter.getIntegerType(targetInfo.pointerSizeInBits),
            maskInfo.shift);
        Value shiftConst = llvm_constant(headerTy, shiftAttr);
        Value shiftedCond = llvm_shl(maskedCond, shiftConst);
        finalCond = llvm_trunc(targetInfo.getI1Type(), shiftedCond);
      } else {
        finalCond = llvm_trunc(targetInfo.getI1Type(), maskedCond);
      }
    }

    auto attrs = op.getAttrs();
    rewriter.replaceOpWithNewOp<LLVM::CondBrOp>(
        op, ValueRange({finalCond}), ArrayRef({trueDest, falseDest}),
        ArrayRef({trueArgs, falseArgs}), attrs);
    return matchSuccess();
  }
};

// The purpose of this conversion is to build a function that contains
// all of the prologue setup our Erlang functions need (in cases where
// this isn't a declaration). Specifically:
//
// - Check if reduction count is exceeded
// - Check if we should garbage collect
//   - If either of the above are true, yield
//
// TODO: Need to actually perform the above, right now we just handle
// the translation to mlir::FuncOp
struct FuncOpConversion : public EIROpConversion<eir::FuncOp> {
  using EIROpConversion::EIROpConversion;

  PatternMatchResult matchAndRewrite(
      eir::FuncOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    SmallVector<NamedAttribute, 2> attrs;
    for (auto fa : op.getAttrs()) {
      if (fa.first.is(SymbolTable::getSymbolAttrName()) ||
          fa.first.is(::mlir::impl::getTypeAttrName())) {
        continue;
      }
    }
    SmallVector<NamedAttributeList, 4> argAttrs;
    for (unsigned i = 0, e = op.getNumArguments(); i < e; ++i) {
      auto aa = ::mlir::impl::getArgAttrs(op, i);
      argAttrs.push_back(NamedAttributeList(aa));
    }
    auto newFunc = rewriter.create<mlir::FuncOp>(op.getLoc(), op.getName(),
                                                 op.getType(), attrs, argAttrs);
    rewriter.inlineRegionBefore(op.getBody(), newFunc.getBody(), newFunc.end());
    rewriter.eraseOp(op);
    return matchSuccess();
  }
};

struct PrintOpConversion : public EIROpConversion<PrintOp> {
  using EIROpConversion::EIROpConversion;

  PatternMatchResult matchAndRewrite(
      PrintOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // If print is called with no operands, just remove it for now
    if (operands.empty()) {
      rewriter.eraseOp(op);
      return matchSuccess();
    }

    LLVMDialect *dialect = typeConverter.getDialect();
    ModuleOp parentModule = op.getParentOfType<ModuleOp>();
    auto printfRef =
        getOrInsertPrintf(rewriter, parentModule, dialect, targetInfo);

    rewriter.replaceOpWithNewOp<mlir::CallOp>(
        op, printfRef, targetInfo.getTermType(), operands);
    return matchSuccess();
  }
};

struct UnreachableOpConversion : public EIROpConversion<UnreachableOp> {
  using EIROpConversion::EIROpConversion;

  PatternMatchResult matchAndRewrite(
      UnreachableOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::LLVM::UnreachableOp>(op, operands);
    return matchSuccess();
  }
};

struct CallOpConversion : public EIROpConversion<CallOp> {
  using EIROpConversion::EIROpConversion;

  PatternMatchResult matchAndRewrite(
      CallOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    CallOpOperandAdaptor adaptor(operands);

    LLVMDialect *dialect = typeConverter.getDialect();
    ModuleOp parentModule = op.getParentOfType<ModuleOp>();
    SmallVector<LLVMType, 2> argTypes;
    for (auto operand : operands) {
      argTypes.push_back(operand.getType().cast<LLVMType>());
    }
    auto opResultTypes = op.getResultTypes();
    SmallVector<Type, 2> resultTypes;
    LLVMType resultType;
    if (opResultTypes.size() == 1) {
      resultType =
          typeConverter.convertType(opResultTypes.front()).cast<LLVMType>();
      if (!resultType) {
        return matchFailure();
      }
      resultTypes.push_back(resultType);
    } else if (opResultTypes.size() > 1) {
      return matchFailure();
    }

    auto calleeName = op.getCallee();
    auto callee =
        getOrInsertFunction(rewriter, parentModule, dialect, targetInfo,
                            calleeName, resultType, argTypes);

    rewriter.replaceOpWithNewOp<mlir::CallOp>(op, callee, resultTypes,
                                              adaptor.operands());
    return matchSuccess();
  }
};

struct CmpEqOpConversion : public EIROpConversion<CmpEqOp> {
  using EIROpConversion::EIROpConversion;

  PatternMatchResult matchAndRewrite(
      CmpEqOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    edsc::ScopedContext context(rewriter, op.getLoc());
    CmpEqOpOperandAdaptor adaptor(operands);

    LLVMDialect *dialect = typeConverter.getDialect();
    ModuleOp parentModule = op.getParentOfType<ModuleOp>();
    auto termTy = targetInfo.getHeaderType();
    ArrayRef<LLVMType> argTypes({termTy, termTy});
    auto int1ty = LLVMType::getInt1Ty(dialect);
    auto callee =
        getOrInsertFunction(rewriter, parentModule, dialect, targetInfo,
                            "__lumen_builtin_cmpeq", int1ty, argTypes);

    auto lhs = adaptor.lhs();
    auto rhs = adaptor.rhs();
    ArrayRef<Value> args({lhs, rhs});
    auto callOp = rewriter.create<mlir::CallOp>(op.getLoc(), callee,
                                                ArrayRef<Type>{int1ty}, args);
    auto result = callOp.getResult(0);

    /*
    auto maskInfo = targetInfo.immediateMask();
    auto maskAttr =
    rewriter.getIntegerAttr(rewriter.getIntegerType(targetInfo.pointerSizeInBits),
    maskInfo.mask); Value maskConst = llvm_constant(termTy, maskAttr); Value
    maskedCond =  llvm_and(result, maskConst); Value loweredCond; if
    (maskInfo.requiresShift()) { auto shiftAttr =
    rewriter.getIntegerAttr(rewriter.getIntegerType(targetInfo.pointerSizeInBits),
    maskInfo.shift); Value shiftConst = llvm_constant(termTy, shiftAttr); Value
    shiftedCond = llvm_shl(maskedCond, shiftConst); loweredCond =
    llvm_trunc(targetInfo.getI1Type(), shiftedCond); } else { loweredCond =
    llvm_trunc(targetInfo.getI1Type(), maskedCond);
    }

    rewriter.replaceOp(op, loweredCond);
    */
    rewriter.replaceOp(op, result);
    return matchSuccess();
  }
};

struct GetElementPtrOpConversion : public EIROpConversion<GetElementPtrOp> {
  using EIROpConversion::EIROpConversion;

  PatternMatchResult matchAndRewrite(
      GetElementPtrOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    edsc::ScopedContext context(rewriter, op.getLoc());
    GetElementPtrOpOperandAdaptor adaptor(operands);

    Value base = adaptor.base();
    Type resultTyOrig = op.getType();
    auto resultTy = typeConverter.convertType(resultTyOrig).cast<LLVMType>();
    // Value ptr = unbox(rewriter, context, typeConverter, targetInfo, resultTy,
    // base);

    auto pointerSize = targetInfo.pointerSizeInBits;
    auto int32Ty = LLVMType::getIntNTy(typeConverter.getDialect(), 32);
    auto indexTy = rewriter.getIntegerType(32);
    auto indexAttr = rewriter.getIntegerAttr(indexTy, op.getIndex());
    Value cns0 = llvm_constant(int32Ty, rewriter.getIntegerAttr(indexTy, 0));
    Value index = llvm_constant(int32Ty, indexAttr);
    Value gep =
        llvm_gep(resultTy.getPointerTo(), base, ArrayRef<Value>({cns0, index}));

    rewriter.replaceOp(op, gep);
    return matchSuccess();
  }
};

struct LoadOpConversion : public EIROpConversion<LoadOp> {
  using EIROpConversion::EIROpConversion;

  PatternMatchResult matchAndRewrite(
      LoadOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    edsc::ScopedContext context(rewriter, op.getLoc());
    LoadOpOperandAdaptor adaptor(operands);

    Value ptr = adaptor.ref();
    Value load = llvm_load(ptr);

    rewriter.replaceOp(op, load);
    return matchSuccess();
  }
};

struct CastOpConversion : public EIROpConversion<CastOp> {
  using EIROpConversion::EIROpConversion;

  PatternMatchResult matchAndRewrite(
      CastOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    edsc::ScopedContext context(rewriter, op.getLoc());
    CastOpOperandAdaptor adaptor(operands);

    Value in = adaptor.input();
    Value out = op.getResult();

    LLVMType inTy = in.getType().cast<LLVMType>();
    Type origOutTy = out.getType();
    LLVMType outTy = typeConverter.convertType(origOutTy).cast<LLVMType>();

    // Remove redundant casts
    if (inTy == outTy) {
      rewriter.replaceOp(op, in);
      return matchSuccess();
    }

    auto termTy = targetInfo.getTermType();
    Value ptr;
    if (inTy == termTy && outTy.isPointerTy()) {
      // This is a cast from opaque term to pointer type, i.e. unboxing
      if (auto boxType = origOutTy.dyn_cast_or_null<BoxType>()) {
        if (boxType.getBoxedType().isa<ConsType>()) {
          // We're unboxing a list
          ptr = unbox_list(rewriter, context, typeConverter, targetInfo, outTy,
                           in);
        } else {
          ptr = unbox(rewriter, context, typeConverter, targetInfo, outTy, in);
        }
      } else {
        ptr = unbox(rewriter, context, typeConverter, targetInfo, outTy, in);
      }
      rewriter.replaceOp(op, ptr);
      return matchSuccess();
    }

    return matchFailure();
  }
};

struct ConstantFloatOpToStdConversion
    : public EIROpConversion<ConstantFloatOp> {
  using EIROpConversion::EIROpConversion;

  PatternMatchResult matchAndRewrite(
      ConstantFloatOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // We lower directly to LLVM when using packed floats
    if (targetInfo.requiresPackedFloats()) return matchFailure();

    // On nanboxed targets though, we can treat floats normally
    auto attr = op.getValue().cast<mlir::FloatAttr>();
    auto newAttr = rewriter.getF64FloatAttr(attr.getValueAsDouble());
    rewriter.replaceOpWithNewOp<mlir::ConstantOp>(op, newAttr);
    return matchSuccess();
  }
};

struct ConstantFloatOpConversion : public EIROpConversion<ConstantFloatOp> {
  using EIROpConversion::EIROpConversion;

  PatternMatchResult matchAndRewrite(
      ConstantFloatOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    edsc::ScopedContext context(rewriter, op.getLoc());

    auto attr = op.getValue().cast<FloatAttr>();
    auto ty = targetInfo.getFloatType();
    auto val =
        llvm_constant(ty, rewriter.getF64FloatAttr(attr.getValueAsDouble()));

    // On nanboxed targets, floats are treated normally
    if (!targetInfo.requiresPackedFloats()) {
      rewriter.replaceOp(op, {val});
      return matchSuccess();
    }

    // All other targets use boxed, packed floats
    // This requires generating a descriptor around the float,
    // which can then either be placed on the heap and boxed, or
    // passed by value on the stack and accessed directly
    auto headerTy = targetInfo.getHeaderType();
    APInt headerVal = targetInfo.encodeHeader(TypeKind::Float, 2);
    auto descTy = targetInfo.getFloatType();
    Value header =
        llvm_constant(headerTy, rewriter.getIntegerAttr(headerTy, headerVal));
    Value desc = llvm_undef(descTy);
    desc = llvm_insertvalue(descTy, desc, header, rewriter.getI64ArrayAttr(0));
    desc = llvm_insertvalue(descTy, desc, val, rewriter.getI64ArrayAttr(1));
    // NOTE: For now we aren't boxing the descriptor, any operations we lower
    // that reference values of our float type will need to insert the
    // appropriate operations to either box the value, or access the f64
    // contained within directly
    rewriter.replaceOp(op, desc);
    return matchSuccess();
  }
};

struct ConstantIntOpConversion : public EIROpConversion<ConstantIntOp> {
  using EIROpConversion::EIROpConversion;

  PatternMatchResult matchAndRewrite(
      ConstantIntOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    edsc::ScopedContext context(rewriter, op.getLoc());

    auto attr = op.getValue().cast<IntegerAttr>();
    auto ty = targetInfo.getFixnumType();
    auto i = (uint64_t)attr.getValue().getLimitedValue();
    auto taggedInt = targetInfo.encodeImmediate(TypeKind::Fixnum, i);
    auto val = llvm_constant(ty, rewriter.getIntegerAttr(ty, taggedInt));

    rewriter.replaceOp(op, {val});
    return matchSuccess();
  }
};

struct ConstantBigIntOpConversion : public EIROpConversion<ConstantBigIntOp> {
  using EIROpConversion::EIROpConversion;

  PatternMatchResult matchAndRewrite(
      ConstantBigIntOp _op, ArrayRef<Value> _operands,
      ConversionPatternRewriter &_rewriter) const override {
    assert(false && "ConstantBigIntOpConversion is unimplemented");
  }
};

struct ConstantAtomOpConversion : public EIROpConversion<ConstantAtomOp> {
  using EIROpConversion::EIROpConversion;

  PatternMatchResult matchAndRewrite(
      ConstantAtomOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    edsc::ScopedContext context(rewriter, op.getLoc());

    auto atomAttr = op.getValue().cast<AtomAttr>();
    auto id = (uint64_t)atomAttr.getValue().getLimitedValue();
    auto ty = targetInfo.getHeaderType();
    auto taggedAtom = targetInfo.encodeImmediate(TypeKind::Atom, id);
    auto val = llvm_constant(
        ty, rewriter.getIntegerAttr(rewriter.getIntegerType(64), taggedAtom));

    rewriter.replaceOp(op, {val});
    return matchSuccess();
  }
};

struct ConstantBinaryOpConversion : public EIROpConversion<ConstantBinaryOp> {
  using EIROpConversion::EIROpConversion;

  PatternMatchResult matchAndRewrite(
      ConstantBinaryOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    edsc::ScopedContext context(rewriter, op.getLoc());

    auto binAttr = op.getValue().cast<BinaryAttr>();
    auto bytes = binAttr.getValue();
    auto byteSize = bytes.size();
    auto headerRaw = binAttr.getHeader();
    auto flagsRaw = binAttr.getFlags();
    auto ty = targetInfo.getBinaryType();
    auto headerTy = targetInfo.getHeaderType();

    LLVMDialect *dialect = op.getContext()->getRegisteredDialect<LLVMDialect>();
    ModuleOp parentModule = op.getParentOfType<ModuleOp>();

    auto boxTag = targetInfo.boxTag();
    auto literalTag = targetInfo.literalTag();
    auto boxedLiteralTag = boxTag | literalTag;
    Value literalTagConst = llvm_constant(
        headerTy,
        rewriter.getIntegerAttr(rewriter.getIntegerType(64), boxedLiteralTag));

    // We use the SHA-1 hash of the value as the name of the global,
    // this provides a nice way to de-duplicate constant strings while
    // not requiring any global state
    auto name = binAttr.getHash();
    Value valPtr =
        getOrCreateGlobalString(context.getLocation(), context.getBuilder(),
                                name, bytes, parentModule, dialect);
    Value valPtrLoad = llvm_load(valPtr);
    Value header = llvm_constant(
        headerTy,
        rewriter.getIntegerAttr(rewriter.getIntegerType(64), headerRaw));
    Value flags = llvm_constant(
        headerTy,
        rewriter.getIntegerAttr(rewriter.getIntegerType(64), flagsRaw));

    Value allocN = llvm_constant(headerTy, rewriter.getI64IntegerAttr(1));
    Value descAlloc =
        llvm_alloca(ty.getPointerTo(), allocN, rewriter.getI64IntegerAttr(8));

    auto tyPtrTy = ty.getPointerTo();
    Value desc = llvm_undef(ty);
    desc = llvm_insertvalue(ty, desc, header, rewriter.getI64ArrayAttr(0));
    desc = llvm_insertvalue(ty, desc, flags, rewriter.getI64ArrayAttr(1));
    desc = llvm_insertvalue(ty, desc, valPtrLoad, rewriter.getI64ArrayAttr(2));
    llvm_store(desc, descAlloc);
    Value descPtrInt = llvm_ptrtoint(headerTy, descAlloc);
    Value boxedDescPtr = llvm_or(descPtrInt, literalTagConst);
    Value boxedDesc = llvm_bitcast(targetInfo.getTermType(), boxedDescPtr);

    rewriter.replaceOp(op, boxedDesc);
    return matchSuccess();
  }
};

struct ConstantNilOpConversion : public EIROpConversion<ConstantNilOp> {
  using EIROpConversion::EIROpConversion;

  PatternMatchResult matchAndRewrite(
      ConstantNilOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    edsc::ScopedContext context(rewriter, op.getLoc());

    auto ty = targetInfo.getTermType();
    auto val = llvm_constant(
        ty, rewriter.getIntegerAttr(ty, targetInfo.getNilValue()));

    rewriter.replaceOp(op, {val});
    return matchSuccess();
  }
};

struct ConstantNoneOpConversion : public EIROpConversion<ConstantNoneOp> {
  using EIROpConversion::EIROpConversion;

  PatternMatchResult matchAndRewrite(
      ConstantNoneOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    edsc::ScopedContext context(rewriter, op.getLoc());

    auto ty = targetInfo.getTermType();
    auto intNTy = rewriter.getIntegerType(targetInfo.pointerSizeInBits);
    auto val = llvm_constant(
        ty, rewriter.getIntegerAttr(intNTy, targetInfo.getNoneValue()));

    rewriter.replaceOp(op, {val});
    return matchSuccess();
  }
};

static bool lowerElementValues(edsc::ScopedContext &context,
                               ConversionPatternRewriter &rewriter,
                               TargetInfo &targetInfo,
                               ArrayRef<Attribute> &elements,
                               SmallVector<Value, 2> &elementValues,
                               SmallVector<LLVMType, 2> &elementTypes) {
  for (auto elementAttr : elements) {
    auto elementType = elementAttr.getType();
    if (auto atomAttr = elementAttr.dyn_cast_or_null<AtomAttr>()) {
      auto id = (uint64_t)atomAttr.getValue().getLimitedValue();
      auto ty = targetInfo.getAtomType();
      auto tagged = targetInfo.encodeImmediate(TypeKind::Atom, id);
      auto val = llvm_constant(ty, rewriter.getIntegerAttr(ty, tagged));
      elementTypes.push_back(ty);
      elementValues.push_back(val);
      continue;
    }
    if (auto boolAttr = elementAttr.dyn_cast_or_null<BoolAttr>()) {
      auto b = boolAttr.getValue();
      uint64_t id = b ? 1 : 0;
      auto ty = targetInfo.getAtomType();
      auto tagged = targetInfo.encodeImmediate(TypeKind::Atom, id);
      auto val = llvm_constant(ty, rewriter.getIntegerAttr(ty, tagged));
      elementTypes.push_back(ty);
      elementValues.push_back(val);
      continue;
    }
    if (auto intAttr = elementAttr.dyn_cast_or_null<IntegerAttr>()) {
      auto i = intAttr.getValue();
      assert(i.getBitWidth() <= targetInfo.pointerSizeInBits &&
             "support for bigint in constant aggregates not yet implemented");
      auto ty = targetInfo.getFixnumType();
      auto tagged =
          targetInfo.encodeImmediate(TypeKind::Fixnum, i.getLimitedValue());
      auto val = llvm_constant(ty, rewriter.getIntegerAttr(ty, tagged));
      elementTypes.push_back(ty);
      elementValues.push_back(val);
      continue;
    }
    if (auto floatAttr = elementAttr.dyn_cast_or_null<FloatAttr>()) {
      auto f = floatAttr.getValue().bitcastToAPInt();
      assert(!targetInfo.requiresPackedFloats() &&
             "support for packed floats in constant aggregates is not yet "
             "implemented");
      auto ty = targetInfo.getFloatType();
      auto val =
          llvm_constant(ty, rewriter.getIntegerAttr(ty, f.getLimitedValue()));
      elementTypes.push_back(ty);
      elementValues.push_back(val);
      continue;
    }
    return false;
  }

  return true;
}

struct ConstantTupleOpConversion : public EIROpConversion<ConstantTupleOp> {
  using EIROpConversion::EIROpConversion;

  PatternMatchResult matchAndRewrite(
      ConstantTupleOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    edsc::ScopedContext context(rewriter, op.getLoc());

    auto attr = op.getValue().cast<SeqAttr>();
    auto elements = attr.getValue();
    auto numElements = elements.size();

    SmallVector<LLVMType, 2> elementTypes;
    elementTypes.resize(numElements);
    SmallVector<Value, 2> elementValues;
    elementValues.resize(numElements);

    auto lowered = lowerElementValues(context, rewriter, targetInfo, elements,
                                      elementValues, elementTypes);
    assert(lowered && "unsupported element type in tuple constant");

    auto dialect = typeConverter.getDialect();
    auto ty = targetInfo.makeTupleType(dialect, elementTypes);

    Value desc = llvm_undef(ty);
    for (auto i = 0; i < numElements; i++) {
      auto val = elementValues[i];
      desc = llvm_insertvalue(ty, desc, val, rewriter.getI64ArrayAttr(i));
    }

    rewriter.replaceOp(op, desc);
    return matchSuccess();
  }
};

struct ConstantListOpConversion : public EIROpConversion<ConstantListOp> {
  using EIROpConversion::EIROpConversion;

  PatternMatchResult matchAndRewrite(
      ConstantListOp op, ArrayRef<mlir::Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    edsc::ScopedContext context(rewriter, op.getLoc());

    auto attr = op.getValue().cast<SeqAttr>();
    auto elements = attr.getValue();

    auto numElements = elements.size();

    // Lower to nil if empty list
    if (numElements == 0) {
      auto nilTy = targetInfo.getNilType();
      auto nil = targetInfo.getNilValue();
      auto val = llvm_constant(nilTy, rewriter.getIntegerAttr(nilTy, nil));
      rewriter.replaceOp(op, {val});
      return matchSuccess();
    }

    SmallVector<LLVMType, 2> elementTypes;
    elementTypes.resize(numElements);
    SmallVector<Value, 2> elementValues;
    elementValues.resize(numElements);

    auto lowered = lowerElementValues(context, rewriter, targetInfo, elements,
                                      elementValues, elementTypes);
    assert(lowered && "unsupported element type in list constant");

    auto consTy = targetInfo.getConsType();

    // Lower to single cons cell if elements <= 2
    if (numElements <= 2) {
      Value desc = llvm_undef(consTy);
      desc = llvm_insertvalue(consTy, desc, elementValues[0],
                              rewriter.getI64ArrayAttr(0));
      if (numElements == 2) {
        desc = llvm_insertvalue(consTy, desc, elementValues[1],
                                rewriter.getI64ArrayAttr(1));
      }
      rewriter.replaceOp(op, desc);
      return matchSuccess();
    }

    // Otherwise, we need to lower multiple cons cells, boxing those
    // that are not the head element
    unsigned cellsRequired = numElements;
    unsigned currentIndex = numElements;
    // Create final cons cell
    Value lastCons = llvm_undef(consTy);
    auto nilTy = targetInfo.getNilType();
    auto nil = targetInfo.getNilValue();
    auto nilVal = llvm_constant(nilTy, rewriter.getIntegerAttr(nilTy, nil));
    lastCons =
        llvm_insertvalue(consTy, lastCons, nilVal, rewriter.getI64ArrayAttr(1));
    lastCons = llvm_insertvalue(consTy, lastCons, elementValues[--currentIndex],
                                rewriter.getI64ArrayAttr(0));
    // Create all cells from tail to head
    Value prev = lastCons;
    for (auto i = cellsRequired; i > 1; --i) {
      Value curr = llvm_undef(consTy);
      auto prevBoxed =
          make_list(rewriter, context, typeConverter, targetInfo, prev);
      curr = llvm_insertvalue(consTy, curr, prevBoxed,
                              rewriter.getI64ArrayAttr(1));
      curr = llvm_insertvalue(consTy, curr, elementValues[--currentIndex],
                              rewriter.getI64ArrayAttr(0));
      prev = curr;
    }

    auto head = make_list(rewriter, context, typeConverter, targetInfo, prev);

    rewriter.replaceOp(op, head);
    return matchSuccess();
  }
};

}  // namespace

static void populateEIRToStandardConversionPatterns(
    OwningRewritePatternList &patterns, mlir::MLIRContext *context,
    LLVMTypeConverter &converter, TargetInfo &targetInfo) {
  patterns.insert<ReturnOpConversion, FuncOpConversion, BranchOpConversion,
                  /*
                  IfOpConversion,
                  ConstructMapOpConversion,
                  MapInsertOpConversion,
                  MapUpdateOpConversion,
                  */
                  PrintOpConversion, ConstantFloatOpToStdConversion>(
      context, converter, targetInfo);
}

/// Populate the given list with patterns that convert from EIR to LLVM
void populateEIRToLLVMConversionPatterns(OwningRewritePatternList &patterns,
                                         MLIRContext *context,
                                         LLVMTypeConverter &converter,
                                         TargetInfo &targetInfo) {
  patterns
      .insert<CondBranchOpConversion, UnreachableOpConversion, CallOpConversion,
              YieldOpConversion, GetElementPtrOpConversion, LoadOpConversion,
              IsTypeOpConversion, CastOpConversion,
              /*
              LogicalAndOpConversion,
              LogicalOrOpConversion,
              */
              CmpEqOpConversion,
              /*
              CmpNeqOpConversion,
              CmpLtOpConversion,
              CmpLteOpConversion,
              CmpGtOpConversion,
              CmpGteOpConversion,
              CmpNerrOpConversion,
              ThrowOpConversion,
              ConsOpConversion,
              TupleOpConversion,
              */
              TraceCaptureOpConversion, TraceConstructOpConversion,
              /*
              BinaryPushOpConversion,
              */
              ConstantFloatOpConversion, ConstantIntOpConversion,
              ConstantBigIntOpConversion, ConstantAtomOpConversion,
              ConstantBinaryOpConversion, ConstantNilOpConversion,
              ConstantNoneOpConversion, ConstantTupleOpConversion,
              ConstantListOpConversion /*,
               ConstantMapOpConversion
               */
              >(context, converter, targetInfo);

  // Populate the type conversions for EIR types.
  converter.addConversion(
      [&](Type type) { return convertType(type, converter, targetInfo); });
}

namespace {

// A pass converting the EIR dialect into the Standard dialect.
class ConvertEIRToLLVMPass : public mlir::ModulePass<ConvertEIRToLLVMPass> {
 public:
  ConvertEIRToLLVMPass(TargetMachine *targetMachine_)
      : targetMachine(targetMachine_),
        mlir::ModulePass<ConvertEIRToLLVMPass>() {}

  ConvertEIRToLLVMPass(const ConvertEIRToLLVMPass &other)
      : targetMachine(other.targetMachine),
        mlir::ModulePass<ConvertEIRToLLVMPass>() {}

  void runOnModule() override {
    // Create the type converter for lowering types to Standard/LLVM IR types
    auto &context = getContext();
    LLVMTypeConverter converter(&context);

    // Initialize target-specific type information, using
    // the LLVMDialect contained in the type converter to
    // create named types
    auto targetInfo = TargetInfo(targetMachine, *converter.getDialect());

    // Populate conversion patterns
    OwningRewritePatternList patterns;
    mlir::populateStdToLLVMConversionPatterns(converter, patterns,
                                              /*useAlloca=*/true,
                                              /*emitCWrappers=*/false);
    populateEIRToStandardConversionPatterns(patterns, &context, converter,
                                            targetInfo);
    populateEIRToLLVMConversionPatterns(patterns, &context, converter,
                                        targetInfo);

    // Define the legality of the operations we're converting to
    mlir::ConversionTarget conversionTarget(context);
    conversionTarget.addLegalDialect<mlir::LLVM::LLVMDialect>();
    conversionTarget.addDynamicallyLegalOp<mlir::FuncOp>([&](mlir::FuncOp op) {
      return converter.isSignatureLegal(op.getType());
    });
    conversionTarget.addLegalOp<ModuleOp, ModuleTerminatorOp>();

    mlir::ModuleOp moduleOp = getModule();
    if (failed(applyFullConversion(moduleOp, conversionTarget, patterns,
                                   &converter))) {
      moduleOp.emitError() << "conversion to LLVM IR dialect failed";
      return signalPassFailure();
    }
  }

 private:
  TargetMachine *targetMachine;
};

}  // namespace

std::unique_ptr<mlir::OpPassBase<mlir::ModuleOp>> createConvertEIRToLLVMPass(
    TargetMachine *targetMachine) {
  return std::make_unique<ConvertEIRToLLVMPass>(targetMachine);
}

}  // namespace eir
}  // namespace lumen
