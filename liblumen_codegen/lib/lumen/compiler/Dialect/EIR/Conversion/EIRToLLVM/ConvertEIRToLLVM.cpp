#include "lumen/compiler/Dialect/EIR/Conversion/EIRToLLVM/ConvertEIRToLLVM.h"

#include "lumen/compiler/Dialect/EIR/Conversion/EIRToLLVM/ComparisonOpConversions.h"
#include "lumen/compiler/Dialect/EIR/Conversion/EIRToLLVM/ConversionSupport.h"
#include "lumen/compiler/Dialect/EIR/Conversion/EIRToLLVM/MathOpConversions.h"

namespace lumen {
namespace eir {

const uint64_t MAX_REDUCTION_COUNT = 20;

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

  if (auto tupleTy = type.dyn_cast_or_null<eir::TupleType>()) {
    if (tupleTy.hasStaticShape()) {
      auto arity = tupleTy.getArity();
      SmallVector<LLVMType, 2> elementTypes;
      elementTypes.reserve(arity);
      for (unsigned i = 0; i < arity; i++) {
        auto elemTy =
            converter.convertType(tupleTy.getElementType(i)).cast<LLVMType>();
        assert(elemTy && "expected convertible element type!");
        elementTypes.push_back(elemTy);
      }
      return targetInfo.makeTupleType(converter.getDialect(), elementTypes);
    } else {
      return termTy;
    }
  }

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

FlatSymbolRefAttr createOrInsertFunction(PatternRewriter &rewriter,
                                         ModuleOp mod, LLVMDialect *dialect,
                                         TargetInfo &targetInfo,
                                         StringRef symbol, LLVMType resultType,
                                         ArrayRef<LLVMType> argTypes) {
  auto *context = mod.getContext();

  if (mod.lookupSymbol<mlir::FuncOp>(symbol))
    return SymbolRefAttr::get(symbol, context);

  if (mod.lookupSymbol<FuncOp>(symbol))
    return SymbolRefAttr::get(symbol, context);

  if (mod.lookupSymbol<LLVM::LLVMFuncOp>(symbol))
    return SymbolRefAttr::get(symbol, context);

  // Create a function declaration for the symbol
  LLVMType fnTy;
  if (resultType) {
    fnTy = LLVMType::getFunctionTy(resultType, argTypes, /*isVarArg=*/false);
  } else {
    auto voidTy = LLVMType::getVoidTy(dialect);
    fnTy = LLVMType::getFunctionTy(voidTy, argTypes, /*isVarArg=*/false);
  }

  // Insert the function into the body of the parent module.
  PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(mod.getBody());
  rewriter.create<LLVM::LLVMFuncOp>(mod.getLoc(), symbol, fnTy);
  return SymbolRefAttr::get(symbol, context);
}

static LLVM::GlobalOp createOrInsertGlobal(PatternRewriter &rewriter,
                                           ModuleOp mod, LLVMDialect *dialect,
                                           TargetInfo &targetInfo,
                                           StringRef name, LLVMType valueTy,
                                           LLVM::Linkage linkage,
                                           LLVM::ThreadLocalMode tlsMode) {
  auto *context = mod.getContext();

  if (auto global = mod.lookupSymbol<LLVM::GlobalOp>(name)) return global;

  // Insert the function into the body of the parent module.
  PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(mod.getBody());
  auto global = rewriter.create<LLVM::GlobalOp>(mod.getLoc(), valueTy,
                                                /*isConstant=*/false, linkage,
                                                tlsMode, name, Attribute());
  return global;
}

static Value getOrCreateGlobalString(Location loc, OpBuilder &builder,
                                     StringRef name, StringRef value,
                                     ModuleOp mod, LLVMDialect *dialect) {
  assert(!name.empty() && "cannot create unnamed global string!");

  auto extendedName = name.str();
  extendedName.append({'_', 'g'});

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
    globalConst = builder.create<LLVM::GlobalOp>(
        loc, type, /*isConstant=*/true, LLVM::Linkage::Internal,
        StringRef(extendedName), builder.getStringAttr(value));
    auto ptrType = LLVMType::getInt8PtrTy(dialect);
    global = builder.create<LLVM::GlobalOp>(loc, ptrType, /*isConstant=*/false,
                                            LLVM::Linkage::Internal, name,
                                            Attribute());
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

static Value do_make_list(OpBuilder &builder, edsc::ScopedContext &context,
                          LLVMTypeConverter &converter, TargetInfo &targetInfo,
                          Value cons) {
  auto headerTy = targetInfo.getUsizeType();
  Value consPtrInt = llvm_ptrtoint(headerTy, cons);
  auto listTag = targetInfo.listTag();
  auto tagAttr = builder.getIntegerAttr(
      builder.getIntegerType(targetInfo.pointerSizeInBits), listTag);
  Value listTagConst = llvm_constant(headerTy, tagAttr);
  return llvm_or(consPtrInt, listTagConst);
}

static Value do_box(OpBuilder &builder, edsc::ScopedContext &context,
                    LLVMTypeConverter &converter, TargetInfo &targetInfo,
                    Value val) {
  auto boxTag = targetInfo.boxTag();
  auto intNTy = builder.getIntegerType(targetInfo.pointerSizeInBits);
  auto termTy = targetInfo.getUsizeType();
  Value valInt = llvm_ptrtoint(termTy, val);
  // No boxing required, pointers are pointers
  if (boxTag == 0) {
    return valInt;
  }
  auto tagAttr = builder.getIntegerAttr(intNTy, boxTag);
  Value boxTagConst = llvm_constant(termTy, tagAttr);
  return llvm_or(valInt, boxTagConst);
}

static Value do_box_literal(OpBuilder &builder, edsc::ScopedContext &context,
                            LLVMTypeConverter &converter,
                            TargetInfo &targetInfo, Value val) {
  auto literalTag = targetInfo.literalTag();
  auto intNTy = builder.getIntegerType(targetInfo.pointerSizeInBits);
  auto termTy = targetInfo.getUsizeType();
  Value valInt = llvm_ptrtoint(termTy, val);
  auto tagAttr = builder.getIntegerAttr(intNTy, literalTag);
  Value literalTagConst = llvm_constant(termTy, tagAttr);
  return llvm_or(valInt, literalTagConst);
}

static Value do_unbox(OpBuilder &builder, edsc::ScopedContext &context,
                      LLVMTypeConverter &converter, TargetInfo &targetInfo,
                      LLVMType innerTy, Value box) {
  auto intNTy = builder.getIntegerType(targetInfo.pointerSizeInBits);
  auto termTy = targetInfo.getUsizeType();
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

static Value do_unbox_list(OpBuilder &builder, edsc::ScopedContext &context,
                           LLVMTypeConverter &converter, TargetInfo &targetInfo,
                           LLVMType innerTy, Value box) {
  auto intNTy = builder.getIntegerType(targetInfo.pointerSizeInBits);
  auto termTy = targetInfo.getUsizeType();
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

static Value do_mask_immediate(OpBuilder &builder, edsc::ScopedContext &context,
                               TargetInfo &targetInfo, Value val) {
  auto maskInfo = targetInfo.immediateMask();

  auto termTy = targetInfo.getUsizeType();
  auto intTy = builder.getIntegerType(targetInfo.pointerSizeInBits);
  auto maskAttr = builder.getIntegerAttr(intTy, maskInfo.mask);
  auto tagAttr = builder.getIntegerAttr(intTy, 4ull << 47);
  Value tagConst = llvm_constant(termTy, tagAttr);
  Value tagged = llvm_or(val, tagConst);
  return tagged;
  /*
  if (maskInfo.requiresShift()) {
    auto shiftAttr = builder.getIntegerAttr(intTy, maskInfo.shift);
    Value shiftConst = llvm_constant(termTy, shiftAttr);
    Value shifted = llvm_shl(val, shiftConst);
    Value maskConst = llvm_constant(termTy, maskAttr);
    return llvm_or(shifted, maskConst);
  } else {
    Value maskConst = llvm_constant(termTy, maskAttr);
    return llvm_or(val, maskConst);
  }
  */
}

Value do_unmask_immediate(OpBuilder &builder, edsc::ScopedContext &context,
                          TargetInfo &targetInfo, Value val) {
  auto maskInfo = targetInfo.immediateMask();

  auto termTy = targetInfo.getUsizeType();
  auto i1Ty = targetInfo.getI1Type();
  auto intTy = builder.getIntegerType(targetInfo.pointerSizeInBits);
  auto maskAttr = builder.getIntegerAttr(intTy, maskInfo.mask);
  auto shiftAttr = builder.getIntegerAttr(intTy, maskInfo.shift);
  Value maskConst = llvm_constant(termTy, maskAttr);
  Value masked = llvm_and(val, maskConst);
  if (maskInfo.requiresShift()) {
    Value shiftConst = llvm_constant(termTy, shiftAttr);
    return llvm_shr(masked, shiftConst);
  } else {
    return masked;
  }
}

/// Conversion Patterns

struct TraceConstructOpConversion : public EIROpConversion<TraceConstructOp> {
  using EIROpConversion::EIROpConversion;

  PatternMatchResult matchAndRewrite(
      TraceConstructOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    ModuleOp parentModule = op.getParentOfType<ModuleOp>();
    auto termTy = getUsizeType();
    auto callee = getOrInsertFunction(
        rewriter, parentModule, "__lumen_builtin_trace_construct", termTy);

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
    ModuleOp parentModule = op.getParentOfType<ModuleOp>();
    auto termTy = getUsizeType();
    auto callee = getOrInsertFunction(rewriter, parentModule,
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

    ModuleOp parentModule = op.getParentOfType<ModuleOp>();
    auto termTy = getUsizeType();
    auto int1Ty = getI1Type();
    auto int32Ty = getI32Type();

    auto matchType = op.getMatchType().cast<OpaqueTermType>();
    // Boxed types and immediate types are dispatched differently
    if (matchType.isBox()) {
      auto boxType = matchType.cast<BoxType>();
      auto boxedType = boxType.getBoxedType();

      // Lists have a unique pointer tag, so we can avoid the function call
      if (boxedType.isa<ConsType>()) {
        Value listTag = llvm_constant(
            termTy, getIntegerAttr(rewriter, targetInfo.listTag()));
        Value listMask = llvm_constant(
            termTy, getIntegerAttr(rewriter, targetInfo.listMask()));
        Value masked = llvm_and(adaptor.value(), listMask);
        rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(op, LLVM::ICmpPredicate::eq,
                                                  listTag, masked);
        return matchSuccess();
      }

      // For tuples with static shape, we use a specialized builtin
      if (auto tupleType = boxedType.dyn_cast_or_null<eir::TupleType>()) {
        if (tupleType.hasStaticShape()) {
          Value arity = llvm_constant(
              termTy, getIntegerAttr(rewriter, tupleType.getArity()));
          ArrayRef<LLVMType> argTypes({termTy, termTy});
          auto callee =
              getOrInsertFunction(rewriter, parentModule,
                                  "__lumen_builtin_is_tuple", int1Ty, argTypes);
          auto isType = rewriter.create<mlir::CallOp>(
              op.getLoc(), callee, int1Ty,
              ArrayRef<Value>{arity, adaptor.value()});
          rewriter.replaceOp(op, isType.getResults());
          return matchSuccess();
        }
      }

      // For all other boxed types, the check is performed via builtin
      auto matchKind = boxedType.getForeignKind();
      Value matchConst =
          llvm_constant(int32Ty, getI32Attr(rewriter, matchKind));
      auto callee = getOrInsertFunction(rewriter, parentModule,
                                        "__lumen_builtin_is_boxed_type", int1Ty,
                                        {int32Ty, termTy});
      Value input = adaptor.value();
      auto isType = rewriter.create<mlir::CallOp>(
          op.getLoc(), callee, int1Ty, ArrayRef<Value>{matchConst, input});
      rewriter.replaceOp(op, isType.getResults());
      return matchSuccess();
    }

    // For immediates, the check is performed via builtin
    //
    // TODO: With some additional foundation-laying, we could lower
    // these checks to precise bit masking/shift operations, rather
    // than a function call
    auto matchKind = matchType.getForeignKind();
    Value matchConst = llvm_constant(int32Ty, getI32Attr(rewriter, matchKind));
    auto callee =
        getOrInsertFunction(rewriter, parentModule, "__lumen_builtin_is_type",
                            int1Ty, {int32Ty, termTy});
    auto isType = rewriter.create<mlir::CallOp>(
        op.getLoc(), callee, int1Ty,
        ArrayRef<Value>{matchConst, adaptor.value()});
    rewriter.replaceOp(op, isType.getResults());

    return matchSuccess();
  }
};

struct YieldOpConversion : public EIROpConversion<YieldOp> {
  using EIROpConversion::EIROpConversion;

  PatternMatchResult matchAndRewrite(
      YieldOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    ModuleOp parentModule = op.getParentOfType<ModuleOp>();
    auto termTy = getUsizeType();
    auto callee = getOrInsertFunction(rewriter, parentModule,
                                      "__lumen_builtin_yield", LLVMType());

    rewriter.replaceOpWithNewOp<mlir::CallOp>(op, callee, ArrayRef<Type>{});
    return matchSuccess();
  }
};

struct YieldCheckOpConversion : public EIROpConversion<YieldCheckOp> {
  using EIROpConversion::EIROpConversion;

  PatternMatchResult matchAndRewrite(
      YieldCheckOp op, ArrayRef<Value> properOperands,
      ArrayRef<Block *> destinations, ArrayRef<ArrayRef<Value>> operands,
      ConversionPatternRewriter &rewriter) const override {
    edsc::ScopedContext context(rewriter, op.getLoc());
    YieldCheckOpOperandAdaptor adaptor(properOperands);
    ModuleOp parentModule = op.getParentOfType<ModuleOp>();

    auto termTy = getUsizeType();
    auto reductionCountGlobal = getOrInsertGlobal(
        rewriter, parentModule, "CURRENT_REDUCTION_COUNT", termTy,
        LLVM::Linkage::External, LLVM::ThreadLocalMode::LocalExec);

    // Load the current reduction count
    Value reductionCount = llvm_load(reductionCountGlobal);
    // If greater than or equal to the max reduction count, yield
    Value maxReductions =
        llvm_constant(termTy, getIntegerAttr(rewriter, MAX_REDUCTION_COUNT));
    Value shouldYield =
        llvm_icmp(LLVM::ICmpPredicate::uge, reductionCount, maxReductions);

    auto trueDest = op.getTrueDest();
    auto falseDest = op.getFalseDest();
    auto trueArgs = ValueRange(op.getTrueOperands());
    auto falseArgs = ValueRange(op.getFalseOperands());

    auto attrs = op.getAttrs();
    ArrayRef<Block *> dests({trueDest, falseDest});
    ArrayRef<ValueRange> destsArgs({trueArgs, falseArgs});
    rewriter.replaceOpWithNewOp<LLVM::CondBrOp>(op, shouldYield, dests,
                                                destsArgs, attrs);
    return matchSuccess();
  }
};

struct IncrementReductionsOpConversion
    : public EIROpConversion<IncrementReductionsOp> {
  using EIROpConversion::EIROpConversion;

  PatternMatchResult matchAndRewrite(
      IncrementReductionsOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    edsc::ScopedContext context(rewriter, op.getLoc());
    ModuleOp parentModule = op.getParentOfType<ModuleOp>();

    auto termTy = getUsizeType();
    auto reductionCount = getOrInsertGlobal(
        rewriter, parentModule, "CURRENT_REDUCTION_COUNT", termTy,
        LLVM::Linkage::External, LLVM::ThreadLocalMode::LocalExec);

    auto incBy = op.increment().getLimitedValue();
    Value increment = llvm_constant(termTy, getIntegerAttr(rewriter, incBy));
    llvm_atomicrmw(termTy, LLVM::AtomicBinOp::add, reductionCount, increment,
                   LLVM::AtomicOrdering::unordered);
    rewriter.eraseOp(op);
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
      auto termTy = getUsizeType();
      auto i1Ty = getI1Type();
      Value maskConst =
          llvm_constant(termTy, getIntegerAttr(rewriter, maskInfo.mask));
      Value maskedCond = llvm_and(cond, maskConst);
      if (maskInfo.requiresShift()) {
        Value shiftConst =
            llvm_constant(termTy, getIntegerAttr(rewriter, maskInfo.shift));
        Value shiftedCond = llvm_shr(maskedCond, shiftConst);
        finalCond = llvm_trunc(i1Ty, shiftedCond);
      } else {
        finalCond = llvm_trunc(i1Ty, maskedCond);
      }
    }

    auto attrs = op.getAttrs();
    ArrayRef<Block *> dests({trueDest, falseDest});
    ArrayRef<ValueRange> destsArgs({trueArgs, falseArgs});
    rewriter.replaceOpWithNewOp<LLVM::CondBrOp>(op, finalCond, dests, destsArgs,
                                                attrs);
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
    edsc::ScopedContext context(rewriter, op.getLoc());
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

    // Insert yield check if this function is not extern
    auto *region = newFunc.getCallableRegion();
    if (region) {
      auto ip = rewriter.saveInsertionPoint();
      auto entry = &region->front();
      auto entryOp = &entry->front();
      // Splits `entry` returning a block containing all the previous
      // contents of entry since we're splitting on the first op. This
      // block is `doYield` because split it a second time to give us
      // the success block. Since the values in `entry` dominate both
      // splits, we don't have to pass arguments
      Block *doYield = entry->splitBlock(&entry->front());
      Block *dontYield = doYield->splitBlock(&doYield->front());
      // Insert yield check in original entry block
      rewriter.setInsertionPointToEnd(entry);
      rewriter.create<YieldCheckOp>(op.getLoc(), doYield, ValueRange{},
                                    dontYield, ValueRange{});
      // Then insert the actual yield point in the yield block
      rewriter.setInsertionPointToEnd(doYield);
      rewriter.create<YieldOp>(op.getLoc());
      // Then post-yield, branch to the real entry block
      rewriter.create<BranchOp>(op.getLoc(), dontYield);
      // Reset the builder to where it was originally
      rewriter.restoreInsertionPoint(ip);
    }

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

    ModuleOp parentModule = op.getParentOfType<ModuleOp>();

    auto termTy = getUsizeType();
    auto printfRef = getOrInsertFunction(
        rewriter, parentModule, "__lumen_builtin_printf", termTy, {termTy});

    rewriter.replaceOpWithNewOp<mlir::CallOp>(op, printfRef, termTy, operands);
    return matchSuccess();
  }
};

struct ThrowOpConversion : public EIROpConversion<ThrowOp> {
  using EIROpConversion::EIROpConversion;

  PatternMatchResult matchAndRewrite(
      ThrowOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    ThrowOpOperandAdaptor adaptor(operands);

    ModuleOp parentModule = op.getParentOfType<ModuleOp>();

    Value reason = adaptor.reason();
    auto termTy = getUsizeType();
    auto voidTy = LLVMType::getVoidTy(dialect);
    auto callee = getOrInsertFunction(
        rewriter, parentModule, "__lumen_builtin_throw", voidTy, {termTy});

    rewriter.create<mlir::CallOp>(op.getLoc(), callee, ArrayRef<Type>{},
                                  operands);
    rewriter.replaceOpWithNewOp<LLVM::UnreachableOp>(op, ArrayRef<Value>{});
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

    // Always increment reduction count when performing a call
    rewriter.create<IncrementReductionsOp>(op.getLoc());

    auto calleeName = op.getCallee();
    auto callee = getOrInsertFunction(rewriter, parentModule, calleeName,
                                      resultType, argTypes);

    auto callOp = rewriter.create<mlir::CallOp>(
        op.getLoc(), callee, resultTypes, adaptor.operands());
    callOp.setAttr("tail", rewriter.getUnitAttr());
    rewriter.replaceOp(op, callOp.getResults());
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
    auto ptrTy = resultTy.getPointerTo();
    auto int32Ty = getI32Type();

    Value cns0 = llvm_constant(int32Ty, getI32Attr(rewriter, 0));
    Value index = llvm_constant(int32Ty, getI32Attr(rewriter, op.getIndex()));
    ArrayRef<Value> indices({cns0, index});
    Value gep = llvm_gep(ptrTy, base, indices);

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
          ptr = unbox_list(rewriter, context, outTy, in);
        } else {
          ptr = unbox(rewriter, context, outTy, in);
        }
      } else {
        ptr = unbox(rewriter, context, outTy, in);
      }
      rewriter.replaceOp(op, ptr);
      return matchSuccess();
    }

    // Convert immediate to opaque term
    if (auto it = op.input().getType().dyn_cast_or_null<OpaqueTermType>()) {
      // Special handling needed for booleans
      if (inTy.isIntegerTy(1)) {
        Value term = llvm_zext(termTy, in);
        Value imm = mask_immediate(rewriter, context, term);
        rewriter.replaceOp(op, imm);
        return matchSuccess();
      }
      if (origOutTy.isa<TermType>()) {
        Value term = llvm_bitcast(termTy, in);
        Value imm = mask_immediate(rewriter, context, term);
        rewriter.replaceOp(op, imm);
        return matchSuccess();
      }
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
    auto headerTy = getUsizeType();
    APInt headerVal = targetInfo.encodeHeader(TypeKind::Float, 2);
    auto descTy = targetInfo.getFloatType();
    Value header = llvm_constant(
        headerTy, getIntegerAttr(rewriter, headerVal.getLimitedValue()));
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
    auto termTy = getUsizeType();
    auto i = (uint64_t)attr.getValue().getLimitedValue();
    auto taggedInt = targetInfo.encodeImmediate(TypeKind::Fixnum, i);
    auto val = llvm_constant(termTy, getIntegerAttr(rewriter, taggedInt));

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
    auto termTy = getUsizeType();
    auto taggedAtom = targetInfo.encodeImmediate(TypeKind::Atom, id);
    Value val = llvm_constant(termTy, getIntegerAttr(rewriter, taggedAtom));

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
    auto ptrTy = ty.getPointerTo();
    auto termTy = getUsizeType();

    ModuleOp parentModule = op.getParentOfType<ModuleOp>();

    auto boxTag = targetInfo.boxTag();
    auto literalTag = targetInfo.literalTag();
    auto boxedLiteralTag = boxTag | literalTag;
    Value literalTagConst =
        llvm_constant(termTy, getIntegerAttr(rewriter, boxedLiteralTag));

    // We use the SHA-1 hash of the value as the name of the global,
    // this provides a nice way to de-duplicate constant strings while
    // not requiring any global state
    auto name = binAttr.getHash();
    Value valPtr =
        getOrCreateGlobalString(context.getLocation(), context.getBuilder(),
                                name, bytes, parentModule, dialect);
    Value valPtrLoad = llvm_load(valPtr);
    Value header = llvm_constant(termTy, getIntegerAttr(rewriter, headerRaw));
    Value flags = llvm_constant(termTy, getIntegerAttr(rewriter, flagsRaw));

    int64_t size = (targetInfo.pointerSizeInBits / 8) * 3;
    Value allocBytes = llvm_constant(termTy, rewriter.getI64IntegerAttr(size));
    Value descAlloc = processAlloc(rewriter, context, parentModule, op.getLoc(),
                                   ty, allocBytes);

    Value desc = llvm_undef(ty);
    desc = llvm_insertvalue(ty, desc, header, rewriter.getI64ArrayAttr(0));
    desc = llvm_insertvalue(ty, desc, flags, rewriter.getI64ArrayAttr(1));
    desc = llvm_insertvalue(ty, desc, valPtrLoad, rewriter.getI64ArrayAttr(2));
    llvm_store(desc, descAlloc);

    // Box the allocated tuple
    auto boxed = make_literal(rewriter, context, descAlloc);

    rewriter.replaceOp(op, boxed);
    return matchSuccess();
  }
};

struct ConstantNilOpConversion : public EIROpConversion<ConstantNilOp> {
  using EIROpConversion::EIROpConversion;

  PatternMatchResult matchAndRewrite(
      ConstantNilOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    edsc::ScopedContext context(rewriter, op.getLoc());

    auto val = llvm_constant(
        getUsizeType(), getIntegerAttr(rewriter, targetInfo.getNilValue()));

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

    auto val = llvm_constant(
        getUsizeType(), getIntegerAttr(rewriter, targetInfo.getNoneValue()));

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
  auto termTy = targetInfo.getTermType();
  auto constIntTy = rewriter.getIntegerType(targetInfo.pointerSizeInBits);
  for (auto elementAttr : elements) {
    auto elementType = elementAttr.getType();
    if (auto atomAttr = elementAttr.dyn_cast_or_null<AtomAttr>()) {
      auto id = (uint64_t)atomAttr.getValue().getLimitedValue();
      auto tagged = targetInfo.encodeImmediate(TypeKind::Atom, id);
      auto val =
          llvm_constant(termTy, rewriter.getIntegerAttr(constIntTy, tagged));
      elementTypes.push_back(termTy);
      elementValues.push_back(val);
      continue;
    }
    if (auto boolAttr = elementAttr.dyn_cast_or_null<BoolAttr>()) {
      auto b = boolAttr.getValue();
      uint64_t id = b ? 1 : 0;
      auto tagged = targetInfo.encodeImmediate(TypeKind::Atom, id);
      auto val =
          llvm_constant(termTy, rewriter.getIntegerAttr(constIntTy, tagged));
      elementTypes.push_back(termTy);
      elementValues.push_back(val);
      continue;
    }
    if (auto intAttr = elementAttr.dyn_cast_or_null<IntegerAttr>()) {
      auto i = intAttr.getValue();
      assert(i.getBitWidth() <= targetInfo.pointerSizeInBits &&
             "support for bigint in constant aggregates not yet implemented");
      auto tagged =
          targetInfo.encodeImmediate(TypeKind::Fixnum, i.getLimitedValue());
      auto val =
          llvm_constant(termTy, rewriter.getIntegerAttr(constIntTy, tagged));
      elementTypes.push_back(termTy);
      elementValues.push_back(val);
      continue;
    }
    if (auto floatAttr = elementAttr.dyn_cast_or_null<FloatAttr>()) {
      auto f = floatAttr.getValue().bitcastToAPInt();
      assert(!targetInfo.requiresPackedFloats() &&
             "support for packed floats in constant aggregates is not yet "
             "implemented");
      auto val = llvm_constant(
          termTy, rewriter.getIntegerAttr(constIntTy, f.getLimitedValue()));
      elementTypes.push_back(termTy);
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

    auto termTy = getUsizeType();
    auto attr = op.getValue().cast<SeqAttr>();
    auto elements = attr.getValue();
    auto numElements = elements.size();

    // Construct tuple header
    auto headerRaw = targetInfo.encodeHeader(TypeKind::Tuple, numElements);
    Value header = llvm_constant(
        termTy, getIntegerAttr(rewriter, headerRaw.getLimitedValue()));

    SmallVector<LLVMType, 2> elementTypes;
    elementTypes.reserve(numElements);
    SmallVector<Value, 2> elementValues;
    elementValues.reserve(numElements);

    auto lowered = lowerElementValues(context, rewriter, targetInfo, elements,
                                      elementValues, elementTypes);
    assert(lowered && "unsupported element type in tuple constant");

    auto ty = getTupleType(elementTypes);
    auto ptrTy = ty.getPointerTo();

    Value allocN = llvm_constant(termTy, rewriter.getI64IntegerAttr(1));
    Value tupleAlloc =
        llvm_alloca(ptrTy, allocN, rewriter.getI64IntegerAttr(8));

    Value tuple = llvm_undef(ty);
    tuple = llvm_insertvalue(ty, tuple, header, rewriter.getI64ArrayAttr(0));
    for (auto i = 0; i < numElements; i++) {
      auto val = elementValues[i];
      tuple = llvm_insertvalue(ty, tuple, val, rewriter.getI64ArrayAttr(i + 1));
    }
    llvm_store(tuple, tupleAlloc);

    auto boxTag = targetInfo.boxTag();
    auto literalTag = targetInfo.literalTag();
    auto boxedLiteralTag = boxTag | literalTag;
    Value literalTagConst =
        llvm_constant(termTy, getIntegerAttr(rewriter, boxedLiteralTag));

    Value tuplePtrInt = llvm_ptrtoint(termTy, tupleAlloc);
    Value boxedTuplePtr = llvm_or(tuplePtrInt, literalTagConst);
    Value boxed = llvm_bitcast(termTy, boxedTuplePtr);

    rewriter.replaceOp(op, boxed);
    return matchSuccess();
  }
};

struct TupleOpConversion : public EIROpConversion<TupleOp> {
  using EIROpConversion::EIROpConversion;

  PatternMatchResult matchAndRewrite(
      TupleOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    edsc::ScopedContext context(rewriter, op.getLoc());
    TupleOpOperandAdaptor adaptor(operands);

    auto termTy = getUsizeType();
    auto elements = adaptor.elements();
    auto numElements = elements.size();

    // Construct tuple header
    auto headerRaw = targetInfo.encodeHeader(TypeKind::Tuple, numElements);
    Value header = llvm_constant(
        termTy, getIntegerAttr(rewriter, headerRaw.getLimitedValue()));

    // Construct tuple type
    SmallVector<LLVMType, 2> elementTypes;
    elementTypes.reserve(numElements);
    for (auto val : elements) {
      auto valTy = val.getType().cast<LLVMType>();
      elementTypes.push_back(valTy);
    }
    auto ty = getTupleType(elementTypes);
    auto ptrTy = ty.getPointerTo();

    // Allocate tuple on the stack and insert all of the elements
    int64_t size = (targetInfo.pointerSizeInBits / 8) * (numElements + 1);
    Value allocBytes = llvm_constant(termTy, rewriter.getI64IntegerAttr(size));
    // Value tupleAlloc =
    //    llvm_alloca(ptrTy, allocN, rewriter.getI64IntegerAttr(8));
    ModuleOp parentModule = op.getParentOfType<ModuleOp>();
    Value tupleAlloc = processAlloc(rewriter, context, parentModule,
                                    op.getLoc(), ty, allocBytes);
    Value tuple = llvm_undef(ty);
    tuple = llvm_insertvalue(ty, tuple, header, rewriter.getI64ArrayAttr(0));
    for (auto i = 0; i < numElements; i++) {
      auto val = elements[i];
      tuple = llvm_insertvalue(ty, tuple, val, rewriter.getI64ArrayAttr(i + 1));
    }
    llvm_store(tuple, tupleAlloc);

    // Box the allocated tuple
    auto boxed = make_box(rewriter, context, tupleAlloc);

    rewriter.replaceOp(op, boxed);
    return matchSuccess();
  }
};

struct ConsOpConversion : public EIROpConversion<ConsOp> {
  using EIROpConversion::EIROpConversion;

  PatternMatchResult matchAndRewrite(
      ConsOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    edsc::ScopedContext context(rewriter, op.getLoc());
    ConsOpOperandAdaptor adaptor(operands);

    auto termTy = getUsizeType();
    auto consTy = targetInfo.getConsType();
    auto ptrTy = consTy.getPointerTo();

    auto head = adaptor.head();
    auto tail = adaptor.tail();

    // Allocate tuple on the stack and insert all of the elements
    Value allocN = llvm_constant(termTy, rewriter.getI64IntegerAttr(1));
    Value consAlloc = llvm_alloca(ptrTy, allocN, rewriter.getI64IntegerAttr(8));
    Value cons = llvm_undef(consTy);
    cons = llvm_insertvalue(consTy, cons, head, rewriter.getI64ArrayAttr(0));
    cons = llvm_insertvalue(consTy, cons, tail, rewriter.getI64ArrayAttr(1));
    llvm_store(cons, consAlloc);

    // Box the allocated cons
    auto boxed = make_list(rewriter, context, consAlloc);

    rewriter.replaceOp(op, boxed);
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

    auto termTy = getUsizeType();
    // Lower to nil if empty list
    if (numElements == 0) {
      auto val = llvm_constant(
          termTy, getIntegerAttr(rewriter, targetInfo.getNilValue()));
      rewriter.replaceOp(op, {val});
      return matchSuccess();
    }

    SmallVector<LLVMType, 2> elementTypes;
    elementTypes.reserve(numElements);
    SmallVector<Value, 2> elementValues;
    elementValues.reserve(numElements);

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
    auto nilVal = llvm_constant(
        termTy, getIntegerAttr(rewriter, targetInfo.getNilValue()));
    lastCons =
        llvm_insertvalue(consTy, lastCons, nilVal, rewriter.getI64ArrayAttr(1));
    lastCons = llvm_insertvalue(consTy, lastCons, elementValues[--currentIndex],
                                rewriter.getI64ArrayAttr(0));
    // Create all cells from tail to head
    Value prev = lastCons;
    for (auto i = cellsRequired; i > 1; --i) {
      Value curr = llvm_undef(consTy);
      auto prevBoxed = make_list(rewriter, context, prev);
      curr = llvm_insertvalue(consTy, curr, prevBoxed,
                              rewriter.getI64ArrayAttr(1));
      curr = llvm_insertvalue(consTy, curr, elementValues[--currentIndex],
                              rewriter.getI64ArrayAttr(0));
      prev = curr;
    }

    auto head = make_list(rewriter, context, prev);

    rewriter.replaceOp(op, head);
    return matchSuccess();
  }
};

static void populateEIRToStandardConversionPatterns(
    OwningRewritePatternList &patterns, mlir::MLIRContext *context,
    LLVMTypeConverter &converter, TargetInfo &targetInfo) {
  patterns.insert<ReturnOpConversion, FuncOpConversion, BranchOpConversion,
                  /*
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
              YieldOpConversion, YieldCheckOpConversion,
              IncrementReductionsOpConversion, GetElementPtrOpConversion,
              LoadOpConversion, IsTypeOpConversion, CastOpConversion,
              /*
              LogicalAndOpConversion,
              LogicalOrOpConversion,
              */
              ThrowOpConversion, TraceCaptureOpConversion,
              TraceConstructOpConversion, ConsOpConversion, TupleOpConversion,
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
  populateComparisonOpConversionPatterns(patterns, context, converter,
                                         targetInfo);
  populateMathOpConversionPatterns(patterns, context, converter, targetInfo);

  // Populate the type conversions for EIR types.
  converter.addConversion(
      [&](Type type) { return convertType(type, converter, targetInfo); });
}

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

std::unique_ptr<mlir::OpPassBase<mlir::ModuleOp>> createConvertEIRToLLVMPass(
    TargetMachine *targetMachine) {
  return std::make_unique<ConvertEIRToLLVMPass>(targetMachine);
}

}  // namespace eir
}  // namespace lumen
