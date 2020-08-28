#include "lumen/EIR/Conversion/ConversionSupport.h"

namespace lumen {
namespace eir {

static bool isa_eir_type(Type t) { return isa<eirDialect>(t.getDialect()); }

static bool isa_std_type(Type t) {
  return isa<mlir::StandardOpsDialect>(t.getDialect());
}

Optional<Type> convertType(Type type, EirTypeConverter &converter,
                           TargetInfo &targetInfo) {
  if (!isa_eir_type(type)) return Optional<Type>();

  MLIRContext *context = type.getContext();
  auto termTy = targetInfo.getUsizeType();

  if (auto ptrTy = type.dyn_cast_or_null<PtrType>()) {
    auto innerTy = converter.convertType(ptrTy.getInnerType()).cast<LLVMType>();
    return innerTy.getPointerTo();
  }

  if (auto refTy = type.dyn_cast_or_null<RefType>()) {
    auto innerTy = converter.convertType(refTy.getInnerType()).cast<LLVMType>();
    return innerTy.getPointerTo();
  }

  if (auto boxTy = type.dyn_cast_or_null<BoxType>()) {
    auto boxedTy = converter.convertType(boxTy.getBoxedType()).cast<LLVMType>();
    return boxedTy.getPointerTo();
  }

  if (auto recvRef = type.dyn_cast_or_null<ReceiveRefType>()) {
    return targetInfo.getI8Type().getPointerTo();
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
      return targetInfo.makeTupleType(elementTypes);
    } else {
      return termTy;
    }
  }

  if (type.isa<eir::ClosureType>()) {
    return targetInfo.makeClosureType(1);
  }

  llvm::outs() << "\ntype: ";
  type.dump();
  llvm::outs() << "\n";
  assert(false && "unimplemented type conversion");

  return llvm::None;
}

Operation *OpConversionContext::getOrInsertFunction(
    ModuleOp mod, StringRef symbol, LLVMType resultTy,
    ArrayRef<LLVMType> argTypes, ArrayRef<NamedAttribute> attrs) const {
  Operation *funcOp = SymbolTable::lookupNearestSymbolFrom(mod, symbol);
  if (funcOp) return funcOp;

  // Create a function declaration for the symbol
  LLVMType fnTy;
  if (resultTy) {
    fnTy = LLVMType::getFunctionTy(resultTy, argTypes, /*isVarArg=*/false);
  } else {
    auto voidTy = LLVMType::getVoidTy(context);
    fnTy = LLVMType::getFunctionTy(voidTy, argTypes, /*isVarArg=*/false);
  }

  PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(mod.getBody());
  auto op = rewriter.create<LLVM::LLVMFuncOp>(mod.getLoc(), symbol, fnTy);
  for (auto attr : attrs) {
    op.setAttr(std::get<Identifier>(attr), std::get<Attribute>(attr));
  }
  return op;
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
    auto i8PtrTy = i8Ty.getPointerTo();
    auto i64Ty = LLVMType::getInt64Ty(context);
    auto strTy = LLVMType::getArrayTy(i8Ty, value.size());

    PatternRewriter::InsertionGuard insertGuard(rewriter);
    // Make sure we insert this global after the definition of the constant
    rewriter.setInsertionPointAfter(globalConst);
    // Insert the global definition
    global = getOrInsertGlobalConstantOp(mod, extendedName, i8PtrTy);
    // Initialize the global with a pointer to the first char of the constant
    // string
    auto &initRegion = global.getInitializerRegion();
    auto *initBlock = rewriter.createBlock(&initRegion);
    auto globalPtr = llvm_addressof(globalConst);
    Value zero = llvm_constant(i64Ty, getIntegerAttr(0));
    ArrayRef<Value> indices{zero, zero};
    Value address = llvm_gep(i8PtrTy, globalPtr, indices);
    rewriter.create<mlir::ReturnOp>(global.getLoc(), address);
  }

  return global;
}

Value OpConversionContext::buildMalloc(ModuleOp mod, LLVMType ty,
                                       unsigned allocTy, Value arity) const {
  auto i8PtrTy = targetInfo.getI8Type().getPointerTo();
  auto ptrTy = ty.getPointerTo();
  auto i32Ty = targetInfo.getI32Type();
  auto usizeTy = getUsizeType();
  StringRef symbolName("__lumen_builtin_malloc");
  auto callee = getOrInsertFunction(mod, symbolName, i8PtrTy, {i32Ty, usizeTy});
  auto allocTyConst = llvm_constant(i32Ty, getU32Attr(allocTy));
  auto calleeSymbol = FlatSymbolRefAttr::get(symbolName, callee->getContext());
  ArrayRef<Value> args{allocTyConst, arity};
  Operation *call = std_call(calleeSymbol, ArrayRef<Type>{i8PtrTy}, args);
  return llvm_bitcast(ptrTy, call->getResult(0));
}

Value OpConversionContext::encodeList(Value cons, bool isLiteral) const {
  auto termTy = getUsizeType();
  Value ptrInt = llvm_ptrtoint(termTy, cons);
  Value tag;
  if (isLiteral) {
    Value listTag = llvm_constant(termTy, getIntegerAttr(targetInfo.listTag()));
    Value literalTag =
        llvm_constant(termTy, getIntegerAttr(targetInfo.literalTag()));
    tag = llvm_or(listTag, literalTag);
  } else {
    tag = llvm_constant(termTy, getIntegerAttr(targetInfo.listTag()));
  }
  return llvm_or(ptrInt, tag);
}

Value OpConversionContext::encodeBox(Value val) const {
  auto rawTag = targetInfo.boxTag();
  auto termTy = getUsizeType();
  // No boxing required, pointers are pointers
  if (rawTag == 0) {
    return llvm_ptrtoint(termTy, val);
  } else {
    Value ptrInt = llvm_ptrtoint(termTy, val);
    Value tag = llvm_constant(termTy, getIntegerAttr(rawTag));
    return llvm_or(ptrInt, tag);
  }
}

Value OpConversionContext::encodeLiteral(Value val) const {
  auto rawTag = targetInfo.literalTag();
  auto termTy = getUsizeType();
  Value ptrInt = llvm_ptrtoint(termTy, val);
  Value tag = llvm_constant(termTy, getIntegerAttr(rawTag));
  return llvm_or(ptrInt, tag);
}

Value OpConversionContext::encodeImmediate(ModuleOp mod, Location loc,
                                           OpaqueTermType ty, Value val) const {
  auto termTy = getUsizeType();
  auto i32Ty = getI32Type();
  StringRef symbolName("__lumen_builtin_encode_immediate");
  auto callee = getOrInsertFunction(mod, symbolName, termTy, {i32Ty, termTy});
  auto calleeSymbol = FlatSymbolRefAttr::get(symbolName, callee->getContext());

  Value kind = llvm_constant(i32Ty, getI32Attr(ty.getTypeKind().getValue()));
  ArrayRef<Value> args{kind, val};
  Operation *call = std_call(calleeSymbol, ArrayRef<Type>{termTy}, args);
  return call->getResult(0);
}

Value OpConversionContext::decodeBox(LLVMType innerTy, Value box) const {
  auto termTy = getUsizeType();
  auto boxTy = box.getType().cast<LLVMType>();
  assert(boxTy == termTy && "expected boxed pointer type");
  auto rawTag = targetInfo.boxTag();
  // No unboxing required, pointers are pointers
  if (rawTag == 0) {
    return llvm_inttoptr(innerTy.getPointerTo(), box);
  } else {
    Value tag = llvm_constant(termTy, getIntegerAttr(rawTag));
    Value neg1 = llvm_constant(termTy, getIntegerAttr(-1));
    Value untagged = llvm_and(box, llvm_xor(tag, neg1));
    return llvm_inttoptr(innerTy.getPointerTo(), untagged);
  }
}

Value OpConversionContext::decodeList(Value box) const {
  auto termTy = targetInfo.getUsizeType();
  Value mask = llvm_constant(termTy, getIntegerAttr(targetInfo.listMask()));
  Value neg1 = llvm_constant(termTy, getIntegerAttr(-1));
  Value untagged = llvm_and(box, llvm_xor(mask, neg1));
  return llvm_inttoptr(targetInfo.getConsType().getPointerTo(), untagged);
}

Value OpConversionContext::decodeImmediate(Value val) const {
  auto termTy = getUsizeType();
  auto maskInfo = targetInfo.immediateMask();

  Value mask = llvm_constant(termTy, getIntegerAttr(maskInfo.mask));
  Value masked = llvm_and(val, mask);
  if (maskInfo.requiresShift()) {
    Value shift = llvm_constant(termTy, getIntegerAttr(maskInfo.shift));
    return llvm_shr(masked, shift);
  } else {
    return masked;
  }
}

}  // namespace eir
}  // namespace lumen
