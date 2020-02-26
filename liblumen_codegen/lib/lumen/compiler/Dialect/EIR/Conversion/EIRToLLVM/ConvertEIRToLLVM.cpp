#include "lumen/compiler/Dialect/EIR/Conversion/EIRToLLVM/ConvertEIRToLLVM.h"
#include "lumen/compiler/Dialect/EIR/IR/EIRTypes.h"
#include "lumen/compiler/Dialect/EIR/IR/EIRAttributes.h"
#include "lumen/compiler/Dialect/EIR/IR/EIROps.h"
#include "lumen/compiler/Target/TargetInfo.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/EDSC/Intrinsics.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"

#include "llvm/Target/TargetMachine.h"

using ::mlir::PatternMatchResult;
using ::mlir::ConversionPatternRewriter;
using ::mlir::LLVMTypeConverter;
using ::mlir::LLVM::LLVMType;
using ::mlir::LLVM::LLVMDialect;
using ::mlir::edsc::intrinsics::OperationBuilder;
using ::mlir::edsc::intrinsics::ValueBuilder;
using ::llvm::TargetMachine;

namespace LLVM = ::mlir::LLVM;

using llvm_add = ValueBuilder<LLVM::AddOp>;
using llvm_and = ValueBuilder<LLVM::AndOp>;
using llvm_or = ValueBuilder<LLVM::OrOp>;
using llvm_bitcast = ValueBuilder<LLVM::BitcastOp>;
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
using llvm_sub = ValueBuilder<LLVM::SubOp>;
using llvm_undef = ValueBuilder<LLVM::UndefOp>;
using llvm_urem = ValueBuilder<LLVM::URemOp>;
using llvm_alloca = ValueBuilder<LLVM::AllocaOp>;
using llvm_return = OperationBuilder<LLVM::ReturnOp>;

namespace lumen {
namespace eir {

static bool isa_eir_type(Type t) {
  return inbounds(t.getKind(),
                  Type::Kind::FIRST_EIR_TYPE,
                  Type::Kind::LAST_EIR_TYPE);
}


static bool isa_std_type(Type t) {
  return inbounds(t.getKind(),
                  Type::Kind::FIRST_STANDARD_TYPE,
                  Type::Kind::LAST_STANDARD_TYPE);
}

static Optional<Type> convertType(Type type, LLVMTypeConverter &converter, TargetInfo &targetInfo) {
  if (!isa_eir_type(type))
    return Optional<Type>();

  MLIRContext *context = type.getContext();
  OpaqueTermType termTy = type.cast<OpaqueTermType>();
  if (termTy.isOpaque())
    return targetInfo.getTermType();

  if (termTy.isImmediate() && !termTy.isBox())
    return targetInfo.getTermType();

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

/// Return a symbol reference to the printf function, inserting it into the
/// module if necessary.
static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter,
                                           ModuleOp mod,
                                           LLVMDialect *dialect,
                                           TargetInfo &targetInfo) {
  auto *context = mod.getContext();
  if (mod.lookupSymbol<LLVM::LLVMFuncOp>("__lumen_builtin_printf"))
    return SymbolRefAttr::get("__lumen_builtin_printf", context);

  // Create a function declaration for printf, the signature is:
  //   * `term (i8*, term)`
  // Where the return value is the atom `ok` or the none value,
  // and the term argument should be a box of list type.
  auto termTy = targetInfo.getTermType();
  //auto i8PtrTy = LLVMType::getInt8PtrTy(dialect);
  //ArrayRef<LLVMType> argTypes({i8PtrTy, termTy});
  ArrayRef<LLVMType> argTypes({termTy});
  auto fnTy = LLVMType::getFunctionTy(termTy, argTypes, /*isVarArg=*/false);

  // Insert the printf function into the body of the parent module.
  PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(mod.getBody());
  rewriter.create<LLVM::LLVMFuncOp>(mod.getLoc(), "__lumen_builtin_printf", fnTy);
  return SymbolRefAttr::get("__lumen_builtin_printf", context);
}

/// Return a value representing an access into a global string with the given
/// name, creating the string if necessary.
static Value getOrCreateGlobalString(Location loc, OpBuilder &builder,
                                     StringRef name, StringRef value,
                                     ModuleOp mod,
                                     LLVMDialect *dialect) {
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
    auto type = LLVMType::getArrayTy(LLVMType::getInt8Ty(dialect), value.size());
    globalConst = builder.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/true,
                                                 LLVM::Linkage::Internal, name,
                                                 builder.getStringAttr(value));
    auto ptrType = LLVMType::getInt8PtrTy(dialect);
    global = builder.create<LLVM::GlobalOp>(loc, ptrType, /*isConstant=*/false,
                                            LLVM::Linkage::Internal, StringRef(extendedName),
                                            Attribute());
    auto &initRegion = global.getInitializerRegion();
    auto *initBlock = builder.createBlock(&initRegion);

    // Get the pointer to the first character in the global string.
    auto globalPtr = builder.create<LLVM::AddressOfOp>(loc, globalConst);
    Value cst0 = llvm_constant(i64Ty, builder.getIntegerAttr(indexTy, 0));
    auto gepPtr = builder.create<LLVM::GEPOp>(loc, i8PtrTy, globalPtr, ArrayRef<Value>({cst0, cst0}));
    builder.create<LLVM::ReturnOp>(loc, gepPtr.getResult());
  } else {
    global = mod.lookupSymbol<LLVM::GlobalOp>(StringRef(extendedName));
  }

  return llvm_addressof(global);
}


namespace {

/// Rewrite Patterns

class TraceConstructOpConversion : public OpRewritePattern<TraceConstructOp> {
public:
  using OpRewritePattern<TraceConstructOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(TraceConstructOp op,
                                     PatternRewriter &rewriter) const override {

    auto context = rewriter.getContext();
    auto traceType = TupleType::get(context, 3);
    rewriter.replaceOpWithNewOp<mlir::CallOp>(
      op, "__lumen_builtin_construct_trace", ArrayRef<Type>{traceType});
    return this->matchSuccess();
  }
};

class YieldOpConversion : public OpRewritePattern<YieldOp> {
public:
  using OpRewritePattern<YieldOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(YieldOp op,
                                     PatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<mlir::CallOp>(
      op, "__lumen_builtin_yield", ArrayRef<Type>{});
    return this->matchSuccess();
  }
};

/// Conversion Patterns

template <typename Op>
class EIROpConversion : public mlir::OpConversionPattern<Op> {
 public:
  explicit EIROpConversion(MLIRContext *context,
                           LLVMTypeConverter &converter_,
                           TargetInfo &targetInfo_,
                           mlir::PatternBenefit benefit = 1)
      : mlir::OpConversionPattern<Op>::OpConversionPattern(context, benefit),
        typeConverter(converter_),
        targetInfo(targetInfo_) {}

 protected:
  LLVMTypeConverter &typeConverter;
  TargetInfo &targetInfo;
};

struct ReturnOpConversion : public EIROpConversion<ReturnOp> {
  using EIROpConversion::EIROpConversion;

  PatternMatchResult
  matchAndRewrite(ReturnOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::ReturnOp>(op, operands);
    return matchSuccess();
  }
};

struct BranchOpConversion : public EIROpConversion<eir::BranchOp> {
  using EIROpConversion::EIROpConversion;

  PatternMatchResult
  matchAndRewrite(eir::BranchOp op, ArrayRef<Value> _properOperands,
                  ArrayRef<Block *> destinations,
                  ArrayRef<ArrayRef<Value>> operands,
                  ConversionPatternRewriter &rewriter) const override {

    auto dest = destinations.front();
    auto destArgs = operands.front();
    rewriter.replaceOpWithNewOp<mlir::BranchOp>(op, dest, destArgs);
    return matchSuccess();
  }
};

struct PrintOpConversion : public EIROpConversion<PrintOp> {
  using EIROpConversion::EIROpConversion;

  PatternMatchResult
  matchAndRewrite(PrintOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {

    // If print is called with no operands, just remove it for now
    if (operands.empty()) {
      rewriter.eraseOp(op);
      return matchSuccess();
    }
    
    LLVMDialect *dialect = typeConverter.getDialect();
    ModuleOp parentModule = op.getParentOfType<ModuleOp>();
    auto printfRef = getOrInsertPrintf(rewriter, parentModule, dialect, targetInfo);

    rewriter.replaceOpWithNewOp<mlir::CallOp>(
      op, printfRef, targetInfo.getTermType(), operands);
    return matchSuccess();
  }
};

struct UnreachableOpConversion : public EIROpConversion<UnreachableOp> {
  using EIROpConversion::EIROpConversion;

  PatternMatchResult
  matchAndRewrite(UnreachableOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::LLVM::UnreachableOp>(op, operands);
    return matchSuccess();
  }
};

struct ConstantFloatOpToStdConversion : public EIROpConversion<ConstantFloatOp> {
  using EIROpConversion::EIROpConversion;

  PatternMatchResult
  matchAndRewrite(ConstantFloatOp op,
                  ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // We lower directly to LLVM when using packed floats
    if (targetInfo.requiresPackedFloats())
      return matchFailure();

    // On nanboxed targets though, we can treat floats normally
    auto attr = op.getValue().cast<mlir::FloatAttr>();
    auto newAttr = rewriter.getF64FloatAttr(attr.getValueAsDouble());
    rewriter.replaceOpWithNewOp<mlir::ConstantOp>(op, newAttr);
    return matchSuccess();
  }
};

// Builds IR to construct a boxed term that points to the given header value,
// it is expected that the header value is a pointer value, not an immediate.
//
// The type of the resulting term is Term
static Value make_box(OpBuilder &builder,
                      edsc::ScopedContext &context,
                      LLVMTypeConverter &converter,
                      TargetInfo &targetInfo,
                      Value header) {
  auto headerTy = targetInfo.getHeaderType();
  Value headerPtrInt = llvm_ptrtoint(headerTy, header);
  auto boxTag = targetInfo.boxTag();
  Value boxTagConst = llvm_constant(headerTy, builder.getIntegerAttr(headerTy, boxTag));
  Value box = llvm_or(headerPtrInt, boxTagConst);
  return llvm_bitcast(targetInfo.getTermType(), box);
}

// Builds IR to construct a boxed list term
// it is expected that the cons cell value is a pointer value, not an immediate.
//
// The type of the resulting term is Term
static Value make_list(OpBuilder &builder,
                       edsc::ScopedContext &context,
                       LLVMTypeConverter &converter,
                       TargetInfo &targetInfo,
                       Value cons) {
  auto headerTy = targetInfo.getHeaderType();
  Value consPtrInt = llvm_ptrtoint(headerTy, cons);
  auto listTag = targetInfo.listTag();
  Value listTagConst = llvm_constant(headerTy, builder.getIntegerAttr(headerTy, listTag));
  Value list = llvm_or(consPtrInt, listTagConst);
  return llvm_bitcast(targetInfo.getTermType(), list);
}

struct ConstantFloatOpConversion : public EIROpConversion<ConstantFloatOp> {
  using EIROpConversion::EIROpConversion;

  PatternMatchResult
  matchAndRewrite(ConstantFloatOp op,
                  ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    edsc::ScopedContext context(rewriter, op.getLoc());

    auto attr = op.getValue().cast<FloatAttr>();
    auto ty = targetInfo.getFloatType();
    auto val = llvm_constant(ty, rewriter.getF64FloatAttr(attr.getValueAsDouble()));

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
    Value header = llvm_constant(headerTy, rewriter.getIntegerAttr(headerTy, headerVal));
    Value desc = llvm_undef(descTy);
    desc = llvm_insertvalue(descTy, desc, header, rewriter.getI64ArrayAttr(0));
    desc = llvm_insertvalue(descTy, desc, val, rewriter.getI64ArrayAttr(1));
    // NOTE: For now we aren't boxing the descriptor, any operations we lower
    // that reference values of our float type will need to insert the appropriate
    // operations to either box the value, or access the f64 contained within
    // directly
    rewriter.replaceOp(op, desc);
    return matchSuccess();
  }
};

struct ConstantIntOpConversion : public EIROpConversion<ConstantIntOp> {
  using EIROpConversion::EIROpConversion;

  PatternMatchResult
  matchAndRewrite(ConstantIntOp op,
                  ArrayRef<Value> operands,
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

  PatternMatchResult
  matchAndRewrite(ConstantBigIntOp _op,
                  ArrayRef<Value> _operands,
                  ConversionPatternRewriter &_rewriter) const override {
    assert(false && "ConstantBigIntOpConversion is unimplemented");
  }
};

struct ConstantAtomOpConversion : public EIROpConversion<ConstantAtomOp> {
  using EIROpConversion::EIROpConversion;

  PatternMatchResult
  matchAndRewrite(ConstantAtomOp op,
                  ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    edsc::ScopedContext context(rewriter, op.getLoc());

    auto atomAttr = op.getValue().cast<AtomAttr>();
    auto id = (uint64_t)atomAttr.getValue().getLimitedValue();
    auto ty = targetInfo.getAtomType();
    auto taggedAtom = targetInfo.encodeImmediate(TypeKind::Atom, id);
    auto val = llvm_constant(ty, rewriter.getIntegerAttr(rewriter.getIntegerType(64), taggedAtom));

    rewriter.replaceOp(op, {val});
    return matchSuccess();
  }
};
 
struct ConstantBinaryOpConversion : public EIROpConversion<ConstantBinaryOp> {
  using EIROpConversion::EIROpConversion;

  PatternMatchResult
  matchAndRewrite(ConstantBinaryOp op,
                  ArrayRef<Value> operands,
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
    Value boxTagConst = llvm_constant(headerTy, rewriter.getIntegerAttr(rewriter.getIntegerType(64), boxTag));

    // We use the SHA-1 hash of the value as the name of the global,
    // this provides a nice way to de-duplicate constant strings while
    // not requiring any global state
    auto name = binAttr.getHash();
    Value valPtr = getOrCreateGlobalString(
      context.getLocation(),
      context.getBuilder(),
      name,
      bytes,
      parentModule,
      dialect
    );
    Value valPtrLoad = llvm_load(valPtr);
    Value valPtrInt = llvm_ptrtoint(headerTy, valPtrLoad);
    Value boxedValPtr = llvm_or(valPtrInt, boxTagConst);
    //Value boxedVal =  llvm_bitcast(targetInfo.getTermType(), boxedValPtr);
    Value header = llvm_constant(headerTy, rewriter.getIntegerAttr(rewriter.getIntegerType(64), headerRaw));
    Value flags = llvm_constant(headerTy, rewriter.getIntegerAttr(rewriter.getIntegerType(64), flagsRaw));

    Value allocN = llvm_constant(headerTy, rewriter.getI64IntegerAttr(1));
    Value descAlloc = llvm_alloca(ty.getPointerTo(), allocN, rewriter.getI64IntegerAttr(8));

    auto tyPtrTy = ty.getPointerTo();
    Value desc = llvm_undef(ty);
    desc = llvm_insertvalue(ty, desc, header, rewriter.getI64ArrayAttr(0));
    desc = llvm_insertvalue(ty, desc, flags, rewriter.getI64ArrayAttr(1));
    desc = llvm_insertvalue(ty, desc, boxedValPtr, rewriter.getI64ArrayAttr(2));
    llvm_store(desc, descAlloc);
    Value descPtrInt = llvm_ptrtoint(headerTy, descAlloc);
    Value boxedDescPtr = llvm_or(descPtrInt, boxTagConst);
    Value boxedDesc = llvm_bitcast(targetInfo.getTermType(), boxedDescPtr);

    rewriter.replaceOp(op, boxedDesc);
    return matchSuccess();
  }
};

struct ConstantNilOpConversion : public EIROpConversion<ConstantNilOp> {
  using EIROpConversion::EIROpConversion;

  PatternMatchResult
  matchAndRewrite(ConstantNilOp op,
                  ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    edsc::ScopedContext context(rewriter, op.getLoc());

    auto ty = targetInfo.getNilType();
    auto val = llvm_constant(ty, rewriter.getIntegerAttr(ty, targetInfo.getNilValue()));

    rewriter.replaceOp(op, {val});
    return matchSuccess();
  }
};

struct ConstantNoneOpConversion : public EIROpConversion<ConstantNoneOp> {
  using EIROpConversion::EIROpConversion;

  PatternMatchResult
  matchAndRewrite(ConstantNoneOp op,
                  ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    edsc::ScopedContext context(rewriter, op.getLoc());

    auto ty = targetInfo.getNoneType();
    auto val = llvm_constant(ty, rewriter.getIntegerAttr(ty, targetInfo.getNoneValue()));

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
      assert(i.getBitWidth() <= targetInfo.pointerSizeInBits && "support for bigint in constant aggregates not yet implemented");
      auto ty = targetInfo.getFixnumType();
      auto tagged = targetInfo.encodeImmediate(TypeKind::Fixnum, i.getLimitedValue());
      auto val = llvm_constant(ty, rewriter.getIntegerAttr(ty, tagged));
      elementTypes.push_back(ty);
      elementValues.push_back(val);
      continue;
    }
    if (auto floatAttr = elementAttr.dyn_cast_or_null<FloatAttr>()) {
      auto f = floatAttr.getValue().bitcastToAPInt();
      assert(!targetInfo.requiresPackedFloats() && "support for packed floats in constant aggregates is not yet implemented");
      auto ty = targetInfo.getFloatType();
      auto val = llvm_constant(ty, rewriter.getIntegerAttr(ty, f.getLimitedValue()));
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

  PatternMatchResult
  matchAndRewrite(ConstantTupleOp op,
                  ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    edsc::ScopedContext context(rewriter, op.getLoc());

    auto attr = op.getValue().cast<SeqAttr>();
    auto elements = attr.getValue();
    auto numElements = elements.size();

    SmallVector<LLVMType, 2> elementTypes;
    elementTypes.resize(numElements);
    SmallVector<Value, 2> elementValues;
    elementValues.resize(numElements);

    auto lowered = lowerElementValues(context, rewriter, targetInfo, elements, elementValues, elementTypes);
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

  PatternMatchResult
  matchAndRewrite(ConstantListOp op,
                  ArrayRef<mlir::Value> operands,
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

    auto lowered = lowerElementValues(context, rewriter, targetInfo, elements, elementValues, elementTypes);
    assert(lowered && "unsupported element type in list constant");

    auto consTy = targetInfo.getConsType();

    // Lower to single cons cell if elements <= 2
    if (numElements <= 2) {
      Value desc = llvm_undef(consTy);
      desc = llvm_insertvalue(consTy, desc, elementValues[0], rewriter.getI64ArrayAttr(0));
      if (numElements == 2) {
        desc = llvm_insertvalue(consTy, desc, elementValues[1], rewriter.getI64ArrayAttr(1));
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
    lastCons = llvm_insertvalue(consTy, lastCons, nilVal, rewriter.getI64ArrayAttr(1));
    lastCons = llvm_insertvalue(consTy, lastCons, elementValues[--currentIndex], rewriter.getI64ArrayAttr(0));
    // Create all cells from tail to head
    Value prev = lastCons;
    for (auto i = cellsRequired; i > 1; --i) {
      Value curr = llvm_undef(consTy);
      auto prevBoxed = make_list(rewriter, context, typeConverter, targetInfo, prev);
      curr = llvm_insertvalue(consTy, curr, prevBoxed, rewriter.getI64ArrayAttr(1));
      curr = llvm_insertvalue(consTy, curr, elementValues[--currentIndex], rewriter.getI64ArrayAttr(0));
      prev = curr;
    }

    auto head = make_list(rewriter, context, typeConverter, targetInfo, prev);

    rewriter.replaceOp(op, head);
    return matchSuccess();
  }
};

}  // namespace

static void populateEIRToStandardConversionPatterns(
    OwningRewritePatternList &patterns,
    mlir::MLIRContext *context,
    LLVMTypeConverter &converter,
    TargetInfo &targetInfo) {
  patterns.insert<ReturnOpConversion,
                  /*FuncOpConversion,
                  */
                  BranchOpConversion,
                  /*
                  CondBranchOpConversion,
                  IfOpConversion,
                  MatchOpConversion,
                  ConstructMapOpConversion,
                  MapInsertOpConversion,
                  MapUpdateOpConversion,
                  */
                  PrintOpConversion,
                  ConstantFloatOpToStdConversion
                  >(context, converter, targetInfo);
}

/// Populate the given list with patterns that convert from EIR to LLVM
void populateEIRToLLVMConversionPatterns(
    OwningRewritePatternList &patterns,
    MLIRContext *context,
    LLVMTypeConverter &converter,
    TargetInfo &targetInfo) {
  patterns.insert<
                  YieldOpConversion,
                  TraceConstructOpConversion>(context);
  patterns.insert<UnreachableOpConversion,
                  /*
                  CallOpConversion,
                  GetElementPtrOpConversion,
                  LoadOpConversion,
                  IsTypeOpConversion,
                  LogicalAndOpConversion,
                  LogicalOrOpConversion,
                  CmpEqOpConversion,
                  CmpNeqOpConversion,
                  CmpLtOpConversion,
                  CmpLteOpConversion,
                  CmpGtOpConversion,
                  CmpGteOpConversion,
                  CmpNerrOpConversion,
                  ThrowOpConversion,
                  CastOpConversion,
                  ConsOpConversion,
                  TupleOpConversion,
                  TraceCaptureOpConversion,
                  BinaryPushOpConversion,
                  */
                  ConstantFloatOpConversion,
                  ConstantIntOpConversion,
                  ConstantBigIntOpConversion,
                  ConstantAtomOpConversion,
                  ConstantBinaryOpConversion,
                  ConstantNilOpConversion,
                  ConstantNoneOpConversion,
                  ConstantTupleOpConversion,
                  ConstantListOpConversion/*,
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
    populateEIRToStandardConversionPatterns(patterns, &context, converter, targetInfo);
    populateEIRToLLVMConversionPatterns(patterns, &context, converter, targetInfo);

    // Define the legality of the operations we're converting to
    mlir::ConversionTarget conversionTarget(context);
    conversionTarget.addLegalDialect<mlir::LLVM::LLVMDialect>();
    conversionTarget.addDynamicallyLegalOp<mlir::FuncOp>(
      [&](mlir::FuncOp op) { return converter.isSignatureLegal(op.getType()); });
    conversionTarget.addLegalOp<ModuleOp, ModuleTerminatorOp>();

    mlir::ModuleOp moduleOp = getModule();
    if (failed(applyFullConversion(moduleOp, conversionTarget,
                                   patterns, &converter))) {
      moduleOp.emitError() << "conversion to LLVM IR dialect failed";
      return signalPassFailure();
    }
  }
private:
  TargetMachine *targetMachine;
};

}  // namespace

std::unique_ptr<mlir::OpPassBase<mlir::ModuleOp>>
createConvertEIRToLLVMPass(TargetMachine *targetMachine) {
  return std::make_unique<ConvertEIRToLLVMPass>(targetMachine);
}

}  // namespace eir
}  // namespace lumen
