#include "lumen/EIR/Conversion/MathOpConversions.h"

namespace lumen {
namespace eir {

template <typename Op>
static void buildDeoptimizationPath(
  Location loc,
  RewritePatternContext<Op> &ctx,
  Op op,
  OpaqueTermType concreteTy,
  StringRef intrinsicFn,
  StringRef runtimeFn,
  Value lhs, Value rhs) {

  Operation *rawOp = op.getOperation();
  Block *current = rawOp->getBlock();

  Value lhsRaw = ctx.decodeImmediate(lhs);
  Value rhsRaw = ctx.decodeImmediate(rhs);

  // Build math op with overflow/underflow intrinsic
  auto callee = ctx.rewriter.getSymbolRefAttr(intrinsicFn);
  auto i1Ty = ctx.getI1Type();
  auto termTy = ctx.getUsizeType();
  auto i64Ty = ctx.getI64Type();
  // TODO: Need to handle 32-bit or non-nanboxed 64 arches here
  auto iFixTy = LLVMType::getIntNTy(ctx.rewriter.getContext(), 46);
  auto resTy = LLVMType::getStructTy(ctx.rewriter.getContext(),
                                     ArrayRef<LLVMType>{iFixTy, i1Ty},
                                     /*packed=*/false);
  auto calleeFn = ctx.getOrInsertFunction(intrinsicFn, resTy, {iFixTy, iFixTy});
  Value lhsTrunc = llvm_trunc(iFixTy, lhsRaw);
  Value rhsTrunc = llvm_trunc(iFixTy, rhsRaw);
  Operation *callOp = llvm_call(ArrayRef<Type>{resTy}, callee, ArrayRef<Value>{lhsTrunc, rhsTrunc});

  // Extract results
  Value results = callOp->getResult(0);
  Value resultFix = llvm_extractvalue(iFixTy, results, ctx.getI64ArrayAttr(0));
  Value obit = llvm_extractvalue(i1Ty, results, ctx.getI64ArrayAttr(1));

  // Split block after the above op
  auto it = llvm::iplist<Operation>::iterator(obit.getDefiningOp());
  auto splitStart = &*std::next(it);
  Block *cont = current->splitBlock(splitStart);
  cont->addArgument(termTy);

  // Create overflow/normal blocks
  Block *overflow = new Block();
  Block *normal = new Block();
  auto nextIt = std::next(Region::iterator(current));
  current->getParent()->getBlocks().insert(nextIt, overflow);
  current->getParent()->getBlocks().insert(nextIt, normal);

  // Based on whether there was overflow/underflow,
  // either branch to the deoptimized path (i.e. call
  // to runtime function) or branch to the intermediate
  // block where we re-encode the result and continue
  // execution where we left off
  ctx.rewriter.setInsertionPointToEnd(current);
  llvm_condbr(obit, normal, ValueRange(), overflow, ValueRange());

  // Handle normal
  ctx.rewriter.setInsertionPointToEnd(normal);
  Value extended = llvm_zext(termTy, resultFix);
  Value encoded = ctx.encodeImmediate(concreteTy, extended);
  llvm_br(ValueRange(encoded), cont);

  // Handle overflow
  ctx.rewriter.setInsertionPointToStart(overflow);
  auto rtCallee = ctx.rewriter.getSymbolRefAttr(runtimeFn);
  Operation *rtCallOp = llvm_call(ArrayRef<Type>{termTy}, rtCallee, ArrayRef<Value>{lhs, rhs});
  llvm_br(rtCallOp->getResults(), cont);

  // Replace original op with the final value
  Value output = cont->getArgument(0);
  ctx.rewriter.replaceOp(op, {output});
}

template <typename Op, typename T>
static Value specializeFloatMathOp(Location loc, RewritePatternContext<Op> &ctx,
                                   Value lhs, Value rhs) {
  auto fpTy = ctx.getDoubleType();
  Value l = eir_cast(lhs, fpTy);
  Value r = eir_cast(rhs, fpTy);
  auto fpOp = ctx.rewriter.template create<T>(loc, l, r);
  return fpOp.getResult();
}

template <typename Op, typename OperandAdaptor>
class SpecializedMathOpConversion : public EIROpConversion<Op> {
public:
  explicit SpecializedMathOpConversion(MLIRContext *context, EirTypeConverter &converter_,
                            TargetInfo &targetInfo_,
                            mlir::PatternBenefit benefit = 1)
      : EIROpConversion<Op>::EIROpConversion(context, converter_, targetInfo_,
                                             benefit) {}

  LogicalResult matchAndRewrite(
      Op op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    OperandAdaptor adaptor(operands);
    auto ctx = getRewriteContext(op, rewriter);

    Value lhs = adaptor.lhs();
    Value rhs = adaptor.rhs();
    Type lhsTy = op.getOperand(0).getType();
    Type rhsTy = op.getOperand(1).getType();
    StringRef intrinsic = Op::intrinsicSymbol();
    StringRef builtinSymbol = Op::builtinSymbol();

    // Use specialized lowerings if types are compatible
    if (lhsTy.isa<FixnumType>() && rhsTy.isa<FixnumType>()) {
      auto fixTy = rewriter.getType<FixnumType>();
      buildDeoptimizationPath(loc, ctx, op, fixTy, intrinsic, builtinSymbol, lhs, rhs);
      return success();
    }

    // Call builtin function
    auto termTy = ctx.getUsizeType();
    auto callee =
        ctx.getOrInsertFunction(builtinSymbol, termTy, {termTy, termTy});

    ArrayRef<Value> args({lhs, rhs});
    auto calleeSymbol =
        FlatSymbolRefAttr::get(builtinSymbol, callee->getContext());

    rewriter.replaceOpWithNewOp<mlir::CallOp>(op, calleeSymbol,
                                              ArrayRef<Type>{termTy}, args);
    return success();
  }

 private:
  using EIROpConversion<Op>::getRewriteContext;
};

template <typename Op, typename OperandAdaptor>
class MathOpConversion : public EIROpConversion<Op> {
 public:
  explicit MathOpConversion(MLIRContext *context, EirTypeConverter &converter_,
                            TargetInfo &targetInfo_,
                            mlir::PatternBenefit benefit = 1)
      : EIROpConversion<Op>::EIROpConversion(context, converter_, targetInfo_,
                                             benefit) {}

  LogicalResult matchAndRewrite(
      Op op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    OperandAdaptor adaptor(operands);
    auto ctx = getRewriteContext(op, rewriter);

    Value lhs = adaptor.lhs();
    Value rhs = adaptor.rhs();

    // Call builtin function
    StringRef builtinSymbol = Op::builtinSymbol();
    auto termTy = ctx.getUsizeType();
    auto callee =
        ctx.getOrInsertFunction(builtinSymbol, termTy, {termTy, termTy});

    ArrayRef<Value> args({lhs, rhs});
    auto calleeSymbol =
        FlatSymbolRefAttr::get(builtinSymbol, callee->getContext());

    rewriter.replaceOpWithNewOp<mlir::CallOp>(op, calleeSymbol,
                                              ArrayRef<Type>{termTy}, args);
    return success();
  }

 private:
  using EIROpConversion<Op>::getRewriteContext;
};

struct NegOpConversion : public EIROpConversion<NegOp> {
  using EIROpConversion::EIROpConversion;

  LogicalResult matchAndRewrite(
      NegOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    NegOpAdaptor adaptor(operands);
    auto ctx = getRewriteContext(op, rewriter);

    Value rhs = adaptor.rhs();
    Type rhsTy = rhs.getType();

    // Call builtin function
    StringRef builtinSymbol("erlang:-/1");
    auto termTy = ctx.getUsizeType();
    auto callee = ctx.getOrInsertFunction(builtinSymbol, termTy, {termTy});

    ArrayRef<Value> args({rhs});
    auto calleeSymbol =
        FlatSymbolRefAttr::get(builtinSymbol, callee->getContext());

    rewriter.replaceOpWithNewOp<mlir::CallOp>(op, calleeSymbol,
                                              ArrayRef<Type>{termTy}, args);
    return success();
  }
};

struct AddOpConversion
    : public SpecializedMathOpConversion<AddOp, AddOpAdaptor> {
  using SpecializedMathOpConversion::SpecializedMathOpConversion;
};
struct SubOpConversion
    : public SpecializedMathOpConversion<SubOp, SubOpAdaptor> {
  using SpecializedMathOpConversion::SpecializedMathOpConversion;
};
struct MulOpConversion
    : public SpecializedMathOpConversion<MulOp, MulOpAdaptor> {
  using SpecializedMathOpConversion::SpecializedMathOpConversion;
};

template <typename Op, typename OperandAdaptor>
class IntegerMathOpConversion : public EIROpConversion<Op> {
 public:
  explicit IntegerMathOpConversion(MLIRContext *context,
                                   EirTypeConverter &converter_,
                                   TargetInfo &targetInfo_,
                                   mlir::PatternBenefit benefit = 1)
      : EIROpConversion<Op>::EIROpConversion(context, converter_, targetInfo_,
                                             benefit) {}

  LogicalResult matchAndRewrite(
      Op op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    OperandAdaptor adaptor(operands);
    auto ctx = getRewriteContext(op, rewriter);

    Value lhs = adaptor.lhs();
    Value rhs = adaptor.rhs();
    Type lhsTy = op.getOperand(0).getType();
    Type rhsTy = op.getOperand(1).getType();

    // Call builtin function
    StringRef builtinSymbol = Op::builtinSymbol();
    auto termTy = ctx.getUsizeType();
    auto callee =
        ctx.getOrInsertFunction(builtinSymbol, termTy, {termTy, termTy});

    ArrayRef<Value> args({lhs, rhs});
    auto calleeSymbol =
        FlatSymbolRefAttr::get(builtinSymbol, callee->getContext());
    rewriter.replaceOpWithNewOp<mlir::CallOp>(op, calleeSymbol,
                                              ArrayRef<Type>{termTy}, args);
    return success();
  }

 private:
  using EIROpConversion<Op>::getRewriteContext;
};

// llvm.sdiv
struct DivOpConversion
    : public IntegerMathOpConversion<DivOp, DivOpAdaptor> {
  using IntegerMathOpConversion::IntegerMathOpConversion;
};
// llvm.srem
struct RemOpConversion
    : public IntegerMathOpConversion<RemOp, RemOpAdaptor> {
  using IntegerMathOpConversion::IntegerMathOpConversion;
};
// llvm.and
struct BandOpConversion
    : public IntegerMathOpConversion<BandOp, BandOpAdaptor> {
  using IntegerMathOpConversion::IntegerMathOpConversion;
};
// llvm.or
struct BorOpConversion
    : public IntegerMathOpConversion<BorOp, BorOpAdaptor> {
  using IntegerMathOpConversion::IntegerMathOpConversion;
};
// llvm.xor
struct BxorOpConversion
    : public IntegerMathOpConversion<BxorOp, BxorOpAdaptor> {
  using IntegerMathOpConversion::IntegerMathOpConversion;
};
// llvm.shl
struct BslOpConversion
    : public IntegerMathOpConversion<BslOp, BslOpAdaptor> {
  using IntegerMathOpConversion::IntegerMathOpConversion;
};
// llvm.ashr
struct BsrOpConversion
    : public IntegerMathOpConversion<BsrOp, BsrOpAdaptor> {
  using IntegerMathOpConversion::IntegerMathOpConversion;
};

template <typename Op, typename OperandAdaptor, typename FloatOp>
class FloatMathOpConversion : public EIROpConversion<Op> {
 public:
  explicit FloatMathOpConversion(MLIRContext *context,
                                 EirTypeConverter &converter_,
                                 TargetInfo &targetInfo_,
                                 mlir::PatternBenefit benefit = 1)
      : EIROpConversion<Op>::EIROpConversion(context, converter_, targetInfo_,
                                             benefit) {}

  LogicalResult matchAndRewrite(
      Op op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    OperandAdaptor adaptor(operands);
    auto ctx = getRewriteContext(op, rewriter);

    Value lhs = adaptor.lhs();
    Value rhs = adaptor.rhs();
    Type lhsTy = op.getOperand(0).getType();
    Type rhsTy = op.getOperand(1).getType();

    // Use specialized lowerings if types are compatible
    if (lhsTy.isa<FloatType>() && rhsTy.isa<FloatType>()) {
      auto result = specializeFloatMathOp<Op, FloatOp>(loc, ctx, lhs, rhs);
      rewriter.replaceOp(op, result);
      return success();
    }

    // Call builtin function
    StringRef builtinSymbol = Op::builtinSymbol();
    auto termTy = ctx.getUsizeType();
    auto callee =
        ctx.getOrInsertFunction(builtinSymbol, termTy, {termTy, termTy});

    ArrayRef<Value> args({lhs, rhs});
    auto calleeSymbol =
        FlatSymbolRefAttr::get(builtinSymbol, callee->getContext());

    rewriter.replaceOpWithNewOp<mlir::CallOp>(op, calleeSymbol,
                                              ArrayRef<Type>{termTy}, args);
    return success();
  }

 private:
  using EIROpConversion<Op>::getRewriteContext;
};

struct FDivOpConversion
    : public FloatMathOpConversion<FDivOp, FDivOpAdaptor, LLVM::FDivOp> {
  using FloatMathOpConversion::FloatMathOpConversion;
};

template <typename Op, typename OperandAdaptor, typename LogicalOp>
class LogicalOpConversion : public EIROpConversion<Op> {
 public:
  explicit LogicalOpConversion(MLIRContext *context,
                               EirTypeConverter &converter_,
                               TargetInfo &targetInfo_,
                               mlir::PatternBenefit benefit = 1)
      : EIROpConversion<Op>::EIROpConversion(context, converter_, targetInfo_,
                                             benefit) {}

  LogicalResult matchAndRewrite(
      Op op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    OperandAdaptor adaptor(operands);
    auto ctx = getRewriteContext(op, rewriter);

    Value lhs = adaptor.lhs();
    Value rhs = adaptor.rhs();
    Type lhsTy = op.getOperand(0).getType();
    Type rhsTy = op.getOperand(1).getType();

    auto i1Ty = ctx.getI1Type();
    auto boolTy = rewriter.getType<BooleanType>();

    if (lhsTy.isa<BooleanType>() && rhsTy.isa<BooleanType>()) {
      Value lhsBool = eir_cast(lhs, lhsTy, i1Ty);
      Value rhsBool = eir_cast(rhs, rhsTy, i1Ty);
      rewriter.replaceOpWithNewOp<LogicalOp>(op, lhsBool, rhsBool);
      return success();
    }

    if (lhsTy.isInteger(1) && rhsTy.isInteger(1)) {
      rewriter.replaceOpWithNewOp<LogicalOp>(op, lhs, rhs);
      return success();
    }

    // Call builtin function
    StringRef builtinSymbol = Op::builtinSymbol();
    auto termTy = ctx.getUsizeType();
    auto callee =
        ctx.getOrInsertFunction(builtinSymbol, termTy, {termTy, termTy});

    ArrayRef<Value> args({lhs, rhs});
    auto calleeSymbol =
        FlatSymbolRefAttr::get(builtinSymbol, callee->getContext());
    auto callOp = rewriter.create<mlir::CallOp>(op.getLoc(), calleeSymbol, ArrayRef<Type>{boolTy}, args);
    rewriter.replaceOpWithNewOp<CastOp>(op, callOp.getResult(0), i1Ty);
    return success();
  }

 private:
  using EIROpConversion<Op>::getRewriteContext;
};

struct LogicalAndOpConversion
    : public LogicalOpConversion<LogicalAndOp, LogicalAndOpAdaptor, LLVM::AndOp> {
  using LogicalOpConversion::LogicalOpConversion;
};

struct LogicalOrOpConversion
    : public LogicalOpConversion<LogicalOrOp, LogicalOrOpAdaptor, LLVM::OrOp> {
  using LogicalOpConversion::LogicalOpConversion;
};

void populateMathOpConversionPatterns(OwningRewritePatternList &patterns,
                                      MLIRContext *context,
                                      EirTypeConverter &converter,
                                      TargetInfo &targetInfo) {
  patterns.insert<AddOpConversion, SubOpConversion, NegOpConversion,
                  MulOpConversion, DivOpConversion, FDivOpConversion,
                  RemOpConversion, BslOpConversion, BsrOpConversion,
                  BandOpConversion, BorOpConversion, BxorOpConversion,
                  LogicalAndOpConversion, LogicalOrOpConversion>(
      context, converter, targetInfo);
}

}  // namespace eir
}  // namespace lumen
