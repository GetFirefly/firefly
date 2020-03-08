#include "lumen/compiler/Dialect/EIR/Conversion/EIRToLLVM/MathOpConversions.h"

namespace lumen {
namespace eir {
template <typename T>
static Value specializeIntegerMathOp(OpBuilder &builder,
                                     edsc::ScopedContext &context,
                                     TargetInfo &targetInfo, Location loc,
                                     Value lhs, Value rhs) {
  Value lhsInt = do_unmask_immediate(builder, context, targetInfo, lhs);
  Value rhsInt = do_unmask_immediate(builder, context, targetInfo, rhs);
  auto mathOp = builder.create<T>(loc, lhsInt, rhsInt);
  return mathOp.getResult();
}

template <typename T>
static Value specializeFloatMathOp(OpBuilder &builder,
                                   edsc::ScopedContext &context,
                                   TargetInfo &targetInfo, LLVMDialect *dialect,
                                   Location loc, Value lhs, Value rhs) {
  auto fpTy = LLVMType::getDoubleTy(dialect);
  if (!targetInfo.requiresPackedFloats()) {
    Value lhsFp = llvm_bitcast(fpTy, lhs);
    Value rhsFp = llvm_bitcast(fpTy, rhs);
    auto fpOp = builder.create<T>(loc, lhs, rhs);
    return fpOp.getResult();
  } else {
    auto int32Ty = LLVMType::getInt32Ty(dialect);
    auto floatTy = targetInfo.getFloatType();
    auto indexTy = builder.getIntegerType(32);
    Value cns0 = llvm_constant(int32Ty, builder.getIntegerAttr(indexTy, 0));
    Value index = llvm_constant(int32Ty, builder.getIntegerAttr(indexTy, 1));
    ArrayRef<Value> indices({cns0, index});
    Value lhsPtr = llvm_gep(fpTy.getPointerTo(), lhs, indices);
    Value rhsPtr = llvm_gep(fpTy.getPointerTo(), rhs, indices);
    Value lhsVal = llvm_load(lhsPtr);
    Value rhsVal = llvm_load(rhsPtr);
    auto fpOp = builder.create<T>(loc, lhsVal, rhsVal);
    return fpOp.getResult();
  }
}

template <typename Op, typename OperandAdaptor, typename IntOp,
          typename FloatOp>
class MathOpConversion : public EIROpConversion<Op> {
 public:
  explicit MathOpConversion(MLIRContext *context, LLVMTypeConverter &converter_,
                            TargetInfo &targetInfo_,
                            mlir::PatternBenefit benefit = 1)
      : EIROpConversion<Op>::EIROpConversion(context, converter_, targetInfo_,
                                             benefit) {}

  PatternMatchResult matchAndRewrite(
      Op op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    edsc::ScopedContext context(rewriter, op.getLoc());
    OperandAdaptor adaptor(operands);

    Value lhs = adaptor.lhs();
    Value rhs = adaptor.rhs();
    Type lhsTy = op.getOperand(0).getType();
    Type rhsTy = op.getOperand(1).getType();

    // Use specialized lowerings if types are compatible
    if (lhsTy.isa<FixnumType>() && rhsTy.isa<FixnumType>()) {
      auto newOp = specializeIntegerMathOp<IntOp>(rewriter, context, targetInfo,
                                                  op.getLoc(), lhs, rhs);
      rewriter.replaceOp(op, newOp);
      return matchSuccess();
    }
    if (lhsTy.isa<FloatType>() && rhsTy.isa<FloatType>()) {
      auto newOp = specializeFloatMathOp<FloatOp>(
          rewriter, context, targetInfo, dialect, op.getLoc(), lhs, rhs);
      rewriter.replaceOp(op, newOp);
      return matchSuccess();
    }

    // Call builtin function
    StringRef builtinSymbol = Op::builtinSymbol();
    ModuleOp parentModule = op.template getParentOfType<ModuleOp>();
    auto termTy = getUsizeType();
    auto callee = getOrInsertFunction(rewriter, parentModule, builtinSymbol,
                                      termTy, {termTy, termTy});

    ArrayRef<Value> args({lhs, rhs});
    auto callOp = rewriter.create<mlir::CallOp>(op.getLoc(), callee,
                                                ArrayRef<Type>{termTy}, args);
    auto result = callOp.getResult(0);

    rewriter.replaceOp(op, result);
    return matchSuccess();
  }

 private:
  using EIROpConversion<Op>::matchSuccess;
  using EIROpConversion<Op>::getUsizeType;
  using EIROpConversion<Op>::getOrInsertFunction;
  using EIROpConversion<Op>::targetInfo;
  using EIROpConversion<Op>::dialect;
};

struct AddOpConversion : public MathOpConversion<AddOp, AddOpOperandAdaptor,
                                                 LLVM::AddOp, LLVM::FAddOp> {
  using MathOpConversion::MathOpConversion;
};
struct SubOpConversion : public MathOpConversion<SubOp, SubOpOperandAdaptor,
                                                 LLVM::SubOp, LLVM::FSubOp> {
  using MathOpConversion::MathOpConversion;
};
struct MulOpConversion : public MathOpConversion<MulOp, MulOpOperandAdaptor,
                                                 LLVM::MulOp, LLVM::FMulOp> {
  using MathOpConversion::MathOpConversion;
};

template <typename Op, typename OperandAdaptor, typename IntOp>
class IntegerMathOpConversion : public EIROpConversion<Op> {
 public:
  explicit IntegerMathOpConversion(MLIRContext *context,
                                   LLVMTypeConverter &converter_,
                                   TargetInfo &targetInfo_,
                                   mlir::PatternBenefit benefit = 1)
      : EIROpConversion<Op>::EIROpConversion(context, converter_, targetInfo_,
                                             benefit) {}

  PatternMatchResult matchAndRewrite(
      Op op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    edsc::ScopedContext context(rewriter, op.getLoc());
    OperandAdaptor adaptor(operands);

    Value lhs = adaptor.lhs();
    Value rhs = adaptor.rhs();
    Type lhsTy = op.getOperand(0).getType();
    Type rhsTy = op.getOperand(1).getType();

    // Use specialized lowerings if types are compatible
    if (lhsTy.isa<FixnumType>() && rhsTy.isa<FixnumType>()) {
      auto newOp = specializeIntegerMathOp<IntOp>(rewriter, context, targetInfo,
                                                  op.getLoc(), lhs, rhs);
      rewriter.replaceOp(op, newOp);
      return matchSuccess();
    }

    // Call builtin function
    StringRef builtinSymbol = Op::builtinSymbol();
    ModuleOp parentModule = op.template getParentOfType<ModuleOp>();
    auto termTy = getUsizeType();
    auto callee = getOrInsertFunction(rewriter, parentModule, builtinSymbol,
                                      termTy, {termTy, termTy});

    ArrayRef<Value> args({lhs, rhs});
    auto callOp = rewriter.create<mlir::CallOp>(op.getLoc(), callee,
                                                ArrayRef<Type>{termTy}, args);
    auto result = callOp.getResult(0);

    rewriter.replaceOp(op, result);
    return matchSuccess();
  }

 private:
  using EIROpConversion<Op>::matchSuccess;
  using EIROpConversion<Op>::getUsizeType;
  using EIROpConversion<Op>::getOrInsertFunction;
  using EIROpConversion<Op>::targetInfo;
  using EIROpConversion<Op>::dialect;
};

struct DivOpConversion
    : public IntegerMathOpConversion<DivOp, DivOpOperandAdaptor, LLVM::SDivOp> {
  using IntegerMathOpConversion::IntegerMathOpConversion;
};
struct RemOpConversion
    : public IntegerMathOpConversion<RemOp, RemOpOperandAdaptor, LLVM::SRemOp> {
  using IntegerMathOpConversion::IntegerMathOpConversion;
};
struct BandOpConversion
    : public IntegerMathOpConversion<BandOp, BandOpOperandAdaptor,
                                     LLVM::AndOp> {
  using IntegerMathOpConversion::IntegerMathOpConversion;
};
struct BorOpConversion
    : public IntegerMathOpConversion<BorOp, BorOpOperandAdaptor, LLVM::OrOp> {
  using IntegerMathOpConversion::IntegerMathOpConversion;
};
struct BxorOpConversion
    : public IntegerMathOpConversion<BxorOp, BxorOpOperandAdaptor,
                                     LLVM::XOrOp> {
  using IntegerMathOpConversion::IntegerMathOpConversion;
};
struct BslOpConversion
    : public IntegerMathOpConversion<BslOp, BslOpOperandAdaptor, LLVM::ShlOp> {
  using IntegerMathOpConversion::IntegerMathOpConversion;
};
struct BsrOpConversion
    : public IntegerMathOpConversion<BsrOp, BsrOpOperandAdaptor, LLVM::AShrOp> {
  using IntegerMathOpConversion::IntegerMathOpConversion;
};

template <typename Op, typename OperandAdaptor, typename FloatOp>
class FloatMathOpConversion : public EIROpConversion<Op> {
 public:
  explicit FloatMathOpConversion(MLIRContext *context,
                                 LLVMTypeConverter &converter_,
                                 TargetInfo &targetInfo_,
                                 mlir::PatternBenefit benefit = 1)
      : EIROpConversion<Op>::EIROpConversion(context, converter_, targetInfo_,
                                             benefit) {}

  PatternMatchResult matchAndRewrite(
      Op op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    edsc::ScopedContext context(rewriter, op.getLoc());
    OperandAdaptor adaptor(operands);

    Value lhs = adaptor.lhs();
    Value rhs = adaptor.rhs();
    Type lhsTy = op.getOperand(0).getType();
    Type rhsTy = op.getOperand(1).getType();

    // Use specialized lowerings if types are compatible
    if (lhsTy.isa<FloatType>() && rhsTy.isa<FloatType>()) {
      auto newOp = specializeIntegerMathOp<FloatOp>(
          rewriter, context, targetInfo, op.getLoc(), lhs, rhs);
      rewriter.replaceOp(op, newOp);
      return matchSuccess();
    }

    // Call builtin function
    StringRef builtinSymbol = Op::builtinSymbol();
    ModuleOp parentModule = op.template getParentOfType<ModuleOp>();
    auto termTy = getUsizeType();
    auto callee = getOrInsertFunction(rewriter, parentModule, builtinSymbol,
                                      termTy, {termTy, termTy});

    ArrayRef<Value> args({lhs, rhs});
    auto callOp = rewriter.create<mlir::CallOp>(op.getLoc(), callee,
                                                ArrayRef<Type>{termTy}, args);
    auto result = callOp.getResult(0);

    rewriter.replaceOp(op, result);
    return matchSuccess();
  }

 private:
  using EIROpConversion<Op>::matchSuccess;
  using EIROpConversion<Op>::getUsizeType;
  using EIROpConversion<Op>::getOrInsertFunction;
  using EIROpConversion<Op>::targetInfo;
  using EIROpConversion<Op>::dialect;
};

struct FDivOpConversion
    : public FloatMathOpConversion<FDivOp, FDivOpOperandAdaptor, LLVM::FDivOp> {
  using FloatMathOpConversion::FloatMathOpConversion;
};

void populateMathOpConversionPatterns(OwningRewritePatternList &patterns,
                                      MLIRContext *context,
                                      LLVMTypeConverter &converter,
                                      TargetInfo &targetInfo) {
  patterns.insert<AddOpConversion, SubOpConversion, MulOpConversion,
                  DivOpConversion, FDivOpConversion, RemOpConversion,
                  BslOpConversion, BsrOpConversion, BandOpConversion,
                  BorOpConversion, BxorOpConversion>(context, converter,
                                                     targetInfo);
}

}  // namespace eir
}  // namespace lumen
