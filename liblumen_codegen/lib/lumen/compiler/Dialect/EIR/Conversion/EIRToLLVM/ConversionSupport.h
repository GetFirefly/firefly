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
using llvm_shr = ValueBuilder<LLVM::LShrOp>;
using llvm_bitcast = ValueBuilder<LLVM::BitcastOp>;
using llvm_zext = ValueBuilder<LLVM::ZExtOp>;
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

namespace lumen {
namespace eir {

FlatSymbolRefAttr createOrInsertFunction(PatternRewriter &rewriter,
                                         ModuleOp mod, LLVMDialect *dialect,
                                         TargetInfo &targetInfo,
                                         StringRef symbol, LLVMType resultType,
                                         ArrayRef<LLVMType> argTypes = {});

static LLVM::GlobalOp createOrInsertGlobal(
    PatternRewriter &rewriter, ModuleOp mod, LLVMDialect *dialect,
    TargetInfo &targetInfo, StringRef name, LLVMType valueTy,
    LLVM::Linkage linkage = LLVM::Linkage::Internal,
    LLVM::ThreadLocalMode tlsMode = LLVM::ThreadLocalMode::NotThreadLocal);

/// Return a value representing an access into a global string with the given
/// name, creating the string if necessary.
static Value getOrCreateGlobalString(Location loc, OpBuilder &builder,
                                     StringRef name, StringRef value,
                                     ModuleOp mod, LLVMDialect *dialect);
// Builds IR to construct a boxed list term
// it is expected that the cons cell value is a pointer value, not an immediate.
//
// The type of the resulting term is Term
static Value do_make_list(OpBuilder &builder, edsc::ScopedContext &context,
                          LLVMTypeConverter &converter, TargetInfo &targetInfo,
                          Value cons);
static Value do_box(OpBuilder &builder, edsc::ScopedContext &context,
                    LLVMTypeConverter &converter, TargetInfo &targetInfo,
                    Value val);
static Value do_box_literal(OpBuilder &builder, edsc::ScopedContext &context,
                            LLVMTypeConverter &converter,
                            TargetInfo &targetInfo, Value val);
static Value do_unbox(OpBuilder &builder, edsc::ScopedContext &context,
                      LLVMTypeConverter &converter, TargetInfo &targetInfo,
                      LLVMType innerTy, Value box);
static Value do_unbox_list(OpBuilder &builder, edsc::ScopedContext &context,
                           LLVMTypeConverter &converter, TargetInfo &targetInfo,
                           LLVMType innerTy, Value box);
static Value do_mask_immediate(OpBuilder &builder, edsc::ScopedContext &context,
                               TargetInfo &targetInfo, Value val);
Value do_unmask_immediate(OpBuilder &builder, edsc::ScopedContext &context,
                          TargetInfo &targetInfo, Value val);

template <typename Op>
class EIROpConversion : public mlir::OpConversionPattern<Op> {
 public:
  explicit EIROpConversion(MLIRContext *context, LLVMTypeConverter &converter_,
                           TargetInfo &targetInfo_,
                           mlir::PatternBenefit benefit = 1)
      : mlir::OpConversionPattern<Op>::OpConversionPattern(context, benefit),
        dialect(converter_.getDialect()),
        typeConverter(converter_),
        targetInfo(targetInfo_) {}

 protected:
  LLVMTypeConverter &typeConverter;
  TargetInfo &targetInfo;
  LLVMDialect *dialect;

  LLVMType getUsizeType() const { return targetInfo.getUsizeType(); }
  LLVMType getI1Type() const { return targetInfo.getI1Type(); }
  LLVMType getI32Type() const { return LLVMType::getIntNTy(dialect, 32); }

  LLVMType getTupleType(ArrayRef<LLVMType> elementTypes) const {
    return targetInfo.makeTupleType(dialect, elementTypes);
  }

  Type getIntegerType(OpBuilder &builder) const {
    return builder.getIntegerType(targetInfo.pointerSizeInBits);
  }

  Attribute getIntegerAttr(OpBuilder &builder, int64_t i) const {
    return builder.getIntegerAttr(getIntegerType(builder), i);
  }

  Attribute getIntegerAttr(OpBuilder &builder, APInt &i) const {
    return builder.getIntegerAttr(getIntegerType(builder), i.getLimitedValue());
  }

  Attribute getI1Attr(OpBuilder &builder, int64_t i) const {
    return builder.getIntegerAttr(builder.getI1Type(), i);
  }

  Attribute getI32Attr(OpBuilder &builder, int64_t i) const {
    return builder.getIntegerAttr(builder.getIntegerType(32), i);
  }

  FlatSymbolRefAttr getOrInsertFunction(
      PatternRewriter &builder, ModuleOp mod, StringRef name, LLVMType retTy,
      ArrayRef<LLVMType> argTypes = {}) const {
    return createOrInsertFunction(builder, mod, dialect, targetInfo, name,
                                  retTy, argTypes);
  }

  Value getOrInsertGlobal(PatternRewriter &builder, ModuleOp mod,
                          StringRef name, LLVMType valueType,
                          LLVM::Linkage linkage = LLVM::Linkage::Internal,
                          LLVM::ThreadLocalMode tlsMode =
                              LLVM::ThreadLocalMode::NotThreadLocal) const {
    auto global = createOrInsertGlobal(builder, mod, dialect, targetInfo, name,
                                       valueType, linkage, tlsMode);
    auto addressOf = builder.create<LLVM::AddressOfOp>(global.getLoc(), global);
    return addressOf.getResult();
  }

  Value processAlloc(PatternRewriter &builder, edsc::ScopedContext &context,
                     ModuleOp parentModule, Location loc, LLVMType ty,
                     Value allocBytes) const {
    auto ptrTy = ty.getPointerTo();
    auto usizeTy = getUsizeType();
    auto callee = getOrInsertFunction(
        builder, parentModule, "__lumen_builtin_malloc", ptrTy, {usizeTy});
    auto call = builder.create<mlir::CallOp>(loc, callee, ArrayRef<Type>{ptrTy},
                                             ArrayRef<Value>{allocBytes});
    return call.getResult(0);
  }

  Value make_list(OpBuilder &builder, edsc::ScopedContext &context,
                  Value cons) const {
    return do_make_list(builder, context, typeConverter, targetInfo, cons);
  }

  Value make_box(OpBuilder &builder, edsc::ScopedContext &context,
                 Value val) const {
    return do_box(builder, context, typeConverter, targetInfo, val);
  }

  Value make_literal(OpBuilder &builder, edsc::ScopedContext &context,
                     Value val) const {
    return do_box_literal(builder, context, typeConverter, targetInfo, val);
  }

  Value unbox(OpBuilder &builder, edsc::ScopedContext &context,
              LLVMType innerTy, Value box) const {
    return do_unbox(builder, context, typeConverter, targetInfo, innerTy, box);
  }

  Value unbox_list(OpBuilder &builder, edsc::ScopedContext &context,
                   LLVMType innerTy, Value box) const {
    return do_unbox_list(builder, context, typeConverter, targetInfo, innerTy,
                         box);
  }

  Value mask_immediate(OpBuilder &builder, edsc::ScopedContext &context,
                       Value val) const {
    return do_mask_immediate(builder, context, targetInfo, val);
  }

  Value unmask_immediate(OpBuilder &builder, edsc::ScopedContext &context,
                         Value val) const {
    return do_unmask_immediate(builder, context, targetInfo, val);
  }
};

}  // namespace eir
}  // namespace lumen

#endif
