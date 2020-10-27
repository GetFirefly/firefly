#ifndef LUMEN_TARGET_TARGETINFO_H
#define LUMEN_TARGET_TARGETINFO_H

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Triple.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/MLIRContext.h"

#include "lumen/term/Encoding.h"

using ::mlir::LLVM::LLVMType;

namespace llvm {
class TargetMachine;
}

namespace lumen {
enum class TargetDialect {
    Unknown,
    TargetNone,
    TargetEIR,
    TargetStandard,
    TargetLLVM,
};

struct TargetInfoImpl {
    TargetInfoImpl() {}
    TargetInfoImpl(const TargetInfoImpl &other)
        : triple(other.triple),
          encoding(other.encoding),
          voidTy(other.voidTy),
          pointerWidthIntTy(other.pointerWidthIntTy),
          i1Ty(other.i1Ty),
          i8Ty(other.i8Ty),
          i32Ty(other.i32Ty),
          i64Ty(other.i64Ty),
          bigIntTy(other.bigIntTy),
          floatTy(other.floatTy),
          doubleTy(other.doubleTy),
          binaryTy(other.binaryTy),
          binPushResultTy(other.binPushResultTy),
          matchResultTy(other.matchResultTy),
          recvContextTy(other.recvContextTy),
          consTy(other.consTy),
          opaqueFnTy(other.opaqueFnTy),
          uniqueTy(other.uniqueTy),
          defTy(other.defTy),
          exceptionTy(other.exceptionTy),
          erlangErrorTy(other.erlangErrorTy),
          nil(other.nil),
          none(other.none),
          listTag(other.listTag),
          listMask(other.listMask),
          boxTag(other.boxTag),
          literalTag(other.literalTag),
          immediateMask(other.immediateMask),
          headerMask(other.headerMask),
          immediateBits(other.immediateBits) {}

    std::string triple;

    Encoding encoding;

    LLVMType voidTy;
    LLVMType pointerWidthIntTy, i1Ty, i8Ty, i32Ty, i64Ty;
    LLVMType bigIntTy, floatTy, doubleTy;
    LLVMType binaryTy, binPushResultTy;
    LLVMType matchResultTy;
    LLVMType recvContextTy;
    LLVMType consTy;
    LLVMType opaqueFnTy;
    LLVMType uniqueTy, defTy;
    LLVMType exceptionTy, erlangErrorTy;

    llvm::APInt nil;
    llvm::APInt none;

    uint64_t listTag;
    uint64_t listMask;
    uint64_t boxTag;
    uint64_t literalTag;
    MaskInfo immediateMask;
    MaskInfo headerMask;
    uint8_t immediateBits;
};

class TargetInfo {
   public:
    explicit TargetInfo(llvm::TargetMachine *, mlir::MLIRContext *);
    explicit TargetInfo(const TargetInfo &other);

    bool is_x86_64() const {
        return archType == llvm::Triple::ArchType::x86_64;
    }
    bool is_wasm32() const {
        return archType == llvm::Triple::ArchType::wasm32;
    }
    bool requiresPackedFloats() const { return !is_x86_64(); }

    mlir::LLVM::LLVMType getConsType() { return impl->consTy; }
    mlir::LLVM::LLVMType getFloatType() { return impl->floatTy; }
    mlir::LLVM::LLVMType getDoubleType() { return impl->doubleTy; };
    mlir::LLVM::LLVMType getBinaryType() { return impl->binaryTy; }
    mlir::LLVM::LLVMType makeClosureType(unsigned size);
    mlir::LLVM::LLVMType makeTupleType(unsigned arity);
    mlir::LLVM::LLVMType makeTupleType(llvm::ArrayRef<mlir::LLVM::LLVMType>);

    mlir::LLVM::LLVMType getVoidType() { return impl->voidTy; };
    mlir::LLVM::LLVMType getUsizeType() { return impl->pointerWidthIntTy; }
    mlir::LLVM::LLVMType getI1Type() { return impl->i1Ty; }
    mlir::LLVM::LLVMType getI8Type() { return impl->i8Ty; }
    mlir::LLVM::LLVMType getI32Type() { return impl->i32Ty; }
    mlir::LLVM::LLVMType getI64Type() { return impl->i64Ty; }
    mlir::LLVM::LLVMType getOpaqueFnType() { return impl->opaqueFnTy; }

    mlir::LLVM::LLVMType getBinaryPushResultType() {
        return impl->binPushResultTy;
    }

    mlir::LLVM::LLVMType getMatchResultType() { return impl->matchResultTy; }

    mlir::LLVM::LLVMType getReceiveRefType() { return impl->recvContextTy; }

    mlir::LLVM::LLVMType getClosureUniqueType() { return impl->uniqueTy; }
    mlir::LLVM::LLVMType getClosureDefinitionType() { return impl->defTy; }

    mlir::LLVM::LLVMType getExceptionType() { return impl->exceptionTy; }
    mlir::LLVM::LLVMType getErlangErrorType() { return impl->erlangErrorTy; }

    uint8_t immediateBits() { return impl->immediateBits; }
    bool isValidImmediateValue(llvm::APInt &value) {
        return value.isIntN(impl->immediateBits);
    }
    bool isValidHeaderValue(llvm::APInt &value) {
        return impl->headerMask.maxAllowedValue >= value.getLimitedValue();
    }

    llvm::APInt encodeImmediate(uint32_t type, uint64_t value);
    llvm::APInt encodeHeader(uint32_t type, uint64_t arity);

    llvm::APInt &getNilValue() const;
    llvm::APInt &getNoneValue() const;

    uint64_t listTag() const;
    uint64_t listMask() const;
    uint64_t boxTag() const;
    uint64_t literalTag() const;
    uint32_t closureHeaderArity(uint32_t envLen) const;
    MaskInfo &immediateMask() const;
    MaskInfo &headerMask() const;

    unsigned pointerSizeInBits;

   private:
    llvm::Triple::ArchType archType;
    std::unique_ptr<TargetInfoImpl> impl;
};

}  // namespace lumen

#endif  // LUMEN_TARGET_TARGETINFO_H
