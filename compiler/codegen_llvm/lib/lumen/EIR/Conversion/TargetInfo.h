#ifndef LUMEN_TARGET_TARGETINFO_H
#define LUMEN_TARGET_TARGETINFO_H

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Triple.h"
#include "lumen/term/Encoding.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

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
        pointerWidthIntTy(other.pointerWidthIntTy),
        i1Ty(other.i1Ty),
        i8Ty(other.i8Ty),
        i32Ty(other.i32Ty),
        bigIntTy(other.bigIntTy),
        floatTy(other.floatTy),
        binaryTy(other.binaryTy),
        binPushResultTy(other.binPushResultTy),
        consTy(other.consTy),
        opaqueFnTy(other.opaqueFnTy),
        uniqueTy(other.uniqueTy),
        defTy(other.defTy),
        recvContextTy(other.recvContextTy),
        nil(other.nil),
        none(other.none),
        listTag(other.listTag),
        listMask(other.listMask),
        boxTag(other.boxTag),
        literalTag(other.literalTag),
        immediateMask(other.immediateMask),
        headerMask(other.headerMask) {}

  std::string triple;

  Encoding encoding;

  LLVMType pointerWidthIntTy, i1Ty, i8Ty, i32Ty;
  LLVMType bigIntTy, floatTy;
  LLVMType binaryTy, binPushResultTy;
  LLVMType matchResultTy;
  LLVMType recvContextTy;
  LLVMType consTy;
  LLVMType opaqueFnTy;
  LLVMType uniqueTy, defTy;
  LLVMType exceptionTy;

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
  explicit TargetInfo(llvm::TargetMachine *, mlir::LLVM::LLVMDialect &);
  explicit TargetInfo(const TargetInfo &other);

  bool is_x86_64() const { return archType == llvm::Triple::ArchType::x86_64; }
  bool is_wasm32() const { return archType == llvm::Triple::ArchType::wasm32; }
  bool requiresPackedFloats() const { return !is_x86_64(); }

  mlir::LLVM::LLVMType getConsType();
  mlir::LLVM::LLVMType getFloatType();
  mlir::LLVM::LLVMType getBinaryType();
  mlir::LLVM::LLVMType makeClosureType(mlir::LLVM::LLVMDialect *,
                                       unsigned size);
  mlir::LLVM::LLVMType makeTupleType(mlir::LLVM::LLVMDialect *, unsigned arity);
  mlir::LLVM::LLVMType makeTupleType(mlir::LLVM::LLVMDialect *,
                                     llvm::ArrayRef<mlir::LLVM::LLVMType>);

  mlir::LLVM::LLVMType getUsizeType();
  mlir::LLVM::LLVMType getI1Type();
  mlir::LLVM::LLVMType getI8Type();
  mlir::LLVM::LLVMType getI32Type();
  mlir::LLVM::LLVMType getOpaqueFnType();

  mlir::LLVM::LLVMType getBinaryPushResultType() {
    return impl->binPushResultTy;
  }

  mlir::LLVM::LLVMType getMatchResultType() { return impl->matchResultTy; }

  mlir::LLVM::LLVMType getReceiveRefType() { return impl->recvContextTy; }

  mlir::LLVM::LLVMType getClosureUniqueType() { return impl->uniqueTy; }
  mlir::LLVM::LLVMType getClosureDefinitionType() { return impl->defTy; }

  mlir::LLVM::LLVMType getExceptionType() { return impl->exceptionTy; }

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
