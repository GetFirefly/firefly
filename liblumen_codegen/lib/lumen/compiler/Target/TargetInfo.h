#ifndef LUMEN_TARGET_TARGETINFO_H
#define LUMEN_TARGET_TARGETINFO_H

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Triple.h"
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

extern "C" struct Encoding {
  uint32_t pointerWidth;
  bool supportsNanboxing;
};

extern "C" struct MaskInfo {
  int32_t shift;
  uint64_t mask;

  bool requiresShift() const { return shift != 0; }
};

struct TargetInfoImpl {
  TargetInfoImpl() {}
  TargetInfoImpl(const TargetInfoImpl &other)
      : triple(other.triple),
        encoding(other.encoding),
        pointerWidthIntTy(other.pointerWidthIntTy),
        i1Ty(other.i1Ty),
        termTy(other.termTy),
        bigIntTy(other.bigIntTy),
        floatTy(other.floatTy),
        binaryTy(other.binaryTy),
        consTy(other.consTy),
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

  LLVMType pointerWidthIntTy, i1Ty;
  LLVMType termTy;
  LLVMType bigIntTy, floatTy;
  LLVMType binaryTy;
  LLVMType consTy;

  llvm::APInt nil;
  llvm::APInt none;

  uint64_t listTag;
  uint64_t listMask;
  uint64_t boxTag;
  uint64_t literalTag;
  MaskInfo immediateMask;
  MaskInfo headerMask;
};

class TargetInfo {
 public:
  explicit TargetInfo(llvm::TargetMachine *, mlir::LLVM::LLVMDialect &);
  explicit TargetInfo(const TargetInfo &other);

  bool is_x86_64() const { return archType == llvm::Triple::ArchType::x86_64; }
  bool is_wasm32() const { return archType == llvm::Triple::ArchType::wasm32; }
  bool requiresPackedFloats() const { return !is_x86_64(); }

  mlir::LLVM::LLVMType getTermType();
  mlir::LLVM::LLVMType getConsType();
  mlir::LLVM::LLVMType getFloatType();
  mlir::LLVM::LLVMType getBinaryType();
  mlir::LLVM::LLVMType makeTupleType(mlir::LLVM::LLVMDialect *,
                                     llvm::ArrayRef<mlir::LLVM::LLVMType>);

  mlir::LLVM::LLVMType getUsizeType();
  mlir::LLVM::LLVMType getI1Type();

  llvm::APInt encodeImmediate(uint32_t type, uint64_t value);
  llvm::APInt encodeHeader(uint32_t type, uint64_t arity);

  llvm::APInt &getNilValue() const;
  llvm::APInt &getNoneValue() const;

  uint64_t listTag() const;
  uint64_t listMask() const;
  uint64_t boxTag() const;
  uint64_t literalTag() const;
  MaskInfo &immediateMask() const;
  MaskInfo &headerMask() const;

  unsigned pointerSizeInBits;

 private:
  llvm::Triple::ArchType archType;
  std::unique_ptr<TargetInfoImpl> impl;
};

}  // namespace lumen

#endif  // LUMEN_TARGET_TARGETINFO_H
