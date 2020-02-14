#ifndef LUMEN_TARGET_TARGETINFO_H
#define LUMEN_TARGET_TARGETINFO_H

#include "llvm/ADT/Triple.h"
#include "llvm/ADT/APInt.h"

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

class TargetInfo {
public:
  TargetInfo() : archType(llvm::Triple::ArchType::UnknownArch), pointerSizeInBits(0) {};
  TargetInfo(llvm::TargetMachine *);
  TargetInfo(const TargetInfo &other) : archType(other.archType), pointerSizeInBits(other.pointerSizeInBits) {};
    
  bool is_x86_64() const { return archType == llvm::Triple::ArchType::x86_64; }
  bool is_wasm32() const { return archType == llvm::Triple::ArchType::wasm32; }

  //llvm::APInt getNilValue() const { return llvm::APInt(pointerSizeInBits, TERM_ENCODING_NIL_VALUE__x86_64, false); }
  //llvm::APInt getNoneValue() const { return llvm::APInt(pointerSizeInBits, TERM_ENCODING_NONE_VALUE__x86_64, false); }

  llvm::Triple::ArchType archType;
  unsigned pointerSizeInBits;
};

} // namespace lumen

#endif // LUMEN_TARGET_TARGETINFO_H
