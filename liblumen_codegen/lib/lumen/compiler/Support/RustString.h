#ifndef LUMEN_RUSTSTRING_H
#define LUMEN_RUSTSTRING_H

#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/raw_ostream.h"

typedef struct OpaqueRustString *RustStringRef;
typedef struct LLVMOpaqueTwine *LLVMTwineRef;

class RawRustStringOstream : public llvm::raw_ostream {
  RustStringRef Str;
  uint64_t Pos;

  void write_impl(const char *Ptr, size_t Size) override;

  uint64_t current_pos() const override { return Pos; }

 public:
  explicit RawRustStringOstream(RustStringRef Str) : Str(Str), Pos(0) {}

  ~RawRustStringOstream() {
    // LLVM requires this.
    flush();
  }
};

#endif  // LUMEN_RUSTSTRING_H
