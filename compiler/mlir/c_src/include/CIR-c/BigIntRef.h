#pragma once

#include <cstdlib>

#ifdef __cplusplus
namespace llvm {
class StringRef;
}

namespace mlir {
namespace cir {
extern "C" {
#endif

enum Sign { SignMinus = 0, SignNoSign, SignPlus };

struct BigIntRef {
  Sign sign;
  const char *digits;
  size_t len;

#ifdef __cplusplus
  llvm::StringRef data() const;
#endif
};

#ifdef __cplusplus
} // extern "C"
} // namespace mlir
} // namespace cir
#endif
