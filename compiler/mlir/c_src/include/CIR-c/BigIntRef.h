#pragma once

#include <cstdlib>

#ifdef __cplusplus
#include <llvm/ADT/ArrayRef.h>

namespace mlir {
namespace cir {
extern "C" {
#endif

enum Sign { SignMinus = 0, SignNoSign, SignPlus };

struct BigIntRef {
  Sign sign;
  const int32_t *digits;
  size_t len;

#ifdef __cplusplus
  llvm::ArrayRef<int32_t> data() const;
#endif
};

#ifdef __cplusplus
} // extern "C"
} // namespace mlir
} // namespace cir
#endif
