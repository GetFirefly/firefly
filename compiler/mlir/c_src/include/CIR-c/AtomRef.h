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

struct AtomRef {
  size_t symbol;
  const char *data;
  size_t len;

#ifdef __cplusplus
  llvm::StringRef strref() const;
#endif
};

#ifdef __cplusplus
} // extern "C"
} // namespace mlir
} // namespace cir
#endif
