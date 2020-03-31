#include "llvm/Support/CommandLine.h"

extern "C" uint32_t LLVMLumenVersionMajor() { return LLVM_VERSION_MAJOR; }
extern "C" uint32_t LLVMLumenVersionMinor() { return LLVM_VERSION_MINOR; }
