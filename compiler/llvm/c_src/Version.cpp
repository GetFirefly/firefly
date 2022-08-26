#include "llvm/Support/CommandLine.h"

extern "C" uint32_t LLVMFireflyVersionMajor() { return LLVM_VERSION_MAJOR; }
extern "C" uint32_t LLVMFireflyVersionMinor() { return LLVM_VERSION_MINOR; }
