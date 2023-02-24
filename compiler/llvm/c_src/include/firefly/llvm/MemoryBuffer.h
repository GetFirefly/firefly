#pragma once

#include "firefly/llvm/CAPI.h"

namespace llvm {
class MemoryBuffer;
}

extern "C" {
DEFINE_C_API_STRUCT(LLVMMemoryBufferRef, void);
}

DEFINE_C_API_PTR_METHODS(LLVMMemoryBufferRef, llvm::MemoryBuffer);
