#ifndef LUMEN_SUPPORT_MEMORY_H
#define LUMEN_SUPPORT_MEMORY_H

#include "llvm-c/Types.h"
#include "llvm/Support/CBindingWrapping.h"
#include "llvm/Support/MemoryBuffer.h"

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(llvm::MemoryBuffer, LLVMMemoryBufferRef);

#endif
