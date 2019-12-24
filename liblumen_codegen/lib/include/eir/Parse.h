#ifndef EIR_PARSE_H
#define EIR_PARSE_H

#include "eir/Context.h"
#include "lumen/LLVM.h"

extern "C" {
MLIRModuleRef MLIRParseFile(MLIRContextRef context, const char *filename);

MLIRModuleRef MLIRParseBuffer(MLIRContextRef context,
                              LLVMMemoryBufferRef buffer);
}

#endif // EIR_PARSE_H
