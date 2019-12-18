#ifndef LUMEN_LLVM_H
#define LUMEN_LLVM_H

// For now we just use the same support system as MLIR,
// but we may add more forward declarations as needed
#include "mlir/Support/LLVM.h"

#include "llvm-c/Types.h"

extern "C" bool LLVMEmitToFileDescriptor(LLVMModuleRef m,
#if defined(_WIN32)
                                         HANDLE handle,
#else
                                         int fd,
#endif
                                         char **errorMessage);

#endif
