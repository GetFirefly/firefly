#include "llvm-c/Core.h"
// On Windows we have a custom output stream type that
// can wrap the raw file handle we get from Rust
#if defined(_WIN32)
#include "firefly/llvm/raw_win32_handle_ostream.h"
#endif
#include "firefly/llvm/CAPI.h"

#include <cstdlib>

extern "C" {
#if defined(_WIN32)
MLIR_CAPI_EXPORTED bool LLVMEmitToFileDescriptor(LLVMModuleRef m, HANDLE handle,
                                         char **errorMessage);
#else
MLIR_CAPI_EXPORTED bool LLVMEmitToFileDescriptor(LLVMModuleRef m, int fd,
                                         char **errorMessage);
#endif

#if defined(_WIN32)
MLIR_CAPI_EXPORTED bool LLVMEmitBitcodeToFileDescriptor(LLVMModuleRef m, HANDLE handle,
                                                char **errorMessage);
#else
MLIR_CAPI_EXPORTED bool LLVMEmitBitcodeToFileDescriptor(LLVMModuleRef m, int fd,
                                                char **errorMessage);
#endif
} // extern "C"
