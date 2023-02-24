#include "firefly/llvm/MemoryBuffer.h"

// On Windows we have a custom output stream type that
// can wrap the raw file handle we get from Rust
#if defined(_WIN32)
#include "firefly/llvm/raw_win32_handle_ostream.h"
#endif

#include "llvm-c/Core.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Module.h"

#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"
#include "mlir/IR/BuiltinOps.h"

#include <cstdlib>

#if defined(_WIN32)
extern "C" bool MLIREmitToFileDescriptor(MlirModule m, HANDLE handle,
                                         char **errorMessage) {
  llvm::raw_win32_handle_ostream stream(handle, /*shouldClose=*/false,
                                        /*unbuffered=*/false);
#else
extern "C" bool MLIREmitToFileDescriptor(MlirModule m, int fd,
                                         char **errorMessage) {
  llvm::raw_fd_ostream stream(fd, /*shouldClose=*/false, /*unbuffered=*/false,
                              llvm::raw_ostream::OStreamKind::OK_FDStream);
#endif
  unwrap(m).print(stream);
  if (stream.has_error()) {
    std::error_code error = stream.error();
    *errorMessage = strdup(error.message().c_str());
    return true;
  }
  stream.flush();
  return false;
}

extern "C" LLVMMemoryBufferRef MLIREmitToMemoryBuffer(MlirModule m) {
  llvm::SmallString<0> codeString;
  llvm::raw_svector_ostream oStream(codeString);
  unwrap(m).print(oStream);
  llvm::StringRef data = oStream.str();
  return LLVMCreateMemoryBufferWithMemoryRangeCopy(data.data(), data.size(),
                                                   "");
}
