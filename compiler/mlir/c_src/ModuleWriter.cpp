#include "lumen/mlir/MLIR.h"
#include "lumen/llvm/MemoryBuffer.h"

// On Windows we have a custom output stream type that
// can wrap the raw file handle we get from Rust
#if defined(_WIN32)
#include "lumen/llvm/raw_win32_handle_ostream.h"
#endif

#include "llvm-c/Core.h"
#include "llvm/IR/Module.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Bitcode/BitcodeWriter.h"

#include "mlir/IR/Module.h"

#include <cstdlib>

#if defined(_WIN32)
extern "C" bool LLVMEmitToFileDescriptor(LLVMModuleRef m, HANDLE handle,
                                         char **errorMessage) {
  raw_win32_handle_ostream stream(handle, /*shouldClose=*/false,
                                  /*unbuffered=*/false);
#else
extern "C" bool LLVMEmitToFileDescriptor(LLVMModuleRef m, int fd,
                                         char **errorMessage) {
  llvm::raw_fd_ostream stream(fd, /*shouldClose=*/false, /*unbuffered=*/false);
#endif
  llvm::Module *mod = llvm::unwrap(m);

  mod->print(stream, nullptr);

  if (stream.has_error()) {
    std::string err = "Error printing to file: " + stream.error().message();
    *errorMessage = strdup(err.c_str());
    return true;
  }

  stream.flush();

  return false;
}

#if defined(_WIN32)
extern "C" bool LLVMEmitBitcodeToFileDescriptor(LLVMModuleRef m, HANDLE handle,
                                                char **errorMessage) {
  raw_win32_handle_ostream stream(handle, /*shouldClose=*/false,
                                  /*unbuffered=*/false);
#else
extern "C" bool LLVMEmitBitcodeToFileDescriptor(LLVMModuleRef m, int fd,
                                                char **errorMessage) {
  llvm::raw_fd_ostream stream(fd, /*shouldClose=*/false, /*unbuffered=*/false);
#endif
  llvm::Module *mod = llvm::unwrap(m);

  llvm::WriteBitcodeToFile(*mod, stream);

  if (stream.has_error()) {
    std::string err = "Error printing to file: " + stream.error().message();
    *errorMessage = strdup(err.c_str());
    return true;
  }

  stream.flush();

  return false;
}

#if defined(_WIN32)
extern "C" bool MLIREmitToFileDescriptor(MLIRModuleRef m, HANDLE handle,
                                         char **errorMessage) {
  llvm::raw_win32_handle_ostream stream(handle, /*shouldClose=*/false,
                                        /*unbuffered=*/false);
#else
extern "C" bool MLIREmitToFileDescriptor(MLIRModuleRef m, int fd,
                                         char **errorMessage) {
  llvm::raw_fd_ostream stream(fd, /*shouldClose=*/false, /*unbuffered=*/false);
#endif
  mlir::ModuleOp *mod = unwrap(m);
  mod->print(stream);
  if (stream.has_error()) {
    std::error_code error = stream.error();
    *errorMessage = strdup(error.message().c_str());
    return true;
  }
  stream.flush();
  return false;
}

extern "C" LLVMMemoryBufferRef MLIREmitToMemoryBuffer(MLIRModuleRef m) {
  mlir::ModuleOp *mod = unwrap(m);
  llvm::SmallString<0> codeString;
  llvm::raw_svector_ostream oStream(codeString);
  mod->print(oStream);
  llvm::StringRef data = oStream.str();
  return LLVMCreateMemoryBufferWithMemoryRangeCopy(data.data(), data.size(),
                                                   "");
}
