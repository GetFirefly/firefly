#include "lumen/LLVM.h"
#include "lumen/Lumen.h"
#include "llvm/Bitcode/BitcodeWriter.h"

// On Windows we have a custom output stream type that
// can wrap the raw file handle we get from Rust
#if defined(_WIN32)
#include "lumen/Support/raw_win32_handle_ostream.h"
#endif

#include <cstdlib>

using namespace llvm;

#if defined(_WIN32)
bool LLVMEmitToFileDescriptor(LLVMModuleRef m, HANDLE handle,
                              char **errorMessage) {
  raw_win32_handle_ostream stream(handle, /*shouldClose=*/false,
                                  /*unbuffered=*/false);
#else
bool LLVMEmitToFileDescriptor(LLVMModuleRef m, int fd, char **errorMessage) {
  raw_fd_ostream stream(fd, /*shouldClose=*/false, /*unbuffered=*/false);
#endif
  Module *mod = unwrap(m);

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
bool LLVMEmitBitcodeToFileDescriptor(LLVMModuleRef m, HANDLE handle,
                              char **errorMessage) {
  raw_win32_handle_ostream stream(handle, /*shouldClose=*/false,
                                  /*unbuffered=*/false);
#else
bool LLVMEmitBitcodeToFileDescriptor(LLVMModuleRef m, int fd, char **errorMessage) {
  raw_fd_ostream stream(fd, /*shouldClose=*/false, /*unbuffered=*/false);
#endif
  Module *mod = unwrap(m);

  llvm::WriteBitcodeToFile(*mod, stream);

  if (stream.has_error()) {
    std::string err = "Error printing to file: " + stream.error().message();
    *errorMessage = strdup(err.c_str());
    return true;
  }

  stream.flush();

  return false;
}
