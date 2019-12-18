#include "lumen/LLVM.h"
#include "lumen/Lumen.h"

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
    std::string err = "Error print to file: " + stream.error().message();
    *errorMessage = strdup(err.c_str());
    return true;
  }

  stream.flush();

  return false;
}
