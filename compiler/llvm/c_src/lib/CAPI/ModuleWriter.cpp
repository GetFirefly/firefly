#include "firefly/llvm/ModuleWriter.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Module.h"



#if defined(_WIN32)
bool LLVMEmitToFileDescriptor(LLVMModuleRef m, HANDLE handle,
                                         char **errorMessage) {
  raw_win32_handle_ostream stream(handle, /*shouldClose=*/false,
                                  /*unbuffered=*/false);
#else
bool LLVMEmitToFileDescriptor(LLVMModuleRef m, int fd,
                                         char **errorMessage) {
  llvm::raw_fd_ostream stream(fd, /*shouldClose=*/false, /*unbuffered=*/false,
                              llvm::raw_ostream::OStreamKind::OK_FDStream);
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
bool LLVMEmitBitcodeToFileDescriptor(LLVMModuleRef m, HANDLE handle,
                                                char **errorMessage) {
  raw_win32_handle_ostream stream(handle, /*shouldClose=*/false,
                                  /*unbuffered=*/false);
#else
bool LLVMEmitBitcodeToFileDescriptor(LLVMModuleRef m, int fd,
                                                char **errorMessage) {
  llvm::raw_fd_ostream stream(fd, /*shouldClose=*/false, /*unbuffered=*/false,
                              llvm::raw_ostream::OStreamKind::OK_FDStream);
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
