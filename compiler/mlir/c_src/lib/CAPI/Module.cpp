#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"

#include "llvm-c/Core.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Module.h"

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

#include "CIR-c/Module.h"

// On Windows we have a custom output stream type that
// can wrap the raw file handle we get from Rust
#if defined(_WIN32)
#include "firefly/llvm/raw_win32_handle_ostream.h"
#endif

#include <cstdlib>

using namespace mlir;

using ::llvm::MemoryBuffer;
using ::llvm::SourceMgr;
using ::llvm::StringRef;

MlirStringRef mlirModuleGetName(MlirModule module) {
  auto name = unwrap(module).getName();
  return wrap(*name);
}

MlirModule mlirModuleClone(MlirModule module) {
  return MlirModule{unwrap(module).getOperation()->clone()};
}

bool mlirOperationIsAModule(MlirOperation op) {
  return isa<ModuleOp>(unwrap(op));
}

MlirModule mlirParseFile(MlirContext context, MlirStringRef path) {
  // Parse the input mlir.
  OwningOpRef<ModuleOp> owned =
      mlir::parseSourceFile<ModuleOp>(unwrap(path), unwrap(context));
  if (!owned) {
    return MlirModule{nullptr};
  }

  return MlirModule{owned.release().getOperation()};
}

#if defined(_WIN32)
bool LLVMEmitToFileDescriptor(LLVMModuleRef m, HANDLE handle,
                              char **errorMessage) {
  raw_win32_handle_ostream stream(handle, /*shouldClose=*/false,
                                  /*unbuffered=*/false);
#else
bool LLVMEmitToFileDescriptor(LLVMModuleRef m, int fd, char **errorMessage) {
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

#if defined(_WIN32)
bool MLIREmitToFileDescriptor(MlirModule m, HANDLE handle,
                              char **errorMessage) {
  llvm::raw_win32_handle_ostream stream(handle, /*shouldClose=*/false,
                                        /*unbuffered=*/false);
#else
bool MLIREmitToFileDescriptor(MlirModule m, int fd, char **errorMessage) {
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

LLVMMemoryBufferRef MLIREmitToMemoryBuffer(MlirModule m) {
  llvm::SmallString<0> codeString;
  llvm::raw_svector_ostream oStream(codeString);
  unwrap(m).print(oStream);
  llvm::StringRef data = oStream.str();
  return LLVMCreateMemoryBufferWithMemoryRangeCopy(data.data(), data.size(),
                                                   "");
}
