#pragma once

#include "mlir-c/Support.h"

#include "llvm/Object/Archive.h"

#include <memory>

struct FireflyNewArchiveMember {
  llvm::StringRef filename;
  llvm::StringRef name;
  llvm::object::Archive::Child child;

  FireflyNewArchiveMember()
      : filename(), name(), child(nullptr, nullptr, nullptr) {}
  ~FireflyNewArchiveMember() {}
};

struct FireflyArchiveIterator {
  bool first;
  llvm::object::Archive::child_iterator current;
  llvm::object::Archive::child_iterator end;
  std::unique_ptr<llvm::Error> err;

  FireflyArchiveIterator(llvm::object::Archive::child_iterator current,
                         llvm::object::Archive::child_iterator end,
                         std::unique_ptr<llvm::Error> err)
      : first(true), current(current), end(end), err(std::move(err)) {}
};

typedef llvm::object::OwningBinary<llvm::object::Archive>
    *LLVMFireflyArchiveRef;
typedef FireflyNewArchiveMember *LLVMFireflyNewArchiveMemberRef;
typedef llvm::object::Archive::Child *LLVMFireflyArchiveChildRef;
typedef llvm::object::Archive::Child const *LLVMFireflyArchiveChildConstRef;
typedef FireflyArchiveIterator *LLVMFireflyArchiveIteratorRef;

extern "C" {
MLIR_CAPI_EXPORTED LLVMFireflyArchiveRef
LLVMFireflyOpenArchive(MlirStringRef path, char **error);

MLIR_CAPI_EXPORTED void
LLVMFireflyDestroyArchive(LLVMFireflyArchiveRef archive);

MLIR_CAPI_EXPORTED LLVMFireflyArchiveIteratorRef LLVMFireflyArchiveIteratorNew(
    LLVMFireflyArchiveRef fireflyArchive, char **error);

MLIR_CAPI_EXPORTED LLVMFireflyArchiveChildConstRef
LLVMFireflyArchiveIteratorNext(LLVMFireflyArchiveIteratorRef iter,
                               char **error);

MLIR_CAPI_EXPORTED void
LLVMFireflyArchiveChildFree(LLVMFireflyArchiveChildRef child);

MLIR_CAPI_EXPORTED void
LLVMFireflyArchiveIteratorFree(LLVMFireflyArchiveIteratorRef iter);

MLIR_CAPI_EXPORTED MlirStringRef LLVMFireflyArchiveChildName(
    LLVMFireflyArchiveChildConstRef child, char **error);

MLIR_CAPI_EXPORTED MlirStringRef
LLVMFireflyArchiveChildData(LLVMFireflyArchiveChildRef child, char **error);

MLIR_CAPI_EXPORTED LLVMFireflyNewArchiveMemberRef
LLVMFireflyNewArchiveMemberFromFile(MlirStringRef name, MlirStringRef filename);

MLIR_CAPI_EXPORTED LLVMFireflyNewArchiveMemberRef
LLVMFireflyNewArchiveMemberFromChild(MlirStringRef name,
                                     LLVMFireflyArchiveChildRef child);

MLIR_CAPI_EXPORTED void
LLVMFireflyNewArchiveMemberFree(LLVMFireflyNewArchiveMemberRef member);

MLIR_CAPI_EXPORTED bool
LLVMFireflyWriteArchive(MlirStringRef filename, size_t numMembers,
                        const LLVMFireflyNewArchiveMemberRef *newMembers,
                        bool writeSymbtab, llvm::object::Archive::Kind kind,
                        char **error);
}
