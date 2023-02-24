#include "firefly/llvm/Archives.h"

#include "mlir/CAPI/Support.h"
#include "llvm-c/Core.h"
#include "llvm-c/Object.h"
#include "llvm/Object/ArchiveWriter.h"
#include "llvm/Support/Memory.h"
#include "llvm/Support/Path.h"

using namespace llvm;
using namespace llvm::object;

LLVMFireflyArchiveRef LLVMFireflyOpenArchive(MlirStringRef path, char **error) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> bufferOr =
      MemoryBuffer::getFile(unwrap(path), -1, false);
  if (!bufferOr) {
    *error = strdup(bufferOr.getError().message().c_str());
    return nullptr;
  }

  Expected<std::unique_ptr<Archive>> archiveOr =
      Archive::create(bufferOr.get()->getMemBufferRef());

  if (!archiveOr) {
    *error = strdup(toString(archiveOr.takeError()).c_str());
    return nullptr;
  }

  OwningBinary<Archive> *archive = new OwningBinary<Archive>(
      std::move(archiveOr.get()), std::move(bufferOr.get()));

  *error = nullptr;
  return archive;
}

void LLVMFireflyDestroyArchive(LLVMFireflyArchiveRef archive) {
  delete archive;
}

LLVMFireflyArchiveIteratorRef
LLVMFireflyArchiveIteratorNew(LLVMFireflyArchiveRef fireflyArchive,
                              char **error) {
  Archive *archive = fireflyArchive->getBinary();
  std::unique_ptr<Error> err = std::make_unique<Error>(Error::success());
  auto cur = archive->child_begin(*err);
  if (*err) {
    *error = strdup(toString(std::move(*err)).c_str());
    return nullptr;
  }
  *error = nullptr;
  auto end = archive->child_end();
  return new FireflyArchiveIterator(cur, end, std::move(err));
}

LLVMFireflyArchiveChildConstRef
LLVMFireflyArchiveIteratorNext(LLVMFireflyArchiveIteratorRef iter,
                               char **error) {
  *error = nullptr;
  if (iter->current == iter->end)
    return nullptr;

  // Advancing the iterator validates the next child, and this can
  // uncover an error. LLVM requires that we check all Errors,
  // so we only advance the iterator if we actually need to fetch
  // the next child.
  // This means we must not advance the iterator in the *first* call,
  // but instead advance it *before* fetching the child in all later calls.
  if (!iter->first) {
    ++iter->current;
    if (*iter->err) {
      auto errString = toString(std::move(*iter->err));
      *error = strdup(errString.c_str());
      return nullptr;
    }
  } else {
    iter->first = false;
  }

  if (iter->current == iter->end)
    return nullptr;

  const Archive::Child &child = *iter->current.operator->();
  return new Archive::Child(child);
}

void LLVMFireflyArchiveChildFree(LLVMFireflyArchiveChildRef child) {
  delete child;
}

void LLVMFireflyArchiveIteratorFree(LLVMFireflyArchiveIteratorRef iter) {
  delete iter;
}

MlirStringRef LLVMFireflyArchiveChildName(LLVMFireflyArchiveChildConstRef child,
                                          char **error) {
  Expected<StringRef> nameOrErr = child->getName();
  if (!nameOrErr) {
    auto errString = toString(nameOrErr.takeError());
    *error = strdup(errString.c_str());
    return wrap(StringRef());
  }
  *error = nullptr;
  return wrap(nameOrErr.get());
}

MlirStringRef LLVMFireflyArchiveChildData(LLVMFireflyArchiveChildRef child,
                                          char **error) {
  Expected<StringRef> bufOrErr = child->getBuffer();
  if (!bufOrErr) {
    auto errString = toString(bufOrErr.takeError());
    *error = strdup(errString.c_str());
    return wrap(StringRef());
  }
  *error = nullptr;
  return wrap(bufOrErr.get());
}

LLVMFireflyNewArchiveMemberRef
LLVMFireflyNewArchiveMemberFromFile(MlirStringRef name,
                                    MlirStringRef filename) {
  FireflyNewArchiveMember *member = new FireflyNewArchiveMember;
  member->filename = unwrap(filename);
  member->name = unwrap(name);
  return member;
}

LLVMFireflyNewArchiveMemberRef
LLVMFireflyNewArchiveMemberFromChild(MlirStringRef name,
                                     LLVMFireflyArchiveChildRef child) {
  assert(child);
  FireflyNewArchiveMember *member = new FireflyNewArchiveMember;
  member->name = unwrap(name);
  member->child = *child;
  return member;
}

void LLVMFireflyNewArchiveMemberFree(LLVMFireflyNewArchiveMemberRef member) {
  delete member;
}

bool LLVMFireflyWriteArchive(MlirStringRef filename, size_t numMembers,
                             const LLVMFireflyNewArchiveMemberRef *newMembers,
                             bool writeSymbtab, Archive::Kind kind,
                             char **error) {
  std::vector<NewArchiveMember> members;

  for (size_t i = 0; i < numMembers; i++) {
    auto member = newMembers[i];
    assert(!member->name.empty());
    if (!member->filename.empty()) {
      Expected<NewArchiveMember> mOrErr =
          NewArchiveMember::getFile(member->filename, /*deterministic=*/true);
      if (!mOrErr) {
        *error = strdup(toString(mOrErr.takeError()).c_str());
        return false;
      }
      mOrErr->MemberName = sys::path::filename(mOrErr->MemberName);
      members.push_back(std::move(*mOrErr));
    } else {
      Expected<NewArchiveMember> mOrErr =
          NewArchiveMember::getOldMember(member->child, /*deterministic=*/true);
      if (!mOrErr) {
        *error = strdup(toString(mOrErr.takeError()).c_str());
        return false;
      }
      members.push_back(std::move(*mOrErr));
    }
  }

  *error = nullptr;
  auto result = writeArchive(unwrap(filename), members, writeSymbtab, kind,
                             /*deterministic=*/true, /*thin=*/false);
  if (!result)
    return true;

  *error = strdup(toString(std::move(result)).c_str());
  return false;
}
