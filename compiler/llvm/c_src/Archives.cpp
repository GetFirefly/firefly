#include "mlir-c/Support.h"
#include "mlir/CAPI/Support.h"
#include "llvm-c/Core.h"
#include "llvm-c/Object.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ArchiveWriter.h"
#include "llvm/Support/Memory.h"
#include "llvm/Support/Path.h"

using namespace llvm;
using namespace llvm::object;

struct LumenNewArchiveMember {
  StringRef filename;
  StringRef name;
  Archive::Child child;

  LumenNewArchiveMember()
      : filename(), name(), child(nullptr, nullptr, nullptr) {}
  ~LumenNewArchiveMember() {}
};

struct LumenArchiveIterator {
  bool first;
  Archive::child_iterator current;
  Archive::child_iterator end;
  std::unique_ptr<Error> err;

  LumenArchiveIterator(Archive::child_iterator current,
                       Archive::child_iterator end, std::unique_ptr<Error> err)
      : first(true), current(current), end(end), err(std::move(err)) {}
};

typedef OwningBinary<Archive> *LLVMLumenArchiveRef;
typedef LumenNewArchiveMember *LLVMLumenNewArchiveMemberRef;
typedef Archive::Child *LLVMLumenArchiveChildRef;
typedef Archive::Child const *LLVMLumenArchiveChildConstRef;
typedef LumenArchiveIterator *LLVMLumenArchiveIteratorRef;

extern "C" LLVMLumenArchiveRef LLVMLumenOpenArchive(MlirStringRef path,
                                                    char *error) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> bufferOr =
      MemoryBuffer::getFile(unwrap(path), -1, false);
  if (!bufferOr) {
    error = strdup(bufferOr.getError().message().c_str());
    return nullptr;
  }

  Expected<std::unique_ptr<Archive>> archiveOr =
      Archive::create(bufferOr.get()->getMemBufferRef());

  if (!archiveOr) {
    error = strdup(toString(archiveOr.takeError()).c_str());
    return nullptr;
  }

  OwningBinary<Archive> *archive = new OwningBinary<Archive>(
      std::move(archiveOr.get()), std::move(bufferOr.get()));

  error = nullptr;
  return archive;
}

extern "C" void LLVMLumenDestroyArchive(LLVMLumenArchiveRef archive) {
  delete archive;
}

extern "C" LLVMLumenArchiveIteratorRef
LLVMLumenArchiveIteratorNew(LLVMLumenArchiveRef lumenArchive, char *error) {
  Archive *archive = lumenArchive->getBinary();
  auto err = std::make_unique<Error>(Error::success());
  auto cur = archive->child_begin(*err);
  if (*err) {
    error = strdup(toString(std::move(*err)).c_str());
    return nullptr;
  }
  auto end = archive->child_end();
  return new LumenArchiveIterator(cur, end, std::move(err));
}

extern "C" LLVMLumenArchiveChildConstRef
LLVMLumenArchiveIteratorNext(LLVMLumenArchiveIteratorRef iter, char *error) {
  error = nullptr;
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
      error = strdup(toString(std::move(*iter->err)).c_str());
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

extern "C" void LLVMLumenArchiveChildFree(LLVMLumenArchiveChildRef child) {
  delete child;
}

extern "C" void LLVMLumenArchiveIteratorFree(LLVMLumenArchiveIteratorRef iter) {
  delete iter;
}

extern "C" MlirStringRef
LLVMLumenArchiveChildName(LLVMLumenArchiveChildConstRef child, char *error) {
  Expected<StringRef> nameOrErr = child->getName();
  if (!nameOrErr) {
    error = strdup(toString(nameOrErr.takeError()).c_str());
    return wrap(StringRef());
  }
  error = nullptr;
  return wrap(nameOrErr.get());
}

extern "C" MlirStringRef
LLVMLumenArchiveChildData(LLVMLumenArchiveChildRef child, char *error) {
  StringRef buf;
  Expected<StringRef> bufOrErr = child->getBuffer();
  if (!bufOrErr) {
    error = strdup(toString(bufOrErr.takeError()).c_str());
    return wrap(buf);
  }
  error = nullptr;
  buf = bufOrErr.get();
  return wrap(buf);
}

extern "C" LLVMLumenNewArchiveMemberRef
LLVMLumenNewArchiveMemberFromFile(MlirStringRef name, MlirStringRef filename) {
  LumenNewArchiveMember *member = new LumenNewArchiveMember;
  member->filename = unwrap(filename);
  member->name = unwrap(name);
  return member;
}

extern "C" LLVMLumenNewArchiveMemberRef
LLVMLumenNewArchiveMemberFromChild(MlirStringRef name,
                                   LLVMLumenArchiveChildRef child) {
  assert(child);
  LumenNewArchiveMember *member = new LumenNewArchiveMember;
  member->name = unwrap(name);
  member->child = *child;
  return member;
}

extern "C" void
LLVMLumenNewArchiveMemberFree(LLVMLumenNewArchiveMemberRef member) {
  delete member;
}

extern "C" bool
LLVMLumenWriteArchive(MlirStringRef filename, size_t numMembers,
                      const LLVMLumenNewArchiveMemberRef *newMembers,
                      bool writeSymbtab, Archive::Kind kind, char *error) {
  std::vector<NewArchiveMember> members;

  for (size_t i = 0; i < numMembers; i++) {
    auto member = newMembers[i];
    assert(!member->name.empty());
    if (member->filename.empty()) {
      Expected<NewArchiveMember> mOrErr =
          NewArchiveMember::getFile(member->filename, /*deterministic=*/true);
      if (!mOrErr) {
        error = strdup(toString(mOrErr.takeError()).c_str());
        return false;
      }
      mOrErr->MemberName = sys::path::filename(mOrErr->MemberName);
      members.push_back(std::move(*mOrErr));
    } else {
      Expected<NewArchiveMember> mOrErr =
          NewArchiveMember::getOldMember(member->child, /*deterministic=*/true);
      if (!mOrErr) {
        error = strdup(toString(mOrErr.takeError()).c_str());
        return false;
      }
      members.push_back(std::move(*mOrErr));
    }
  }

  error = nullptr;
  auto result = writeArchive(unwrap(filename), members, writeSymbtab, kind,
                             /*deterministic=*/true, /*thin=*/false);
  if (!result)
    return true;

  error = strdup(toString(std::move(result)).c_str());
  return false;
}
