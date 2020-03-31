#include "lumen/llvm/ErrorHandling.h"

#include "llvm-c/Core.h"
#include "llvm-c/Object.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ArchiveWriter.h"
#include "llvm/Support/Memory.h"
#include "llvm/Support/Path.h"

using namespace llvm;
using namespace llvm::object;

enum class LLVMLumenResult { Success, Failure };

struct LumenArchiveMember {
  const char *Filename;
  const char *Name;
  Archive::Child Child;

  LumenArchiveMember()
      : Filename(nullptr), Name(nullptr), Child(nullptr, nullptr, nullptr) {}
  ~LumenArchiveMember() {}
};

struct LumenArchiveIterator {
  bool First;
  Archive::child_iterator Cur;
  Archive::child_iterator End;
  std::unique_ptr<Error> Err;

  LumenArchiveIterator(Archive::child_iterator Cur, Archive::child_iterator End,
                       std::unique_ptr<Error> Err)
      : First(true), Cur(Cur), End(End), Err(std::move(Err)) {}
};

enum class LLVMLumenArchiveKind {
  Other,
  GNU,
  BSD,
  COFF,
};

static Archive::Kind fromRust(LLVMLumenArchiveKind Kind) {
  switch (Kind) {
    case LLVMLumenArchiveKind::GNU:
      return Archive::K_GNU;
    case LLVMLumenArchiveKind::BSD:
      return Archive::K_BSD;
    case LLVMLumenArchiveKind::COFF:
      return Archive::K_COFF;
    default:
      report_fatal_error("Bad ArchiveKind.");
  }
}

typedef OwningBinary<Archive> *LLVMLumenArchiveRef;
typedef LumenArchiveMember *LLVMLumenArchiveMemberRef;
typedef Archive::Child *LLVMLumenArchiveChildRef;
typedef Archive::Child const *LLVMLumenArchiveChildConstRef;
typedef LumenArchiveIterator *LLVMLumenArchiveIteratorRef;

extern "C" LLVMLumenArchiveRef LLVMLumenOpenArchive(char *Path) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufOr =
      MemoryBuffer::getFile(Path, -1, false);
  if (!BufOr) {
    LLVMLumenSetLastError(BufOr.getError().message().c_str());
    return nullptr;
  }

  Expected<std::unique_ptr<Archive>> ArchiveOr =
      Archive::create(BufOr.get()->getMemBufferRef());

  if (!ArchiveOr) {
    LLVMLumenSetLastError(toString(ArchiveOr.takeError()).c_str());
    return nullptr;
  }

  OwningBinary<Archive> *Ret = new OwningBinary<Archive>(
      std::move(ArchiveOr.get()), std::move(BufOr.get()));

  return Ret;
}

extern "C" void LLVMLumenDestroyArchive(LLVMLumenArchiveRef archive) {
  delete archive;
}

extern "C" LLVMLumenArchiveIteratorRef LLVMLumenArchiveIteratorNew(
    LLVMLumenArchiveRef lumenArchive) {
  Archive *archive = lumenArchive->getBinary();
  auto err = std::make_unique<Error>(Error::success());
  auto cur = archive->child_begin(*err);
  if (*err) {
    LLVMLumenSetLastError(toString(std::move(*err)).c_str());
    return nullptr;
  }
  auto end = archive->child_end();
  return new LumenArchiveIterator(cur, end, std::move(err));
}

extern "C" LLVMLumenArchiveChildConstRef LLVMLumenArchiveIteratorNext(
    LLVMLumenArchiveIteratorRef RAI) {
  if (RAI->Cur == RAI->End) return nullptr;

  // Advancing the iterator validates the next child, and this can
  // uncover an error. LLVM requires that we check all Errors,
  // so we only advance the iterator if we actually need to fetch
  // the next child.
  // This means we must not advance the iterator in the *first* call,
  // but instead advance it *before* fetching the child in all later calls.
  if (!RAI->First) {
    ++RAI->Cur;
    if (*RAI->Err) {
      LLVMLumenSetLastError(toString(std::move(*RAI->Err)).c_str());
      return nullptr;
    }
  } else {
    RAI->First = false;
  }

  if (RAI->Cur == RAI->End) return nullptr;

  const Archive::Child &Child = *RAI->Cur.operator->();
  Archive::Child *Ret = new Archive::Child(Child);

  return Ret;
}

extern "C" void LLVMLumenArchiveChildFree(LLVMLumenArchiveChildRef Child) {
  delete Child;
}

extern "C" void LLVMLumenArchiveIteratorFree(LLVMLumenArchiveIteratorRef RAI) {
  delete RAI;
}

extern "C" const char *LLVMLumenArchiveChildName(
    LLVMLumenArchiveChildConstRef Child, size_t *Size) {
  Expected<StringRef> NameOrErr = Child->getName();
  if (!NameOrErr) {
    // rustc_codegen_llvm currently doesn't use this error string, but it might
    // be useful in the future, and in the mean time this tells LLVM that the
    // error was not ignored and that it shouldn't abort the process.
    LLVMLumenSetLastError(toString(NameOrErr.takeError()).c_str());
    return nullptr;
  }
  StringRef Name = NameOrErr.get();
  *Size = Name.size();
  return Name.data();
}

extern "C" const char *LLVMLumenArchiveChildData(LLVMLumenArchiveChildRef Child,
                                                 size_t *Size) {
  StringRef Buf;
  Expected<StringRef> BufOrErr = Child->getBuffer();
  if (!BufOrErr) {
    LLVMLumenSetLastError(toString(BufOrErr.takeError()).c_str());
    return nullptr;
  }
  Buf = BufOrErr.get();
  *Size = Buf.size();
  return Buf.data();
}

extern "C" LLVMLumenArchiveMemberRef LLVMLumenArchiveMemberNew(
    char *Filename, char *Name, LLVMLumenArchiveChildRef Child) {
  LumenArchiveMember *Member = new LumenArchiveMember;
  Member->Filename = Filename;
  Member->Name = Name;
  if (Child) Member->Child = *Child;
  return Member;
}

extern "C" void LLVMLumenArchiveMemberFree(LLVMLumenArchiveMemberRef Member) {
  delete Member;
}

extern "C" LLVMLumenResult LLVMLumenWriteArchive(
    char *Dst, size_t NumMembers, const LLVMLumenArchiveMemberRef *NewMembers,
    bool WriteSymbtab, LLVMLumenArchiveKind LumenKind) {
  std::vector<NewArchiveMember> Members;
  auto Kind = fromRust(LumenKind);

  for (size_t I = 0; I < NumMembers; I++) {
    auto Member = NewMembers[I];
    assert(Member->Name);
    if (Member->Filename) {
      Expected<NewArchiveMember> MOrErr =
          NewArchiveMember::getFile(Member->Filename, true);
      if (!MOrErr) {
        LLVMLumenSetLastError(toString(MOrErr.takeError()).c_str());
        return LLVMLumenResult::Failure;
      }
      MOrErr->MemberName = sys::path::filename(MOrErr->MemberName);
      Members.push_back(std::move(*MOrErr));
    } else {
      Expected<NewArchiveMember> MOrErr =
          NewArchiveMember::getOldMember(Member->Child, true);
      if (!MOrErr) {
        LLVMLumenSetLastError(toString(MOrErr.takeError()).c_str());
        return LLVMLumenResult::Failure;
      }
      Members.push_back(std::move(*MOrErr));
    }
  }

  auto Result = writeArchive(Dst, Members, WriteSymbtab, Kind, true, false);
  if (!Result) return LLVMLumenResult::Success;
  LLVMLumenSetLastError(toString(std::move(Result)).c_str());

  return LLVMLumenResult::Failure;
}
