#include "llvm/IR/Attributes.h"
#include "llvm-c/Core.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/ErrorHandling.h"

using ::llvm::AttrBuilder;
using ::llvm::Attribute;
using ::llvm::StringRef;
using ::llvm::unwrap;

namespace firefly {
namespace Attribute {
enum AttrKind {
  AlwaysInline = 0,
  ByVal = 1,
  Cold = 2,
  InlineHint = 3,
  MinSize = 4,
  Naked = 5,
  NoAlias = 6,
  NoCapture = 7,
  NoInline = 8,
  NonNull = 9,
  NoRedZone = 10,
  NoReturn = 11,
  NoUnwind = 12,
  OptimizeForSize = 13,
  ReadOnly = 14,
  ArgMemOnly = 15,
  SExt = 16,
  StructRet = 17,
  UWTable = 18,
  ZExt = 19,
  InReg = 20,
  SanitizeThread = 21,
  SanitizeAddress = 22,
  SanitizeMemory = 23,
  NonLazyBind = 24,
  OptimizeNone = 25,
  ReturnsTwice = 26,
};
}
} // namespace firefly

static Attribute::AttrKind fromRust(firefly::Attribute::AttrKind kind) {
  switch (kind) {
  case firefly::Attribute::AlwaysInline:
    return Attribute::AlwaysInline;
  case firefly::Attribute::ByVal:
    return Attribute::ByVal;
  case firefly::Attribute::Cold:
    return Attribute::Cold;
  case firefly::Attribute::InlineHint:
    return Attribute::InlineHint;
  case firefly::Attribute::MinSize:
    return Attribute::MinSize;
  case firefly::Attribute::Naked:
    return Attribute::Naked;
  case firefly::Attribute::NoAlias:
    return Attribute::NoAlias;
  case firefly::Attribute::NoCapture:
    return Attribute::NoCapture;
  case firefly::Attribute::NoInline:
    return Attribute::NoInline;
  case firefly::Attribute::NonNull:
    return Attribute::NonNull;
  case firefly::Attribute::NoRedZone:
    return Attribute::NoRedZone;
  case firefly::Attribute::NoReturn:
    return Attribute::NoReturn;
  case firefly::Attribute::NoUnwind:
    return Attribute::NoUnwind;
  case firefly::Attribute::OptimizeForSize:
    return Attribute::OptimizeForSize;
  case firefly::Attribute::ReadOnly:
    return Attribute::ReadOnly;
  case firefly::Attribute::ArgMemOnly:
    return Attribute::ArgMemOnly;
  case firefly::Attribute::SExt:
    return Attribute::SExt;
  case firefly::Attribute::StructRet:
    return Attribute::StructRet;
  case firefly::Attribute::UWTable:
    return Attribute::UWTable;
  case firefly::Attribute::ZExt:
    return Attribute::ZExt;
  case firefly::Attribute::InReg:
    return Attribute::InReg;
  case firefly::Attribute::SanitizeThread:
    return Attribute::SanitizeThread;
  case firefly::Attribute::SanitizeAddress:
    return Attribute::SanitizeAddress;
  case firefly::Attribute::SanitizeMemory:
    return Attribute::SanitizeMemory;
  case firefly::Attribute::NonLazyBind:
    return Attribute::NonLazyBind;
  case firefly::Attribute::OptimizeNone:
    return Attribute::OptimizeNone;
  case firefly::Attribute::ReturnsTwice:
    return Attribute::ReturnsTwice;
  }
  llvm::report_fatal_error("invalid attribute kind");
}

extern "C" void LLVMFireflyAddAlignmentCallSiteAttr(LLVMValueRef instr,
                                                  unsigned index,
                                                  uint32_t bytes) {
  auto call = unwrap<llvm::CallBase>(instr);
  AttrBuilder b(call->getContext());
  b.addAlignmentAttr(bytes);
  call->setAttributes(
      call->getAttributes().addAttributesAtIndex(call->getContext(), index, b));
}

extern "C" void LLVMFireflyAddDereferenceableCallSiteAttr(LLVMValueRef instr,
                                                        unsigned index,
                                                        uint64_t bytes) {
  auto call = unwrap<llvm::CallBase>(instr);
  AttrBuilder b(call->getContext());
  b.addDereferenceableAttr(bytes);
  call->setAttributes(
      call->getAttributes().addAttributesAtIndex(call->getContext(), index, b));
}

extern "C" void
LLVMFireflyAddDereferenceableOrNullCallSiteAttr(LLVMValueRef instr,
                                              unsigned index, uint64_t bytes) {
  auto call = unwrap<llvm::CallBase>(instr);
  AttrBuilder b(call->getContext());
  b.addDereferenceableOrNullAttr(bytes);
  call->setAttributes(
      call->getAttributes().addAttributesAtIndex(call->getContext(), index, b));
}

extern "C" void LLVMFireflyAddByValCallSiteAttr(LLVMValueRef instr,
                                              unsigned index, LLVMTypeRef ty) {
  auto call = unwrap<llvm::CallBase>(instr);
  Attribute attr = Attribute::getWithByValType(call->getContext(), unwrap(ty));
  call->addAttributeAtIndex(index, attr);
}

extern "C" void LLVMFireflyAddAlignmentAttr(LLVMValueRef fn, unsigned index,
                                          uint32_t bytes) {
  auto *f = unwrap<llvm::Function>(fn);
  AttrBuilder b(f->getContext());
  b.addAlignmentAttr(bytes);
  f->addParamAttrs(index, b);
}

extern "C" void LLVMFireflyAddDereferenceableAttr(LLVMValueRef fn, unsigned index,
                                                uint64_t bytes) {
  auto *f = unwrap<llvm::Function>(fn);
  AttrBuilder b(f->getContext());
  b.addDereferenceableAttr(bytes);
  f->addParamAttrs(index, b);
}

extern "C" void LLVMFireflyAddDereferenceableOrNullAttr(LLVMValueRef fn,
                                                      unsigned index,
                                                      uint64_t bytes) {
  auto *f = unwrap<llvm::Function>(fn);
  AttrBuilder b(f->getContext());
  b.addDereferenceableOrNullAttr(bytes);
  f->addParamAttrs(index, b);
}

extern "C" void LLVMFireflyAddByValAttr(LLVMValueRef fn, unsigned index,
                                      LLVMTypeRef ty) {
  auto *f = unwrap<llvm::Function>(fn);
  Attribute attr = Attribute::getWithByValType(f->getContext(), unwrap(ty));
  f->addParamAttr(index, attr);
}
