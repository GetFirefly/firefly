#include "llvm-c/Core.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/ErrorHandling.h"

using ::llvm::Attribute;
using ::llvm::AttrBuilder;
using ::llvm::unwrap;

namespace lumen {
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
}


static Attribute::AttrKind fromRust(lumen::Attribute::AttrKind kind) {
  switch (kind) {
  case lumen::Attribute::AlwaysInline:
    return Attribute::AlwaysInline;
  case lumen::Attribute::ByVal:
    return Attribute::ByVal;
  case lumen::Attribute::Cold:
    return Attribute::Cold;
  case lumen::Attribute::InlineHint:
    return Attribute::InlineHint;
  case lumen::Attribute::MinSize:
    return Attribute::MinSize;
  case lumen::Attribute::Naked:
    return Attribute::Naked;
  case lumen::Attribute::NoAlias:
    return Attribute::NoAlias;
  case lumen::Attribute::NoCapture:
    return Attribute::NoCapture;
  case lumen::Attribute::NoInline:
    return Attribute::NoInline;
  case lumen::Attribute::NonNull:
    return Attribute::NonNull;
  case lumen::Attribute::NoRedZone:
    return Attribute::NoRedZone;
  case lumen::Attribute::NoReturn:
    return Attribute::NoReturn;
  case lumen::Attribute::NoUnwind:
    return Attribute::NoUnwind;
  case lumen::Attribute::OptimizeForSize:
    return Attribute::OptimizeForSize;
  case lumen::Attribute::ReadOnly:
    return Attribute::ReadOnly;
  case lumen::Attribute::ArgMemOnly:
    return Attribute::ArgMemOnly;
  case lumen::Attribute::SExt:
    return Attribute::SExt;
  case lumen::Attribute::StructRet:
    return Attribute::StructRet;
  case lumen::Attribute::UWTable:
    return Attribute::UWTable;
  case lumen::Attribute::ZExt:
    return Attribute::ZExt;
  case lumen::Attribute::InReg:
    return Attribute::InReg;
  case lumen::Attribute::SanitizeThread:
    return Attribute::SanitizeThread;
  case lumen::Attribute::SanitizeAddress:
    return Attribute::SanitizeAddress;
  case lumen::Attribute::SanitizeMemory:
    return Attribute::SanitizeMemory;
  case lumen::Attribute::NonLazyBind:
    return Attribute::NonLazyBind;
  case lumen::Attribute::OptimizeNone:
    return Attribute::OptimizeNone;
  case lumen::Attribute::ReturnsTwice:
    return Attribute::ReturnsTwice;
  }
  llvm::report_fatal_error("invalid attribute kind");
}

extern "C" void LLVMLumenAddCallSiteAttribute(LLVMValueRef instr, unsigned index,
                                              lumen::Attribute::AttrKind lumenAttr) {
  auto call = llvm::CallSite(unwrap<llvm::Instruction>(instr));
  auto attr = Attribute::get(call->getContext(), fromRust(lumenAttr));
  call.addAttribute(index, attr);
}

extern "C" void LLVMLumenAddAlignmentCallSiteAttr(LLVMValueRef instr,
                                                  unsigned index,
                                                  uint32_t bytes) {
  auto call = llvm::CallSite(unwrap<llvm::Instruction>(instr));
  AttrBuilder b;
  b.addAlignmentAttr(bytes);
  call.setAttributes(call.getAttributes().addAttributes(
      call->getContext(), index, b));
}

extern "C" void LLVMLumenAddDereferenceableCallSiteAttr(LLVMValueRef instr,
                                                        unsigned index,
                                                        uint64_t bytes) {
  auto call = llvm::CallSite(unwrap<llvm::Instruction>(instr));
  AttrBuilder b;
  b.addDereferenceableAttr(bytes);
  call.setAttributes(call.getAttributes().addAttributes(
      call->getContext(), index, b));
}

extern "C" void LLVMLumenAddDereferenceableOrNullCallSiteAttr(LLVMValueRef instr,
                                                              unsigned index,
                                                              uint64_t bytes) {
  auto call = llvm::CallSite(unwrap<llvm::Instruction>(instr));
  AttrBuilder b;
  b.addDereferenceableOrNullAttr(bytes);
  call.setAttributes(call.getAttributes().addAttributes(
      call->getContext(), index, b));
}

extern "C" void LLVMLumenAddByValCallSiteAttr(LLVMValueRef instr, unsigned index,
                                              LLVMTypeRef ty) {
  auto call = llvm::CallSite(unwrap<llvm::Instruction>(instr));
  Attribute attr = Attribute::getWithByValType(call->getContext(), unwrap(ty));
  call.addAttribute(index, attr);
}

extern "C" void LLVMLumenAddFunctionAttribute(LLVMValueRef fn, unsigned index,
                                              lumen::Attribute::AttrKind lumenAttr) {
  auto *f = unwrap<llvm::Function>(fn);
  Attribute attr = Attribute::get(f->getContext(), fromRust(lumenAttr));
  AttrBuilder b(attr);
  f->addAttributes(index, b);
}

extern "C" void LLVMLumenAddAlignmentAttr(LLVMValueRef fn,
                                          unsigned index,
                                          uint32_t bytes) {
  auto *f = unwrap<llvm::Function>(fn);
  AttrBuilder b;
  b.addAlignmentAttr(bytes);
  f->addAttributes(index, b);
}

extern "C" void LLVMLumenAddDereferenceableAttr(LLVMValueRef fn, unsigned index,
                                                uint64_t bytes) {
  auto *f = unwrap<llvm::Function>(fn);
  AttrBuilder b;
  b.addDereferenceableAttr(bytes);
  f->addAttributes(index, b);
}

extern "C" void LLVMLumenAddDereferenceableOrNullAttr(LLVMValueRef fn,
                                                      unsigned index,
                                                      uint64_t bytes) {
  auto *f = unwrap<llvm::Function>(fn);
  AttrBuilder b;
  b.addDereferenceableOrNullAttr(bytes);
  f->addAttributes(index, b);
}

extern "C" void LLVMLumenAddByValAttr(LLVMValueRef fn, unsigned index,
                                      LLVMTypeRef ty) {
  auto *f = unwrap<llvm::Function>(fn);
  Attribute attr = Attribute::getWithByValType(f->getContext(), unwrap(ty));
  f->addAttribute(index, attr);
}

extern "C" void LLVMLumenAddFunctionAttrStringValue(LLVMValueRef fn,
                                                    unsigned index,
                                                    const char *name,
                                                    const char *value) {
  auto *f = unwrap<llvm::Function>(fn);
  AttrBuilder b;
  b.addAttribute(name, value);
  f->addAttributes(index, b);
}

extern "C" void LLVMLumenRemoveFunctionAttributes(LLVMValueRef fn,
                                                  unsigned index,
                                                  lumen::Attribute::AttrKind lumenAttr) {
  auto *f = unwrap<llvm::Function>(fn);
  Attribute attr = Attribute::get(f->getContext(), fromRust(lumenAttr));
  AttrBuilder b(attr);
  auto attrList = f->getAttributes();
  auto newAttrList = attrList.removeAttributes(f->getContext(), index, b);
  f->setAttributes(newAttrList);
}

