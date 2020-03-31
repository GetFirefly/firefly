#include "lumen/llvm/RustString.h"

#include "llvm-c/Types.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/CBindingWrapping.h"

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(llvm::Twine, LLVMTwineRef);
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(llvm::Value, LLVMValueRef);
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(llvm::Type, LLVMTypeRef);

extern "C" void LLVMRustStringWriteImpl(RustStringRef Str, const char *Ptr,
                                        size_t Size);

void RawRustStringOstream::write_impl(const char *Ptr, size_t Size) {
  LLVMRustStringWriteImpl(Str, Ptr, Size);
  Pos += Size;
}

extern "C" void LLVMLumenWriteTwineToString(LLVMTwineRef T, RustStringRef Str) {
  RawRustStringOstream OS(Str);
  unwrap(T)->print(OS);
}

extern "C" void LLVMLumenWriteTypeToString(LLVMTypeRef Ty, RustStringRef Str) {
  RawRustStringOstream OS(Str);
  llvm::Type *ty = unwrap(Ty);
  ty->print(OS);
}

extern "C" void LLVMLumenWriteValueToString(LLVMValueRef V, RustStringRef Str) {
  RawRustStringOstream OS(Str);
  if (!V) {
    OS << "(null)";
  } else {
    llvm::Value *val = unwrap(V);
    OS << "(";
    val->getType()->print(OS);
    OS << ":";
    val->print(OS);
    OS << ")";
  }
}
