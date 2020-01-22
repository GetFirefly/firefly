#include "lumen/Support/RustString.h"

extern "C" void LLVMRustStringWriteImpl(RustStringRef Str, const char *Ptr,
                                        size_t Size);

void RawRustStringOstream::write_impl(const char *Ptr, size_t Size) {
    LLVMRustStringWriteImpl(Str, Ptr, Size);
    Pos += Size;
}
