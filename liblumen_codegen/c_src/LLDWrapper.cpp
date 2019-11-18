#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"
#include "lld/Common/Driver.h"
#pragma clang diagnostic pop

using namespace llvm;

extern "C" bool lumen_lld_elf_link(const char* argv[], unsigned argc) {
    return lld::elf::link(makeArrayRef(argv, argc), true, llvm::errs());
}

extern "C" bool lumen_lld_coff_link(const char* argv[], unsigned argc) {
    return lld::coff::link(makeArrayRef(argv, argc), true, llvm::errs());
}

extern "C" bool lumen_lld_mingw_link(const char* argv[], unsigned argc) {
    return lld::mingw::link(makeArrayRef(argv, argc));
}

extern "C" bool lumen_lld_mach_o_link(const char* argv[], unsigned argc) {
    return lld::mach_o::link(makeArrayRef(argv, argc), true, llvm::errs());
}

extern "C" bool lumen_lld_wasm_link(const char* argv[], unsigned argc) {
    return lld::wasm::link(makeArrayRef(argv, argc), true, llvm::errs());
}
