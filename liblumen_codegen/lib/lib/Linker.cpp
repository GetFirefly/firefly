//===----------------------------------------------------------------------===//
//
// This file is based on the entry point for lld, and contains the driver
// function which executes the linked lld library.
//
// lld is a single executable that contains four different linkers for ELF,
// COFF, WebAssembly and Mach-O. The main function dispatches according to
// argv[0] (i.e. command name). The most common name for each target is shown
// below:
//
//  - ld.lld:    ELF (Unix)
//  - ld64:      Mach-O (macOS)
//  - lld-link:  COFF (Windows)
//  - ld-wasm:   WebAssembly
//
// Lumen invokes lld by invoking this entry point with the correct name for
// the target platform.
//===----------------------------------------------------------------------===//

#include "lld/Common/Driver.h"
#include "lld/Common/Memory.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include <cstdlib>

using namespace lld;
using namespace llvm;
using namespace llvm::sys;

enum Flavor {
  Invalid,
  Gnu,     // -flavor gnu
  WinLink, // -flavor link
  Darwin,  // -flavor darwin
  Wasm,    // -flavor wasm
};

static Flavor getFlavor(StringRef s) {
  return StringSwitch<Flavor>(s)
      .CasesLower("ld", "ld.lld", "gnu", Gnu)
      .CasesLower("wasm", "ld-wasm", Wasm)
      .CaseLower("link", WinLink)
      .CasesLower("ld64", "ld64.lld", "darwin", Darwin)
      .Default(Invalid);
}

static cl::TokenizerCallback getDefaultQuotingStyle() {
  if (Triple(sys::getProcessTriple()).getOS() == Triple::Win32)
    return cl::TokenizeWindowsCommandLine;
  return cl::TokenizeGNUCommandLine;
}

static bool isPETargetName(StringRef s) {
  return s == "i386pe" || s == "i386pep" || s == "thumb2pe" || s == "arm64pe";
}

static bool isPETarget(std::vector<const char *> &v) {
  for (auto it = v.begin(); it + 1 != v.end(); ++it) {
    if (StringRef(*it) != "-m")
      continue;
    return isPETargetName(*(it + 1));
  }
  // Expand response files (arguments in the form of @<filename>)
  // to allow detecting the -m argument from arguments in them.
  SmallVector<const char *, 256> expandedArgs(v.data(), v.data() + v.size());
  cl::ExpandResponseFiles(saver, getDefaultQuotingStyle(), expandedArgs);
  for (auto it = expandedArgs.begin(); it + 1 != expandedArgs.end(); ++it) {
    if (StringRef(*it) != "-m")
      continue;
    return isPETargetName(*(it + 1));
  }
  return false;
}

static Flavor parseProgname(StringRef progname) {
#if __APPLE__
  // Use Darwin driver for "ld" on Darwin.
  if (progname == "ld")
    return Darwin;
#endif

#if LLVM_ON_UNIX
  // Use GNU driver for "ld" on other Unix-like system.
  if (progname == "ld")
    return Gnu;
#endif

  // Progname may be something like "lld-gnu". Parse it.
  SmallVector<StringRef, 3> v;
  progname.split(v, "-");
  for (StringRef s : v)
    if (Flavor f = getFlavor(s))
      return f;
  return Invalid;
}

static Flavor parseFlavor(std::vector<const char *> &v) {
  // Parse -flavor option.
  if (v.size() > 1 && v[1] == StringRef("-flavor")) {
    if (v.size() <= 2) {
      llvm::errs() << "missing linker arg value for '-flavor'"
                   << "\n";
      return Flavor::Invalid;
    }
    Flavor f = getFlavor(v[2]);
    if (f == Invalid) {
      llvm::errs() << "unknown linker flavor: " + StringRef(v[2]) << "\n";
      return f;
    }
    v.erase(v.begin() + 1, v.begin() + 3);
    return f;
  }

  // Deduct the flavor from argv[0].
  StringRef arg0 = path::filename(v[0]);
  if (arg0.endswith_lower(".exe"))
    arg0 = arg0.drop_back(4);
  return parseProgname(arg0);
}

// If this function returns true, lld calls _exit() so that it quickly
// exits without invoking destructors of globally allocated objects.
static bool canExitEarly() { return false; }

/// Universal linker main(). This linker emulates the gnu, darwin, or
/// windows linker based on the argv[0] or -flavor option.
extern "C" int LLVMLumenLink(int argc, const char **argv) {
  std::vector<const char *> args(argv, argv + argc);
  switch (parseFlavor(args)) {
  case Gnu:
    if (isPETarget(args))
      return !mingw::link(args, canExitEarly(), llvm::outs(), llvm::errs());
    return !elf::link(args, canExitEarly(), llvm::outs(), llvm::errs());
  case WinLink:
    return !coff::link(args, canExitEarly(), llvm::outs(), llvm::errs());
  case Darwin:
    return !mach_o::link(args, canExitEarly(), llvm::outs(), llvm::errs());
  case Wasm:
    return !wasm::link(args, canExitEarly(), llvm::outs(), llvm::errs());
  default:
    llvm::errs() << "Compiler provided invalid linker name. This is a bug!"
                 << "\n";
    return 1;
  }
}
