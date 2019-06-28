// This file is based on the implementation of tools/lld/lld.cpp.
// It contains the main entry point the linker (i.e. `lld`), and handles
// dispatching to the platform-specific driver
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
// lld can be invoked as "lld" along with "-flavor" option. This is for
// backward compatibility and not recommended.
//
#include "LumenLinker.h"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#include "lld/Common/Driver.h"
#include "lld/Common/Memory.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SystemUtils.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/WithColor.h"
#pragma clang diagnostic pop
#include <cstdlib>

using namespace lumen;
using namespace lld;
using namespace llvm;
using namespace llvm::sys;

LLVM_ATTRIBUTE_NORETURN static void die(const Twine &s) {
    errs() << s << "\n";
    exit(1);
}

static cl::TokenizerCallback getDefaultQuotingStyle() {
    if (Triple(sys::getProcessTriple()).getOS() == Triple::Win32)
        return cl::TokenizeWindowsCommandLine;
    return cl::TokenizeGNUCommandLine;
}

static cl::list<std::string> InputFilenames(
    cl::Positional, 
    cl::OneOrMore,
    cl::desc("<input bitcode files>")
);

static cl::opt<std::string> OutputFilename(
    "o", 
    cl::desc("Override output filename"),
    cl::init("-"),
    cl::value_desc("filename")
);

static cl::opt<bool> OnlyNeeded("only-needed", cl::desc("Link only needed symbols"));
static cl::opt<bool> OutputAssembly("S", cl::desc("Write output as LLVM assembly"), cl::Hidden);
static cl::opt<bool> Verbose("v", cl::desc("Print information about actions taken"));
static cl::opt<bool> SuppressWarnings(
    "suppress-warnings", 
    cl::desc("Suppress all linking warnings"),
    cl::init(false)
);

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
    cl::ExpandResponseFiles(Saver, getDefaultQuotingStyle(), expandedArgs);
    for (auto it = expandedArgs.begin(); it + 1 != expandedArgs.end(); ++it) {
        if (StringRef(*it) != "-m")
            continue;
        return isPETargetName(*(it + 1));
    }
    return false;
}

// If this function returns true, lld calls _exit() so that it quickly
// exits without invoking destructors of globally allocated objects.
//
// We don't want to do that if we are running tests though, because
// doing that breaks leak sanitizer. So, lit sets this environment variable,
// and we use it to detect whether we are running tests or not.
static bool canExitEarly() { return StringRef(getenv("LLD_IN_TEST")) != "1"; }

/// Universal linker entry point. This linker emulates the gnu, darwin, or
/// windows linker based on the argv[0] or -flavor option.
bool lld(Flavor flavor, char** argv, int argc) {
    InitLLVM x(argc, argv);

    std::vector<const char *> args(argv, argv + argc);
    switch (flavor) {
        case Flavor::Gnu:
            if (isPETarget(args)) {
                return mingw::link(args);
            }
            return elf::link(args, canExitEarly());
        case Flavor::WinLink:
            return coff::link(args, canExitEarly());
        case Flavor::Darwin:
            return mach_o::link(args, canExitEarly());
        case Flavor::Wasm:
            return wasm::link(args, canExitEarly());
        default:
            die("Invalid linker flavor!");
    }
}

namespace {
struct LLVMLinkDiagnosticHandler : public DiagnosticHandler {
    bool handleDiagnostics(const DiagnosticInfo &di) override {
        unsigned severity = di.getSeverity();
        switch (severity) {
            case DS_Error:
                WithColor::error();
                break;
            case DS_Warning:
                if (SuppressWarnings)
                    return true;
                WithColor::warning();
                break;
            case DS_Remark:
            case DS_Note:
                llvm_unreachable("Only expecting warnings and errors");
        }

        DiagnosticPrinterRawOStream dp(errs());
        di.print(dp);
        errs() << '\n';
        return true;
    }
};
}

// Read the specified bitcode file in and return it. This routine searches the
// link path for the specified file to try to find it...
static std::unique_ptr<Module> loadFile(const char *argv0,
                                        const std::string &f,
                                        LLVMContext &context) {
    SMDiagnostic err;

    if (Verbose)
        errs() << "Loading '" << f << "'\n";

    std::unique_ptr<Module> result;
    result = parseIRFile(f, err, context);

    if (!result) {
        err.print(argv0, errs());
        return nullptr;
    }

    return result;
}

static bool linkFiles(const char *argv0, LLVMContext &context, Linker &l,
                      const cl::list<std::string> &files,
                      unsigned flags) {
    // Filter out flags that don't apply to the first file we load.
    unsigned applicable_flags = flags & Linker::Flags::OverrideFromSrc;

    for (const auto &file : files) {
        std::unique_ptr<Module> m = loadFile(argv0, file, context);
        if (!m.get()) {
            errs() << argv0 << ": ";
            WithColor::error() << " loading file '" << file << "'\n";
            return false;
        }

        if (Verbose)
            errs() << "Linking in '" << file << "'\n";

        if (l.linkInModule(std::move(m), applicable_flags))
            return false;

        // All linker flags apply to linking of subsequent files.
        applicable_flags = flags;
    }

    return true;
}

bool link(char** argv, int argc) {
    InitLLVM x(argc, argv);

    LLVMContext Context;
    Context.setDiagnosticHandler(llvm::make_unique<LLVMLinkDiagnosticHandler>(), true);
    Context.enableDebugTypeODRUniquing();

    cl::ParseCommandLineOptions(argc, argv, "llvm linker\n");

    auto composite = make_unique<Module>("llvm-link", Context);
    Linker l(*composite);

    unsigned flags = Linker::Flags::None;
    if (OnlyNeeded)
        flags |= Linker::Flags::LinkOnlyNeeded;

    // First add all the regular input files
    if (!linkFiles(argv[0], Context, l, InputFilenames, flags))
        return 1;

    std::error_code ec;
    ToolOutputFile out(OutputFilename, ec, sys::fs::F_None);
    if (ec) {
        WithColor::error() << ec.message() << '\n';
        return 1;
    }

    if (verifyModule(*composite, &errs())) {
        errs() << argv[0] << ": ";
        WithColor::error() << "linked module is broken!\n";
        return 1;
    }

    if (Verbose)
        errs() << "Writing bitcode...\n";

    if (OutputAssembly) {
        composite->print(out.os(), nullptr, false);
    } else if (!CheckBitcodeOutputToConsole(out.os(), true)) {
        WriteBitcodeToFile(*composite, out.os(), true);
    }

    // Declare success.
    out.keep();

    return 0;
}

bool lumen_lld(LinkerFlavor flavor, char** argv, int argc) {
    return lld(to_flavor(flavor), argv, argc);
}

bool lumen_link(char** argv, int argc) {
    return link(argv, argc);
}