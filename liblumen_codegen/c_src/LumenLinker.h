#ifndef LUMEN_LINKER_H
#define LUMEN_LINKER_H

#ifdef __cplusplus
extern "C" {
#endif
    typedef enum {
        LinkerFlavorGnu,
        LinkerFlavorWinLink,
        LinkerFlavorDarwin,
        LinkerFlavorWasm
    } LinkerFlavor;

    /**
     * Invokes the native linker entry point with the given argv/argc
     * 
     * This in turn handles invoking `lld` with the proper parameters
     * for the desired target and output type
     */
    bool lumen_lld(LinkerFlavor flavor, char** argv, int argc);

    /**
     * Invokes the LLVM IR linker on the given input files, which
     * may be either LLVM bitcode or assembly, and writes the resulting 
     * module to `outfile`. If `to_assembly` is true, the output is LLVM
     * assembly rather than LLVM bitcode.
     * 
     * The output of this linking phase can be used as an input to LLD
     */
    bool lumen_link(char** argv, int argc);
#ifdef __cplusplus
}

namespace lumen {
    enum Flavor {
        Invalid,
        Gnu,     // -flavor gnu
        WinLink, // -flavor link
        Darwin,  // -flavor darwin
        Wasm,    // -flavor wasm
    };

    inline Flavor to_flavor(LinkerFlavor flavor) {
        switch (flavor) {
            case LinkerFlavorGnu:
                return Flavor::Gnu;
            case LinkerFlavorWinLink:
                return Flavor::WinLink;
            case LinkerFlavorDarwin:
                return Flavor::Darwin;
            case LinkerFlavorWasm:
                return Flavor::Wasm;
            default:
                return Flavor::Invalid;
        }
    }
} // end of namespace lumen
#endif

#endif