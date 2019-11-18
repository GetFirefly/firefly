extern "C" {
    // Invokes the IR/bitcode linker
    pub fn lumen_link(argv: *const *const libc::c_char, argc: libc::c_int) -> bool;
}