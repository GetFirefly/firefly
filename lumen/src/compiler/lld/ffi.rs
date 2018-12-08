extern "C" {
    pub fn lumen_lld_elf_link(argv: *const *const libc::c_char, argc: libc::c_uint) -> bool;
    pub fn lumen_lld_coff_link(argv: *const *const libc::c_char, argc: libc::c_uint) -> bool;
    pub fn lumen_lld_mingw_link(argv: *const *const libc::c_char, argc: libc::c_uint) -> bool;
    pub fn lumen_lld_mach_o_link(argv: *const *const libc::c_char, argc: libc::c_uint) -> bool;
    pub fn lumen_lld_wasm_link(argv: *const *const libc::c_char, argc: libc::c_uint) -> bool;
}
