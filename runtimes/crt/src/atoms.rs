//use std::os::raw::c_uint;

//use liblumen_core::atoms::ConstantAtom;

extern "C" {
    /// This symbol is defined by the linker, and indicates the start of the atom strings section
    #[link_name = "__atoms_start"]
    pub static ATOMS_START: *const std::os::raw::c_char;

    /// This symbol is defined by the linker, and indicates the end of the atom strings section
    #[link_name = "__atoms_end"]
    pub static ATOMS_END: *const std::os::raw::c_char;

    // This symbol is defined in the compiled executable,
    // and specifies the number of atoms in the atom table.
    //#[link_name = "__LUMEN_ATOM_TABLE_SIZE"]
    //pub static NUM_ATOMS: c_uint;

    // This symbol is defined in the compiled executable,
    // and provides a pointer to the atom table, or more specifically,
    // a pointer to the first pointer in the atom table. The atom table
    // is an array of pointers to null-terminated strings, each of which
    // is an atom.
    //
    // Combined with `NUM_ATOMS`, this can be used to obtain a slice
    // of pointers, which can each be turned into a `CStr` with static
    // lifetime.
    //#[link_name = "__LUMEN_ATOM_TABLE"]
    //pub static ATOM_TABLE: *const ConstantAtom;
}

extern "C-unwind" {
    /// This function is defined in `liblumen_alloc::erts::term::atom`
    //pub fn InitializeLumenAtomTable(table: *const ConstantAtom, len: c_uint) -> bool;
    pub fn InitializeLumenAtomTable(
        start: *const std::os::raw::c_char,
        end: *const std::os::raw::c_char,
    ) -> bool;
}
