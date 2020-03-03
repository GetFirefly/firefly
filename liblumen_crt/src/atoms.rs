use std::os::raw::c_uint;

use liblumen_core::atoms::ConstantAtom;

extern "C" {
    /// This symbol is defined in the compiled executable,
    /// and specifies the number of atoms in the atom table.
    #[link_name = "__LUMEN_ATOM_TABLE_SIZE"]
    pub static NUM_ATOMS: c_uint;

    /// This symbol is defined in the compiled executable,
    /// and provides a pointer to the atom table, or more specifically,
    /// a pointer to the first pointer in the atom table. The atom table
    /// is an array of pointers to null-terminated strings, each of which
    /// is an atom.
    ///
    /// Combined with `NUM_ATOMS`, this can be used to obtain a slice
    /// of pointers, which can each be turned into a `CStr` with static
    /// lifetime.
    #[link_name = "__LUMEN_ATOM_TABLE"]
    pub static ATOM_TABLE: *const ConstantAtom;
}

#[link(name = "liblumen_alloc")]
extern "C" {
    /// This function is defined in `liblumen_alloc::erts::term::atom`
    pub fn InitializeLumenAtomTable(table: *const ConstantAtom, len: c_uint) -> bool;
}
