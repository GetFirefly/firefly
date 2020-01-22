#![feature(main)]

use std::os::raw::c_uint;

use liblumen_core::atoms::ConstantAtom;

extern "C" {
    /// This symbol is defined in the compiled executable,
    /// and specifies the number of atoms in the atom table.
    #[link_name = "__LUMEN_ATOM_TABLE_SIZE"]
    static NUM_ATOMS: c_uint;

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
    static ATOM_TABLE: *const ConstantAtom;

    /// The target-defined entry point for the generated executable.
    ///
    /// Each target has its own runtime crate, which uses `#[entry]` from
    /// the `liblumen_core` crate to specify its entry point. This is
    /// called after the atom table, and any other core runtime functionality,
    /// is initialized and ready for use.
    #[link_name = "lumen_entry"]
    fn lumen_entry() -> i32;
}

#[link(name = "liblumen_alloc")]
extern "C" {
    /// This function is defined in `liblumen_alloc::erts::term::atom`
    fn InitializeLumenAtomTable(table: *const ConstantAtom, len: c_uint) -> bool;
}

/// The primary entry point for the Lumen runtime
///
/// This function is responsible for setting up any core functionality required
/// by the higher-level runtime, e.g. initializing the atom table. Once initialized,
/// this function invokes the platform-specific entry point which handles starting
/// up the schedulers and other high-level runtime functionality.
#[main]
pub fn lumen_start() -> i32 {
    // Initialize atom table
    if unsafe { InitializeLumenAtomTable(ATOM_TABLE, NUM_ATOMS) } == false {
        return 102;
    }

    // Invoke platform-specific entry point
    unsafe {
        lumen_entry()
    }
}
