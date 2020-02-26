#![feature(main)]

use std::panic;

mod atoms;
mod symbols;

extern "C" {
    /// The target-defined entry point for the generated executable.
    ///
    /// Each target has its own runtime crate, which uses `#[entry]` from
    /// the `liblumen_core` crate to specify its entry point. This is
    /// called after the atom table, and any other core runtime functionality,
    /// is initialized and ready for use.
    #[link_name = "lumen_entry"]
    fn lumen_entry() -> i32;
}

#[no_mangle]
pub extern "C" fn main(argc: i32, argv: *const *const std::os::raw::c_char) -> i32 {
    let exit_code = panic::catch_unwind(move || main_internal());
    exit_code.unwrap_or(101)
}

/// The primary entry point for the Lumen runtime
///
/// This function is responsible for setting up any core functionality required
/// by the higher-level runtime, e.g. initializing the atom table. Once initialized,
/// this function invokes the platform-specific entry point which handles starting
/// up the schedulers and other high-level runtime functionality.
#[main]
pub fn main_internal() -> i32 {
    use crate::atoms::*;
    use crate::symbols::*;

    // Initialize atom table
    if unsafe { InitializeLumenAtomTable(ATOM_TABLE, NUM_ATOMS) } == false {
        return 102;
    }

    // Initialize the dispatch table
    if unsafe { InitializeLumenDispatchTable(SYMBOL_TABLE, NUM_SYMBOLS) } == false {
        return 103;
    }

    // Invoke platform-specific entry point
    unsafe { lumen_entry() }
}
