#![feature(rustc_attrs)]
#![feature(c_unwind)]
#![feature(never_type)]

use core::panic::RefUnwindSafe;

mod atoms;
mod symbols;

extern "Rust" {
    /// The target-defined entry point for the generated executable.
    ///
    /// Each target has its own runtime crate, with its own entry point
    /// exposed as `firefly_entry`, the core runtime (this crate) calls
    /// into that entry after global initialization of target-agnostic
    /// resources is complete.
    #[link_name = "firefly_entry"]
    fn firefly_entry() -> i32;

    #[link_name = env!("LANG_START_SYMBOL_NAME")]
    fn lang_start(
        main: &(dyn Fn() -> i32 + Sync + RefUnwindSafe),
        argc: isize,
        argv: *const *const i8,
        sigpipe: u8,
    ) -> Result<isize, !>;
}

#[no_mangle]
pub extern "C" fn main(argc: i32, argv: *const *const std::os::raw::c_char) -> i32 {
    const SIG_IGN: u8 = 0;
    unsafe {
        match lang_start(&move || main_internal(), argc as isize, argv, SIG_IGN) {
            Ok(v) => v as i32,
            _ => unreachable!(),
        }
    }
}

/// The primary entry point for the Firefly runtime
///
/// This function is responsible for setting up any core functionality required
/// by the higher-level runtime, e.g. initializing the atom table. Once initialized,
/// this function invokes the platform-specific entry point which handles starting
/// up the schedulers and other high-level runtime functionality.
#[rustc_main]
pub fn main_internal() -> i32 {
    // Initialize atom table
    if unsafe { atoms::init(atoms::start(), atoms::end()) } == false {
        return 102;
    }

    // Initialize the dispatch table
    if unsafe { symbols::init(symbols::start(), symbols::end()) } == false {
        return 103;
    }

    // Invoke platform-specific entry point
    unsafe { firefly_entry() }
}
