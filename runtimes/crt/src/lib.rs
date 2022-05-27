#![feature(rustc_attrs)]
#![feature(c_unwind)]

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

    #[allow(improper_ctypes)]
    #[link_name = env!("LANG_START_SYMBOL_NAME")]
    fn lang_start(main: &dyn Fn() -> i32, argc: isize, argv: *const *const i8) -> isize;
}

#[no_mangle]
pub extern "C" fn main(argc: i32, argv: *const *const std::os::raw::c_char) -> i32 {
    unsafe { lang_start(&move || main_internal(), argc as isize, argv) as i32 }
}

/// The primary entry point for the Lumen runtime
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
    unsafe { lumen_entry() }
}
