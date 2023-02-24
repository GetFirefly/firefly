#![deny(warnings)]
// Used for the FFI bridge
#![feature(extern_types)]

pub mod archives;
pub mod builder;
pub mod cli;
pub mod codegen;
pub mod debuginfo;
pub mod diagnostics;
pub mod ir;
//pub mod jit;
pub mod passes;
pub mod profiling;
pub mod support;
pub mod target;

pub use self::ir::*;
pub use self::support::{OwnedStringRef, StringRef};

use std::sync::Once;

use anyhow::anyhow;
use firefly_session::Options;

static INIT: Once = Once::new();

/// Performs one-time initialization of LLVM
///
/// This should be called at program startup
pub fn init(options: &Options) -> anyhow::Result<()> {
    extern "C" {
        fn LLVMIsMultithreaded() -> bool;
    }

    let mut is_multithreaded = true;
    // Before we touch LLVM, make sure that multithreading is enabled.
    INIT.call_once(|| {
        if unsafe { LLVMIsMultithreaded() } {
            // Initialize diagnostics handlers
            diagnostics::init();
            // Initialize all passes
            passes::init();
            // Initialize all targets
            target::init();
            // Configure LLVM via command line options
            cli::init(options);
        } else {
            is_multithreaded = false;
        }
    });
    if is_multithreaded {
        Ok(())
    } else {
        Err(anyhow!(
            "expected LLVM to be compiled with multithreading enabled!"
        ))
    }
}

/// Ensures a panic is raised if LLVM has not been initialized in this function
pub(crate) fn require_inited() {
    INIT.call_once(|| panic!("LLVM is not initialized"));
}
