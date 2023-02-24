// Used to represent FFI-opaque types
#![feature(extern_types)]
// Allow overlapping implementations of certain IR traits
#![feature(min_specialization)]

mod diagnostics;
mod dialect;
mod ir;
mod pass;
mod support;

pub use self::dialect::*;
pub use self::ir::*;
pub use self::pass::*;
pub use self::support::{LogicalResult, OwnedStringRef, StringRef};

/// Initializes various MLIR subystems
///
/// * Registers all MLIR-specific command-line options provided to the compiler
/// * Registers all MLIR built-in passes
///
/// NOTE: It is important this is called before invoking any MLIR APIs, as it
/// guarantees that MLIR is properly configured. Without this, MLIR may behave
/// differently than expected, or raise an error due to unregistered passes.
pub fn init(_options: &firefly_session::Options) -> anyhow::Result<()> {
    extern "C" {
        #[link_name = "mlirRegisterCommandLineOptions"]
        fn register_mlir_cli_options();
    }

    unsafe {
        register_mlir_cli_options();
    }

    Ok(())
}
