pub mod conversions;
mod manager;
mod options;
pub mod transforms;
pub mod translations;

pub use self::manager::{OpPassManager, PassManager};
pub use self::options::{OpPrintingFlags, PassManagerOptions};

use crate::*;

extern "C" {
    type MlirPass;
}

/// The base trait for all owned MLIR passes
pub trait Pass {
    /// Returns an opaque borrowed reference to the pass for use in FFI
    fn base(&self) -> PassBase;

    /// Consumes this pass and returns an opaque handle representing the pass for use by a pass manager
    fn to_owned(self) -> OwnedPass;
}

/// A marker trait for passes which run on a specific type of operation
pub trait OpPass<T: Operation>: Pass {}

/// This type is an opaque handle for passes obtained via MLIR's C API
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct PassBase(*mut MlirPass);
impl std::fmt::Pointer for PassBase {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:p}", self.0)
    }
}

/// Represents a pass whose ownership was handed over to Rust
///
/// Passes are generally short-lived, i.e. they are registered, created,
/// then added to a PassManager, at which point ownership is transfered.
///
/// In the rare case that a pass is created, but not used, we've implemented
/// `Drop` to ensure passes still get cleaned up properly.
#[repr(transparent)]
pub struct OwnedPass(PassBase);
impl Pass for OwnedPass {
    #[inline]
    fn base(&self) -> PassBase {
        self.0
    }

    #[inline(always)]
    fn to_owned(self) -> OwnedPass {
        self
    }
}
impl OwnedPass {
    pub fn release(self) -> PassBase {
        self.0
    }
}
impl std::fmt::Pointer for OwnedPass {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:p}", self.0)
    }
}
impl Drop for OwnedPass {
    fn drop(&mut self) {
        extern "C" {
            #[link_name = "mlirPassDestroy"]
            fn mlir_pass_destroy(pass: PassBase);
        }

        unsafe { mlir_pass_destroy(self.0) }
    }
}
