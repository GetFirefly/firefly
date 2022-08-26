mod adapter;
mod manager;

pub use self::adapter::*;
pub use self::manager::*;

use crate::codegen::{CodeGenOptLevel, CodeGenOptSize};

/// Initializes all LLVM/MLIR passes
pub fn init() {
    extern "C" {
        fn LLVMFireflyInitializePasses();
    }
    unsafe {
        LLVMFireflyInitializePasses();
    }
}

/// Prints all of the currently available LLVM/MLIR passes
///
/// NOTE: Can be called without initializing LLVM
pub fn print() {
    extern "C" {
        fn LLVMFireflyPrintPasses();
    }
    unsafe {
        LLVMFireflyPrintPasses();
    }
}

/// Represents a combined optimization level (speed + size) for a pass builder
///
/// By default no optimizations are enabled, i.e. -O0
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum PassBuilderOptLevel {
    O0 = 0,
    O1,
    O2,
    O3,
    Os,
    Oz,
}
impl Default for PassBuilderOptLevel {
    fn default() -> Self {
        Self::O0
    }
}
impl PassBuilderOptLevel {
    pub fn from_codegen_opts(speed: CodeGenOptLevel, size: CodeGenOptSize) -> Self {
        match size {
            CodeGenOptSize::Default if speed > CodeGenOptLevel::None => Self::Os,
            CodeGenOptSize::Aggressive if speed > CodeGenOptLevel::None => Self::Oz,
            _ => match speed {
                CodeGenOptLevel::Less => Self::O1,
                CodeGenOptLevel::Default => Self::O2,
                CodeGenOptLevel::Aggressive => Self::O3,
                _ => Self::O0,
            },
        }
    }
}

/// Represents the link-time optimizer stage to run
///
/// By default the stage is PreLinkNoLTO
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum OptStage {
    /// No link-time optimization needed/enabled
    PreLinkNoLTO,
    /// Performs optimizations/transformations that leaves IR untouched which
    /// is better optimized during the link-time optimization phase
    PreLinkThinLTO,
    /// Same as PreLinkThinLTO, but oriented towards full LTO
    PreLinkFatLTO,
    /// Runs link-time optimization passes for a ThinLTO build
    ThinLTO,
    /// Runs link-time optimization passes for a full LTO build
    FatLTO,
}
impl Default for OptStage {
    fn default() -> Self {
        Self::PreLinkNoLTO
    }
}

/// Represents the set of sanitizers that are enabled for a pass pipeline
///
/// By default no sanitizers are enabled.
#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct SanitizerOptions {
    memory: bool,
    thread: bool,
    address: bool,
    recover: bool,
    memory_track_origins: u32,
}

/// The type of the callback invoked before each pass for profiling purposes
pub type SelfProfileBeforePassCallback = unsafe extern "C" fn(
    *mut std::ffi::c_void,
    *const std::os::raw::c_char,
    *const std::os::raw::c_char,
);

/// The type of the callback invoked after each pass for profiling purposes
pub type SelfProfileAfterPassCallback = unsafe extern "C" fn(*mut std::ffi::c_void);
