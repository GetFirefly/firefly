#![feature(extern_types)]
#![feature(const_cstr_unchecked)]
#![feature(crate_visibility_modifier)]

pub mod archives;
pub mod attributes;
pub mod builder;
pub mod config;
pub mod context;
pub mod diagnostics;
pub mod enums;
pub mod funclet;
pub mod module;
pub mod passes;
pub mod profiling;
pub mod sys;
pub mod target;
pub mod utils;

pub use self::context::{Context, ContextRef};
pub use self::module::{Module, ModuleRef};

use std::sync::Once;

use anyhow::anyhow;

use liblumen_session::Options;

pub type Block = *mut crate::sys::LLVMBasicBlock;
pub type Type = *mut crate::sys::LLVMType;
pub type Value = *mut crate::sys::LLVMValue;
pub type Metadata = *mut crate::sys::LLVMOpaqueMetadata;

pub type Result<T> = std::result::Result<T, anyhow::Error>;

static INIT: Once = Once::new();

extern "C" {
    pub fn LLVMLumenVersionMajor() -> u32;
    pub fn LLVMLumenVersionMinor() -> u32;
}

/// Returns the current version of LLVM
///
/// NOTE: Can be called without initializing LLVM
pub fn version() -> String {
    unsafe { format!("{}.{}", LLVMLumenVersionMajor(), LLVMLumenVersionMinor()) }
}

/// Performs one-time initialization of LLVM
///
/// This should be called at program startup
pub fn init(options: &Options) -> anyhow::Result<()> {
    let mut is_multithreaded = true;
    unsafe {
        // Before we touch LLVM, make sure that multithreading is enabled.
        INIT.call_once(|| {
            if crate::sys::core::LLVMIsMultithreaded() == 1 {
                // Initialize diagnostics handlers
                diagnostics::init();
                // Initialize all passes
                passes::init();
                // Initialize all targets
                target::init();
                // Configure LLVM
                config::init(options);
            } else {
                is_multithreaded = false;
            }
        });
    }
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

/// A result type for use with LLVM APIs
#[derive(Copy, Clone, PartialEq)]
#[repr(C)]
#[allow(dead_code)] // Variants constructed by C++.
pub enum LLVMResult {
    Success,
    Failure,
}
impl LLVMResult {
    pub fn into_result(self) -> std::result::Result<(), ()> {
        match self {
            Self::Success => Ok(()),
            Self::Failure => Err(()),
        }
    }
}
