pub mod target;
pub mod passes;
pub mod diagnostics;
pub mod string;
pub mod util;

pub use self::util::init;

pub type Value = llvm_sys::LLVMValue;

use std::fmt;

#[derive(Copy, Clone, PartialEq)]
#[repr(C)]
#[allow(dead_code)] // Variants constructed by C++.
pub enum LLVMLumenResult {
    Success,
    Failure,
}
impl LLVMLumenResult {
    pub fn into_result(self) -> Result<(), ()> {
        match self {
            Self::Success => Ok(()),
            Self::Failure => Err(()),
        }
    }
}

/// LLVMLumenFileType
#[derive(Copy, Clone)]
#[repr(C)]
pub enum FileType {
    #[allow(dead_code)]
    Other,
    AssemblyFile,
    ObjectFile,
}

/// LLVMCodeGenOptLevel
#[derive(Debug, Copy, Clone, PartialEq)]
#[repr(C)]
pub enum CodeGenOptLevel {
    Other,
    None,
    Less,
    Default,
    Aggressive,
}
impl fmt::Display for CodeGenOptLevel {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Other => f.write_str("other"),
            Self::None => f.write_str("none"),
            Self::Less => f.write_str("less"),
            Self::Default => f.write_str("default"),
            Self::Aggressive => f.write_str("aggressive"),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
#[repr(C)]
pub enum CodeGenOptSize {
    Other,
    None,
    Default,
    Aggressive,
}
impl fmt::Display for CodeGenOptSize {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Other => f.write_str("other"),
            Self::None => f.write_str("none"),
            Self::Default => f.write_str("default"),
            Self::Aggressive => f.write_str("aggressive"),
        }
    }
}
