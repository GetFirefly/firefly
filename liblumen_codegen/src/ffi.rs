pub mod target;
pub mod passes;
pub mod diagnostics;
pub mod string;
pub mod util;

pub use self::util::init;

pub type Value = llvm_sys::LLVMValue;

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
#[derive(Copy, Clone, PartialEq)]
#[repr(C)]
pub enum CodeGenOptLevel {
    #[allow(dead_code)]
    Other,
    None,
    Less,
    Default,
    Aggressive,
}

#[derive(Copy, Clone, PartialEq)]
#[repr(C)]
pub enum CodeGenOptSize {
    Other,
    None,
    Default,
    Aggressive,
}
