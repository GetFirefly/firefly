#[macro_use]
pub mod macros;

mod llvm;
mod linker;

use std::path::Path;

use crate::syntax::ast::ast::ModuleDecl;

pub use self::llvm::OutputType;

/// Represents an error which occurs during code generation
#[derive(Debug)]
pub enum CodeGenError {
    ValidationError(String),
    LinkerError(String),
    LLVMError(String),
}
impl CodeGenError {
    pub fn llvm(reason: &str) -> CodeGenError {
        CodeGenError::LLVMError(reason.to_string())
    }
    pub fn invalid(reason: &str) -> CodeGenError {
        CodeGenError::ValidationError(reason.to_string())
    }
}
impl std::fmt::Display for CodeGenError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        use self::CodeGenError::*;
        match *self {
            ValidationError(ref e) => write!(f, "invalid codegen input: {}", e),
            LLVMError(ref e) => write!(f, "LLVM failed: {}", e),
            LinkerError(ref e) => write!(f, "Linker failed: {}", e)
        }
    }
}
impl std::error::Error for CodeGenError {
    fn description(&self) -> &str {
        use self::CodeGenError::*;
        match *self {
            ValidationError(ref e) => e,
            LLVMError(ref e) => e,
            LinkerError(ref e) => e
        }
    }
}

pub fn initialize() {
    llvm::initialize();
}

pub fn generate_to_file(
    _mods: &[ModuleDecl],
    _path: &Path,
    _output_type: OutputType,
) -> Result<(), CodeGenError> {
    Ok(())
}
