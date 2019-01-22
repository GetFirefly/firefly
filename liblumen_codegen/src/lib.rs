#[macro_use]
pub mod macros;

mod linker;
mod lld;
mod llvm;

use failure::{Error, Fail};

use liblumen_syntax::ast::Module;

pub use self::lld::link;
pub use self::llvm::OutputType;

pub type CodegenResult = Result<(), Error>;

/// Represents an error which occurs during code generation
#[derive(Fail, Debug)]
pub enum CodeGenError {
    #[fail(display = "invalid codegen input: {}", _0)]
    ValidationError(String),

    #[fail(display = "linker error: {}", _0)]
    LinkerError(String),

    #[fail(display = "llvm error: {}", _0)]
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

/// Initializes LLVM for use by the code generator
pub fn initialize() {
    llvm::initialize();
}

/// Runs the code generator using the given set of modules and selected output type
pub fn run(_mods: Vec<Module>, _type: OutputType) -> CodegenResult {
    Ok(())
}
