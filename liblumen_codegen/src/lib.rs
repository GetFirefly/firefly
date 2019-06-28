#[macro_use]
pub mod macros;

//mod linker;
mod llvm;
mod lower;
mod config;

use core::fmt;

use failure::{Error, Fail};
use inkwell::targets::Target;

use libeir_ir::Module;

pub use self::llvm::enums::{OptimizationLevel, RelocMode, CodeModel, OutputType};
pub use self::config::{ConfigBuilder, Config};

pub type Result<T> = core::result::Result<T, Error>;

/// Represents an error which occurs during code generation
#[derive(Fail, Debug)]
pub enum CodeGenError {
    #[fail(display = "invalid target: {}", _0)]
    InvalidTarget(String),

    #[fail(display = "invalid codegen input: {}", _0)]
    ValidationError(String),

    #[fail(display = "linker error: {}", _0)]
    LinkerError(String),

    #[fail(display = "llvm error: {}", _0)]
    LLVMError(String),
}
impl CodeGenError {
    pub fn llvm(reason: &str) -> Self {
        CodeGenError::LLVMError(reason.to_string())
    }
    pub fn invalid(reason: &str) -> Self {
        CodeGenError::ValidationError(reason.to_string())
    }

    pub fn no_target_machine(triple: String, cpu: String, features: String) -> Self {
        let mut message = format!("configured target has no backend: {}", triple);
        if !cpu.is_empty() && !features.is_empty() {
            message = format!("{} (cpu = \"{}\", features = \"{}\")", message, cpu, features);
        } else if !cpu.is_empty() {
            message = format!("{} (cpu = \"{}\")", message, cpu);
        } else if !features.is_empty() {
            message = format!("{} (features = \"{}\")", message, features);
        }
        CodeGenError::InvalidTarget(message.to_owned())
    }
}

/// Runs the code generator against the given set of modules with the selected output type
pub fn run(modules: Vec<Module>, config: &Config) -> Result<()> {
    // Lower EIR modules to LLVM modules compiled to bitcode or assembly
    let lls = modules.iter()
        .map(|m| lower::module(m, config))
        .collect::<Vec<PathBuf>>();
    // Link together LLVM assembly/bitcode files into an object file
    //let obj = linker::link(lls, config)?;
    // Perform native object file linking and generation
    //let _bin = linker::link_native(obj, config)?;

    Ok(())
}
