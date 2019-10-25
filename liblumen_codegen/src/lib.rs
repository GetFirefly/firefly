//mod linker;
mod llvm;
mod lower;
mod config;

use std::path::PathBuf;

use thiserror::Error;
pub use anyhow::Result;

use libeir_ir::Module;

pub use self::llvm::enums::{OptimizationLevel, RelocMode, CodeModel, OutputType};
pub use self::config::{ConfigBuilder, Config};

/// Represents an error which occurs during code generation
#[derive(Error, Debug)]
pub enum CodeGenError {
    #[error("invalid target: {0}")]
    InvalidTarget(String),

    #[error("invalid codegen input: {0}")]
    ValidationError(String),

    #[error("linker error: {0}")]
    LinkerError(String),

    #[error("llvm error: {0}")]
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
