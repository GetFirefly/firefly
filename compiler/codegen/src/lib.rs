#![feature(extern_types)]
#![feature(arbitrary_enum_discriminant)]
#![feature(associated_type_bounds)]
#![feature(try_blocks)]
#![feature(option_result_contains)]
#![feature(or_patterns)]

pub mod builder;
pub mod generators;
pub mod linker;
pub mod meta;

use liblumen_llvm as llvm;
use liblumen_mlir as mlir;

pub use self::builder::GeneratedModule;

pub type Result<T> = std::result::Result<T, anyhow::Error>;

/// Perform initialization of MLIR/LLVM for code generation
pub fn init(options: &liblumen_session::Options) -> Result<()> {
    mlir::init(options)?;
    llvm::init(options)?;

    Ok(())
}
