#![feature(arbitrary_enum_discriminant)]
#![feature(associated_type_bounds)]
#![feature(try_blocks)]
#![feature(generic_associated_types)]
#![feature(let_else)]
#![feature(once_cell)]

pub mod linker;
pub mod meta;
pub mod passes;

use liblumen_llvm as llvm;
use liblumen_mlir as mlir;

/// Perform initialization of MLIR/LLVM for code generation
pub fn init(options: &liblumen_session::Options) -> anyhow::Result<()> {
    mlir::init(options)?;
    llvm::init(options)?;

    Ok(())
}
