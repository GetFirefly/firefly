#![feature(extern_types)]
#![feature(arbitrary_enum_discriminant)]
#![feature(associated_type_bounds)]

pub mod builder;
pub mod generators;
pub mod linker;
pub mod meta;

pub use self::builder::GeneratedModule;

pub type Result<T> = std::result::Result<T, anyhow::Error>;

/// Perform initialization of MLIR/LLVM for code generation
pub fn init(options: &liblumen_session::Options) -> Result<()> {
    liblumen_llvm::init(options)?;
    liblumen_mlir::init(options)?;

    unsafe {
        MLIRRegisterDialects();
    }

    Ok(())
}

extern "C" {
    fn MLIRRegisterDialects();
}
