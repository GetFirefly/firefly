mod lld;
mod llvm;

use std::path::PathBuf;

use super::config::Config;

#[derive(Debug)]
pub enum LinkerError {
    LinkingFailed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputType {
    DynamicLibrary,
    DynamicExecutable,
    MainExecutable,
    Preload,
    RelocatableObject,
    StaticExecutable,
}

/// Links LLVM assembly/bitcode files into a single object file
pub fn link(inputs: Vec<PathBuf>, config: Config) -> Result<(), LinkerError> {
    use self::llvm::Linker;

    let mut linker = Linker::new(config);
    linker.add_inputs(inputs);
    linker.link()
}

/// Links object files together and generates a native executable/library
pub fn link_native(inputs: Vec<PathBuf>, config: Config) -> Result<(), LinkerError> {
    use self::lld::NativeLinker;

    let mut linker = NativeLinker::new(config);
    linker.add_inputs(inputs);
    linker.link()
}