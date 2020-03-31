extern "C" {
    pub fn LLVMLumenInitializePasses();
    pub fn LLVMLumenPrintPasses();
}

/// Initializes all LLVM/MLIR passes
pub fn init() {
    unsafe {
        LLVMLumenInitializePasses();
    }
}

/// Prints all of the currently available LLVM/MLIR passes
///
/// NOTE: Can be called without initializing LLVM
pub fn print() {
    unsafe {
        LLVMLumenPrintPasses();
    }
}
